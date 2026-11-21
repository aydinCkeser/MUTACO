"""Retained mixed-precision oracle identity and Torch regression path."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.stats import gamma, norm
import torch

from mutaco.simulator import (
    SimulationContext,
    SimulationResult,
    SimulatorCapabilities,
    TensorContractError,
)

from .profiles import (
    RETAINED_M12,
    RETAINED_ORACLE_IDENTITY,
    parameter_profile,
)
from .reference import RetainedGraphFixture
from .scientific import GraphRun


@dataclass(frozen=True)
class RetainedOracleConfig:
    node_count: int = 6
    batch_size: int = 2
    repeat_size: int = 1
    temperature: float = 0.15
    dt: float = 0.1
    steps: int = 50

    def __post_init__(self) -> None:
        for name in ("node_count", "batch_size", "repeat_size", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dt <= 0 or self.temperature < 0:
            raise ValueError("retained dt must be positive and temperature nonnegative")


@dataclass(frozen=True)
class RetainedRealization:
    weight: torch.Tensor
    inverse_viscosity: torch.Tensor
    self_coupling: torch.Tensor
    bias_scale: torch.Tensor
    mass: torch.Tensor
    coupling: torch.Tensor
    initial_phase: torch.Tensor
    thermal_noise: torch.Tensor
    forces: torch.Tensor
    gauges: torch.Tensor


def _legacy_gamma(
    uniforms: np.ndarray,
    mean: float,
    variance: float,
) -> np.ndarray:
    if variance == 0:
        return np.full(uniforms.shape, mean)
    shape = mean**2 / variance
    scale = 1.0 if np.isclose(mean, 0.0) else variance / mean
    return gamma.ppf(uniforms, a=shape, scale=scale)


def generate_retained_realization(
    theta: torch.Tensor,
    config: RetainedOracleConfig,
    evaluation_seed: int,
) -> RetainedRealization:
    """Reproduce the retained single-generator NumPy/SciPy draw order."""

    if theta.shape != (2,) or theta.dtype is not torch.float64 or theta.device.type != "cpu":
        raise TensorContractError("retained oracle theta must be CPU float64 with shape [2]")
    coupling_mean, coupling_cv = (float(value) for value in theta)
    B, R, N = config.batch_size, config.repeat_size, config.node_count
    forces = np.asarray(RETAINED_M12.forces, dtype=np.float64)
    gauges = np.asarray(RETAINED_M12.gauges, dtype=np.float64)
    V = len(forces) * len(gauges)
    rng = np.random.default_rng(evaluation_seed)

    normals = rng.multivariate_normal(
        mean=np.zeros(5),
        cov=np.eye(5),
        size=B * N,
    )
    uniforms = norm.cdf(normals)
    definitions = (
        ("weight", 1.0, 0.25**2),
        ("inverse_viscosity", 1.0, 0.25**2),
        ("self_coupling", 0.0, 0.0),
        ("bias_scale", 1.0, 0.25**2),
        ("mass", 1e-3, 0.0),
    )
    fields = {
        name: _legacy_gamma(uniforms[:, index], mean, variance).reshape(B, N)
        for index, (name, mean, variance) in enumerate(definitions)
    }
    coupling_batches = []
    for _ in range(B):
        matrix = rng.normal(
            coupling_mean,
            coupling_mean * coupling_cv,
            size=(N, N),
        )
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        coupling_batches.append(matrix)
    coupling = np.stack(coupling_batches)
    initial = rng.uniform(0.0, 2.0 * np.pi, size=(V, B, R, N)).astype(np.float32)
    noise = np.stack(
        tuple(
            rng.normal(
                0.0,
                np.sqrt(2.0 * config.temperature / config.dt),
                size=(V, B, R, N),
            )
            for _ in range(config.steps)
        )
    )
    return RetainedRealization(
        weight=torch.from_numpy(fields["weight"]),
        inverse_viscosity=torch.from_numpy(fields["inverse_viscosity"]),
        self_coupling=torch.from_numpy(fields["self_coupling"]),
        bias_scale=torch.from_numpy(fields["bias_scale"]),
        mass=torch.from_numpy(fields["mass"]),
        coupling=torch.from_numpy(coupling),
        initial_phase=torch.from_numpy(initial),
        thermal_noise=torch.from_numpy(noise),
        forces=torch.from_numpy(forces),
        gauges=torch.from_numpy(gauges),
    )


def run_retained_torch(
    realization: RetainedRealization,
    *,
    dt: float,
    temperature: float,
) -> GraphRun:
    """Torch port of the retained CPU evolution using its mixed dtype contract."""

    V, B, R, N = realization.initial_phase.shape
    steps = realization.thermal_noise.shape[0]
    force = realization.forces.repeat_interleave(realization.gauges.numel())
    gauge = realization.gauges.repeat(realization.forces.numel())
    phase = realization.initial_phase.clone()
    velocity = torch.zeros_like(phase)
    previous_acceleration = torch.zeros_like(phase)
    previous_phase = phase.clone()
    z_accum = torch.zeros((V, B, R), dtype=torch.float32)
    phi_accum = torch.zeros_like(z_accum)
    weight = realization.weight[None, :, None, :]
    alpha = realization.inverse_viscosity[None, :, None, :]
    bias_scale = realization.bias_scale[None, :, None, :]
    mass = realization.mass[None, :, None, :]
    coupling = realization.coupling[None, :, None, :, :]
    force_view = force[:, None, None, None]
    gauge_view = gauge[:, None, None, None]
    cond1 = (realization.weight * realization.mass / realization.inverse_viscosity) < (
        0.1 * temperature
    )
    cond2 = (realization.mass / realization.inverse_viscosity) < (0.1 * dt)
    cond3 = (realization.weight * realization.mass) < (
        0.1 * realization.inverse_viscosity.square()
    )
    cond4 = (realization.weight * realization.mass) < (
        0.1 * realization.inverse_viscosity * temperature
    )
    cond5 = (temperature * realization.mass) < (0.1 * realization.inverse_viscosity)
    mask_compact = (cond1 | cond2 | cond3 | cond4 | cond5)[None, :, None, :]
    mask = mask_compact.expand_as(phase)
    first_step: dict[str, torch.Tensor] = {}
    start = perf_counter()
    for step in range(steps):
        restoring = -weight * torch.sin(phase - bias_scale * gauge_view)
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        pair = (-(coupling * torch.sin(difference)).sum(dim=-1) / N).to(torch.float32)
        bias = bias_scale * force_view
        noise_term = realization.thermal_noise[step] * torch.sqrt(alpha)
        alpha_force = alpha * (restoring + pair + bias) + noise_term
        acceleration_candidate = (alpha_force - velocity) / mass
        acceleration = torch.where(mask, 0.0, acceleration_candidate)
        returned_velocity = torch.where(mask, alpha_force, velocity)
        new_phase = phase.clone()
        new_velocity = velocity.clone()
        new_previous_acceleration = previous_acceleration.clone()
        new_phase[mask] += (returned_velocity[mask] * dt).to(torch.float32)
        new_velocity[mask] = returned_velocity[mask].to(torch.float32)
        inverse_mask = ~mask
        new_phase[inverse_mask] += (
            returned_velocity[inverse_mask] * dt
            + 0.5 * previous_acceleration[inverse_mask] * dt**2
        ).to(torch.float32)
        new_velocity[inverse_mask] += (
            0.5
            * (
                previous_acceleration[inverse_mask]
                + acceleration[inverse_mask]
            )
            * dt
        ).to(torch.float32)
        new_previous_acceleration[inverse_mask] = acceleration[inverse_mask].to(torch.float32)
        phase, velocity, previous_acceleration = (
            new_phase,
            new_velocity,
            new_previous_acceleration,
        )
        z_accum += torch.abs(torch.mean(torch.exp(1j * phase), dim=-1))
        phi_accum += torch.mean(phase - previous_phase, dim=-1)
        previous_phase = phase.clone()
        if step == 0:
            first_step = {
                "restoring": restoring.clone(),
                "coupling": pair.clone(),
                "bias": bias.clone(),
                "noise_term": noise_term.clone(),
                "alpha_force": alpha_force.clone(),
                "underdamped_acceleration": acceleration_candidate.clone(),
                "returned_velocity": returned_velocity.clone(),
                "returned_acceleration": acceleration.clone(),
                "phase": phase.clone(),
                "velocity": velocity.clone(),
                "Z_accumulator": z_accum.clone(),
                "phase_increment_accumulator": phi_accum.clone(),
                "overdamped_mask": mask_compact.clone(),
            }
    raw_omega = phi_accum / (steps * dt)
    raw_z = z_accum / steps
    response = torch.cat(
        (
            raw_omega.mean(dim=(-1, -2)),
            raw_z.mean(dim=(-1, -2)),
        )
    ).to(torch.float64).contiguous()
    return GraphRun(
        response=response,
        raw_omega=raw_omega,
        raw_synchronization=raw_z,
        first_step=first_step,
        final_phase=phase,
        sampled_steps=steps,
        timings={"evolution_seconds": perf_counter() - start},
    )


def run_retained_numpy_reference(
    realization: RetainedRealization,
    *,
    dt: float,
    temperature: float,
) -> GraphRun:
    """Transparent NumPy translation of retained mixed-precision evolution."""

    weight = realization.weight.numpy()[None, :, None, :]
    alpha = realization.inverse_viscosity.numpy()[None, :, None, :]
    bias_scale = realization.bias_scale.numpy()[None, :, None, :]
    mass = realization.mass.numpy()[None, :, None, :]
    coupling = realization.coupling.numpy()[None, :, None, :, :]
    phase = realization.initial_phase.numpy().copy()
    thermal = realization.thermal_noise.numpy()
    forces = realization.forces.numpy()
    gauges = realization.gauges.numpy()
    force = np.repeat(forces, len(gauges))[:, None, None, None]
    gauge = np.tile(gauges, len(forces))[:, None, None, None]
    velocity = np.zeros_like(phase)
    previous_acceleration = np.zeros_like(phase)
    previous_phase = phase.copy()
    z_accum = np.zeros(phase.shape[:-1], dtype=np.float32)
    phi_accum = np.zeros_like(z_accum)
    base_weight = realization.weight.numpy()
    base_alpha = realization.inverse_viscosity.numpy()
    base_mass = realization.mass.numpy()
    mask_base = (
        (base_weight * base_mass / base_alpha < 0.1 * temperature)
        | (base_mass / base_alpha < 0.1 * dt)
        | (base_weight * base_mass < 0.1 * base_alpha**2)
        | (base_weight * base_mass < 0.1 * base_alpha * temperature)
        | (temperature * base_mass < 0.1 * base_alpha)
    )
    mask_compact = mask_base[None, :, None, :]
    mask = np.broadcast_to(mask_compact, phase.shape)
    first_step: dict[str, torch.Tensor] = {}
    start = perf_counter()
    for step in range(thermal.shape[0]):
        restoring = -weight * np.sin(phase - bias_scale * gauge)
        difference = phase[..., :, None] - phase[..., None, :]
        pair = (
            -(coupling * np.sin(difference)).sum(axis=-1) / phase.shape[-1]
        ).astype(np.float32)
        bias = bias_scale * force
        noise_term = thermal[step] * np.sqrt(alpha)
        alpha_force = alpha * (restoring + pair + bias) + noise_term
        acceleration_candidate = (alpha_force - velocity) / mass
        acceleration = np.where(mask, 0.0, acceleration_candidate)
        returned_velocity = np.where(mask, alpha_force, velocity)
        phase[mask] += (returned_velocity[mask] * dt).astype(np.float32)
        velocity[mask] = returned_velocity[mask].astype(np.float32)
        inverse_mask = ~mask
        phase[inverse_mask] += (
            returned_velocity[inverse_mask] * dt
            + 0.5 * previous_acceleration[inverse_mask] * dt**2
        ).astype(np.float32)
        velocity[inverse_mask] += (
            0.5
            * (previous_acceleration[inverse_mask] + acceleration[inverse_mask])
            * dt
        ).astype(np.float32)
        previous_acceleration[inverse_mask] = acceleration[inverse_mask].astype(np.float32)
        z_accum += np.abs(np.mean(np.exp(1j * phase), axis=-1))
        phi_accum += np.mean(phase - previous_phase, axis=-1)
        previous_phase = phase.copy()
        if step == 0:
            first_step = {
                key: torch.from_numpy(np.ascontiguousarray(value.copy()))
                for key, value in {
                    "restoring": restoring,
                    "coupling": pair,
                    "bias": bias,
                    "noise_term": noise_term,
                    "alpha_force": alpha_force,
                    "underdamped_acceleration": acceleration_candidate,
                    "returned_velocity": returned_velocity,
                    "returned_acceleration": acceleration,
                    "phase": phase,
                    "velocity": velocity,
                    "Z_accumulator": z_accum,
                    "phase_increment_accumulator": phi_accum,
                    "overdamped_mask": mask_compact,
                }.items()
            }
    raw_omega = phi_accum / (thermal.shape[0] * dt)
    raw_z = z_accum / thermal.shape[0]
    response = np.concatenate(
        (
            raw_omega.mean(axis=(-1, -2)),
            raw_z.mean(axis=(-1, -2)),
        )
    ).astype(np.float64)
    return GraphRun(
        response=torch.from_numpy(np.ascontiguousarray(response)),
        raw_omega=torch.from_numpy(np.ascontiguousarray(raw_omega)),
        raw_synchronization=torch.from_numpy(np.ascontiguousarray(raw_z)),
        first_step=first_step,
        final_phase=torch.from_numpy(np.ascontiguousarray(phase)),
        sampled_steps=thermal.shape[0],
        timings={"evolution_seconds": perf_counter() - start},
    )


def retained_realization_from_fixture(fixture: RetainedGraphFixture) -> RetainedRealization:
    configuration = fixture.configuration
    return RetainedRealization(
        weight=fixture.tensor("realized_weight")[0],
        inverse_viscosity=fixture.tensor("realized_inverse_viscosity")[0],
        self_coupling=fixture.tensor("realized_self_coupling")[0],
        bias_scale=fixture.tensor("realized_bias_scale")[0],
        mass=fixture.tensor("realized_mass")[0],
        coupling=fixture.tensor("realized_total_coupling")[0],
        initial_phase=fixture.tensor("initial_phase")[0],
        thermal_noise=fixture.tensor("thermal_noise")[:, 0],
        forces=torch.tensor(configuration["forces"], dtype=torch.float64),
        gauges=torch.tensor(configuration["gauges"], dtype=torch.float64),
    )


@dataclass(frozen=True)
class RetainedOracleSimulator:
    config: RetainedOracleConfig = RetainedOracleConfig()

    @property
    def capabilities(self) -> SimulatorCapabilities:
        profile = parameter_profile("p2-reduced-coupling")
        return SimulatorCapabilities(
            parameter_spec=profile.parameter_spec,
            response_spec=RETAINED_M12.response_spec,
            simulator_identity=f"{RETAINED_ORACLE_IDENTITY}.p2.m12.mixed",
            deterministic=False,
            supports_batch=False,
            supports_context=True,
            definition_metadata={
                "model_identity": RETAINED_ORACLE_IDENTITY,
                "dtype_policy": "historical-mixed",
                "rng_policy": "retained-single-numpy-generator-draw-order",
                "node_count": self.config.node_count,
                "batch_size": self.config.batch_size,
                "repeat_size": self.config.repeat_size,
                "dt": self.config.dt,
                "steps": self.config.steps,
                "burn_in_steps": 0,
                "sample_stride": 1,
            },
        )

    def evaluate(
        self,
        theta: torch.Tensor,
        context: SimulationContext | None = None,
    ) -> SimulationResult:
        if context is None:
            raise ValueError("retained oracle evaluation requires SimulationContext")
        start = perf_counter()
        realization = generate_retained_realization(theta, self.config, context.evaluation_seed)
        run = run_retained_torch(
            realization,
            dt=self.config.dt,
            temperature=self.config.temperature,
        )
        return SimulationResult(
            run.response,
            {
                "model_identity": RETAINED_ORACLE_IDENTITY,
                "simulator_identity": self.capabilities.simulator_identity,
                "evaluation_seed": context.evaluation_seed,
                "rng_policy": "retained-single-numpy-generator-draw-order",
                "native_dtype_policy": "historical-mixed",
                "boundary_dtype": "float64",
                "device": "cpu",
                "elapsed_seconds": perf_counter() - start,
                "evolution_seconds": run.timings["evolution_seconds"],
                "transport_call_count": 1,
            },
        )
