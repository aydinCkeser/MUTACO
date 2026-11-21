"""Approved overdamped graph-oscillator model and independent NumPy reference."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import torch

from mutaco.identity import ScientificIdentity
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.simulator import (
    SimulationContext,
    SimulationResult,
    SimulatorCapabilities,
    TensorContractError,
)

from .config import GraphSimulationConfig
from .profiles import (
    SCIENTIFIC_IDENTITY,
    SCIENTIFIC_M20,
    GraphParameterProfile,
    parameter_profile,
)
from .runtime import GraphDeviceResolution, resolve_graph_device


NODE_PARAMETER_IDS = (
    "weight",
    "inverse_viscosity",
    "bias_scale",
)


@dataclass(frozen=True)
class ScientificConditionGrid:
    """Explicit force/gauge conditions with isolated supplemental RNG streams."""

    force_ticks: tuple[int, ...]
    force_step: float
    gauges: tuple[float, ...]
    identifier: str = "canonical_m20_v1"

    def __post_init__(self) -> None:
        if not self.force_ticks or len(set(self.force_ticks)) != len(self.force_ticks):
            raise ValueError("condition force ticks must be nonempty and unique")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.force_ticks
        ):
            raise ValueError("condition force ticks must be nonnegative integers")
        if not np.isfinite(self.force_step) or self.force_step <= 0.0:
            raise ValueError("condition force_step must be finite and positive")
        if not self.gauges or any(not np.isfinite(value) for value in self.gauges):
            raise ValueError("condition gauges must be nonempty and finite")
        if len(set(self.gauges)) != len(self.gauges):
            raise ValueError("condition gauges must be unique")
        if not self.identifier:
            raise ValueError("condition grid identifier must be nonempty")

    @property
    def forces(self) -> tuple[float, ...]:
        return tuple(float(tick) * self.force_step for tick in self.force_ticks)

    @property
    def condition_count(self) -> int:
        return len(self.force_ticks) * len(self.gauges)

    def condition_pairs(self) -> tuple[tuple[int, int, int, float, float], ...]:
        return tuple(
            (force_index, gauge_index, tick, force, gauge)
            for force_index, (tick, force) in enumerate(
                zip(self.force_ticks, self.forces, strict=True)
            )
            for gauge_index, gauge in enumerate(self.gauges)
        )

    def canonical_condition_index(self, tick: int, gauge_index: int) -> int | None:
        canonical_ticks = (0, 25, 50, 75, 100)
        if self.force_step != 0.05 or self.gauges != (0.0, float(torch.pi)):
            return None
        try:
            force_index = canonical_ticks.index(tick)
        except ValueError:
            return None
        return force_index * 2 + gauge_index

    def document(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "force_ticks": list(self.force_ticks),
            "force_step": self.force_step,
            "forces": list(self.forces),
            "gauges": list(self.gauges),
            "condition_count": self.condition_count,
        }


CANONICAL_CONDITION_GRID = ScientificConditionGrid(
    force_ticks=(0, 25, 50, 75, 100),
    force_step=0.05,
    gauges=(0.0, float(torch.pi)),
)


def dense_force_condition_grid() -> ScientificConditionGrid:
    return ScientificConditionGrid(
        force_ticks=tuple(range(101)),
        force_step=0.05,
        gauges=(0.0, float(torch.pi)),
        identifier="dense_force_0p05_v1",
    )


@dataclass(frozen=True)
class ScientificRealization:
    """Fully realized inputs, all on one device in approved float32."""

    weight: torch.Tensor  # [B,N]
    inverse_viscosity: torch.Tensor  # [B,N]
    bias_scale: torch.Tensor  # [B,N]
    coupling: torch.Tensor  # [B,N,N]
    initial_phase: torch.Tensor  # [B,R,N]
    thermal_noise: torch.Tensor  # [T,V,B,R,N], standard normal
    seed_manifest: Mapping[str, int]
    conditions: ScientificConditionGrid = CANONICAL_CONDITION_GRID

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.weight,
            self.inverse_viscosity,
            self.bias_scale,
            self.coupling,
            self.initial_phase,
            self.thermal_noise,
        )

    def validate(self, config: GraphSimulationConfig, device: torch.device) -> None:
        B, R, N = config.batch_size, config.repeat_size, config.node_count
        V, T = self.conditions.condition_count, config.total_steps
        expected = ((B, N), (B, N), (B, N), (B, N, N), (B, R, N), (T, V, B, R, N))
        for tensor, shape in zip(self.tensors(), expected):
            if tensor.shape != shape or tensor.device != device or tensor.dtype is not torch.float32:
                raise TensorContractError(
                    f"scientific realization tensor contract mismatch: "
                    f"shape={tuple(tensor.shape)}, device={tensor.device}, dtype={tensor.dtype}"
                )
            if not tensor.is_contiguous() or not bool(torch.isfinite(tensor).all()):
                raise TensorContractError("scientific realization must be finite and contiguous")
        if not torch.equal(self.coupling, self.coupling.transpose(-1, -2)):
            raise TensorContractError("scientific coupling matrix must be exactly symmetric")
        if bool(torch.diagonal(self.coupling, dim1=-2, dim2=-1).ne(0).any()):
            raise TensorContractError("scientific coupling diagonal must be exactly zero")


@dataclass(frozen=True)
class GraphRun:
    response: torch.Tensor
    raw_omega: torch.Tensor
    raw_synchronization: torch.Tensor
    first_step: Mapping[str, torch.Tensor]
    final_phase: torch.Tensor
    sampled_steps: int
    timings: Mapping[str, float]


def _gamma_field(
    mean: torch.Tensor,
    cv: torch.Tensor,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    cv_squared = cv.square()
    concentration_scalar = torch.where(
        cv_squared == 0,
        torch.ones_like(cv_squared),
        cv_squared.reciprocal(),
    )
    concentration = concentration_scalar.expand(shape).contiguous()
    sampled = torch._standard_gamma(concentration, generator=generator)
    varying = sampled * (mean * cv_squared)
    return torch.where(cv_squared == 0, mean.expand(shape), varying).contiguous()


def _scientific_seed_deriver(
    config: GraphSimulationConfig,
    evaluation_seed: int,
) -> SeedDeriver:
    identity = ScientificIdentity.create(config.identity_document())
    return SeedDeriver(identity, evaluation_seed)


def generate_scientific_realization(
    full_parameters: Mapping[str, torch.Tensor],
    config: GraphSimulationConfig,
    context: SimulationContext,
    *,
    device: torch.device,
    conditions: ScientificConditionGrid = CANONICAL_CONDITION_GRID,
) -> ScientificRealization:
    """Draw fixed disorder, initial phases and thermal increments for one evaluation.

    B disorder realizations have node fields [B,N] and symmetric zero-diagonal
    couplings [B,N,N]; R initial repeats have phases [B,R,N]. Thermal standard
    normals [T,V,B,R,N] are independent across addressed condition/repeat streams.
    The same disorder and initial phases are reused across V force/gauge conditions.

    Node fields are Gamma with the declared means/CVs; couplings are Gaussian
    with std=mean*CV and can be negative. A zero node CV gives the constant mean.
    All arrays are device-native float32. Canonical thermal addresses are retained
    when extra force conditions are added, and the seed manifest records the draws.
    """

    deriver = _scientific_seed_deriver(config, context.evaluation_seed)
    B, R, N = config.batch_size, config.repeat_size, config.node_count
    V, T = conditions.condition_count, config.total_steps
    fields: dict[str, list[torch.Tensor]] = {name: [] for name in NODE_PARAMETER_IDS}
    seed_manifest: dict[str, int] = {}
    field_parameters = {
        "weight": ("weight_mean", "weight_CV"),
        "inverse_viscosity": (None, "inverse_viscosity_CV"),
        "bias_scale": ("bias_scale_mean", "bias_scale_CV"),
    }
    for b in range(B):
        for field_name, (mean_name, cv_name) in field_parameters.items():
            coordinate = SeedCoordinate(
                "quenched_disorder",
                ("node_parameter", field_name, b),
            )
            seed = deriver.seed_for(coordinate)
            seed_manifest[f"node_parameter/{field_name}/B={b}"] = seed
            mean = (
                full_parameters[mean_name]
                if mean_name is not None
                else torch.tensor(1.0, dtype=torch.float32, device=device)
            )
            fields[field_name].append(
                _gamma_field(
                    mean,
                    full_parameters[cv_name],
                    (N,),
                    generator=deriver.generator(coordinate, device=device),
                )
            )
    node_fields = {
        name: torch.stack(values, dim=0).contiguous()
        for name, values in fields.items()
    }

    edge_rows, edge_cols = torch.triu_indices(N, N, offset=1, device=device)
    coupling_batches = []
    coupling_mean = full_parameters["pairwise_coupling_mean"]
    coupling_std = coupling_mean * full_parameters["pairwise_coupling_CV"]
    for b in range(B):
        coordinate = SeedCoordinate(
            "quenched_disorder",
            ("pairwise_coupling", b),
        )
        seed_manifest[f"pairwise_coupling/B={b}"] = deriver.seed_for(coordinate)
        values = torch.randn(
            (edge_rows.numel(),),
            dtype=torch.float32,
            device=device,
            generator=deriver.generator(coordinate, device=device),
        )
        values = values * coupling_std + coupling_mean
        matrix = torch.zeros((N, N), dtype=torch.float32, device=device)
        matrix[edge_rows, edge_cols] = values
        matrix[edge_cols, edge_rows] = values
        coupling_batches.append(matrix)
    coupling = torch.stack(coupling_batches).contiguous()

    initial = torch.empty((B, R, N), dtype=torch.float32, device=device)
    for b in range(B):
        for r in range(R):
            coordinate = SeedCoordinate("initial_state", (b, r))
            seed_manifest[f"initial_state/B={b}/R={r}"] = deriver.seed_for(coordinate)
            initial[b, r] = torch.rand(
                (N,),
                dtype=torch.float32,
                device=device,
                generator=deriver.generator(coordinate, device=device),
            ) * (2.0 * torch.pi)

    thermal = torch.empty((T, V, B, R, N), dtype=torch.float32, device=device)
    for b in range(B):
        for r in range(R):
            for v, (_, gauge_index, tick, _, _) in enumerate(
                conditions.condition_pairs()
            ):
                canonical_index = conditions.canonical_condition_index(
                    tick, gauge_index
                )
                # Preserve historical streams when extra force conditions are added.
                if canonical_index is not None:
                    coordinate = SeedCoordinate(
                        "thermal_noise", (b, r, canonical_index)
                    )
                    manifest_key = (
                        f"thermal_noise/B={b}/R={r}/V={canonical_index}"
                    )
                else:
                    coordinate = SeedCoordinate(
                        "thermal_noise_dense_force_v1",
                        (b, r, tick, gauge_index),
                    )
                    manifest_key = (
                        f"thermal_noise_dense_force_v1/B={b}/R={r}/"
                        f"force_tick={tick}/gauge={gauge_index}"
                    )
                seed_manifest[manifest_key] = deriver.seed_for(coordinate)
                thermal[:, v, b, r, :] = torch.randn(
                    (T, N),
                    dtype=torch.float32,
                    device=device,
                    generator=deriver.generator(coordinate, device=device),
                )
    realization = ScientificRealization(
        weight=node_fields["weight"],
        inverse_viscosity=node_fields["inverse_viscosity"],
        bias_scale=node_fields["bias_scale"],
        coupling=coupling,
        initial_phase=initial,
        thermal_noise=thermal,
        seed_manifest=seed_manifest,
        conditions=conditions,
    )
    realization.validate(config, device)
    return realization


def run_scientific_torch(
    realization: ScientificRealization,
    config: GraphSimulationConfig,
) -> GraphRun:
    """Evolve captured overdamped graph phases by float32 Euler-Maruyama steps.

    Arrays use condition/disorder/repeat/node axes [V,B,R,N]. Drift combines the
    sinusoidal local restoring term, symmetric pair interactions divided by N and
    the applied bias, all multiplied by inverse viscosity. Thermal increments have
    scale sqrt(2*temperature*dt*inverse_viscosity). Phases remain unwrapped so their
    increments measure mean angular velocity rather than wrapped angle changes.

    After burn-in, sample mean node velocity and |mean_nodes exp(i*phase)| at the
    configured stride. Average in time, then over B/R. response [2*V] contains all
    velocities followed by all synchronizations, each force-major/gauge-minor.
    GraphRun also retains [V,B,R] time averages and final phases. No random draws
    occur here: comparisons with NumPy must reuse the captured realization.
    """

    device = realization.weight.device
    realization.validate(config, device)
    forces = torch.tensor(
        realization.conditions.forces, dtype=torch.float32, device=device
    )
    gauges = torch.tensor(
        realization.conditions.gauges, dtype=torch.float32, device=device
    )
    force_grid = forces.repeat_interleave(len(gauges))
    gauge_grid = gauges.repeat(len(forces))
    V, B, R, N = force_grid.numel(), config.batch_size, config.repeat_size, config.node_count
    phase = realization.initial_phase.unsqueeze(0).expand(V, B, R, N).clone()
    omega_accum = torch.zeros((V, B, R), dtype=torch.float32, device=device)
    z_accum = torch.zeros_like(omega_accum)
    alpha = realization.inverse_viscosity.unsqueeze(0).unsqueeze(2)
    weight = realization.weight.unsqueeze(0).unsqueeze(2)
    bias_scale = realization.bias_scale.unsqueeze(0).unsqueeze(2)
    coupling = realization.coupling.unsqueeze(0).unsqueeze(2)
    force = force_grid[:, None, None, None]
    gauge = gauge_grid[:, None, None, None]
    dt = torch.tensor(config.dt, dtype=torch.float32, device=device)
    noise_scale = torch.sqrt(
        torch.tensor(2.0 * config.temperature * config.dt, dtype=torch.float32, device=device)
        * alpha
    )
    first_step: dict[str, torch.Tensor] = {}
    kernel_start = perf_counter()
    sampled = 0
    for step in range(config.total_steps):
        previous = phase
        restoring = -weight * torch.sin(phase - bias_scale * gauge)
        phase_difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        pair_coupling = -(
            coupling * torch.sin(phase_difference)
        ).sum(dim=-1) / float(N)
        bias = bias_scale * force
        drift = alpha * (restoring + pair_coupling + bias)
        thermal_increment = noise_scale * realization.thermal_noise[step]
        phase = (phase + drift * dt + thermal_increment).contiguous()
        if step == 0:
            first_step = {
                "restoring": restoring.clone(),
                "coupling": pair_coupling.clone(),
                "bias": bias.expand_as(phase).clone(),
                "drift": drift.clone(),
                "thermal_increment": thermal_increment.clone(),
                "phase": phase.clone(),
            }
        if step >= config.burn_in_steps and (
            (step - config.burn_in_steps) % config.sample_stride == 0
        ):
            omega_accum += (phase - previous).mean(dim=-1) / dt
            z_accum += torch.abs(torch.mean(torch.exp(1j * phase), dim=-1))
            sampled += 1
    kernel_seconds = perf_counter() - kernel_start
    if sampled != config.sampled_step_count:
        raise RuntimeError("scientific sampled-step accounting drifted")
    raw_omega = omega_accum / float(sampled)
    raw_z = z_accum / float(sampled)
    response = torch.cat(
        (
            raw_omega.mean(dim=(-1, -2)),
            raw_z.mean(dim=(-1, -2)),
        )
    ).contiguous()
    if response.shape != (2 * V,) or not bool(torch.isfinite(response).all()):
        raise TensorContractError("scientific response is invalid")
    return GraphRun(
        response=response,
        raw_omega=raw_omega,
        raw_synchronization=raw_z,
        first_step=first_step,
        final_phase=phase,
        sampled_steps=sampled,
        timings={"evolution_seconds": kernel_seconds},
    )


def run_scientific_numpy_reference(
    realization: ScientificRealization,
    config: GraphSimulationConfig,
) -> GraphRun:
    """Transparent independent NumPy implementation using captured realized inputs."""

    arrays = [tensor.detach().cpu().numpy().copy() for tensor in realization.tensors()]
    weight, alpha_value, bias_scale_value, coupling_value, initial, thermal = arrays
    forces = np.asarray(realization.conditions.forces, dtype=np.float32)
    gauges = np.asarray(realization.conditions.gauges, dtype=np.float32)
    force = np.repeat(forces, len(gauges))[:, None, None, None]
    gauge = np.tile(gauges, len(forces))[:, None, None, None]
    V, B, R, N = force.shape[0], config.batch_size, config.repeat_size, config.node_count
    phase = np.broadcast_to(initial[None], (V, B, R, N)).copy()
    omega_accum = np.zeros((V, B, R), dtype=np.float32)
    z_accum = np.zeros_like(omega_accum)
    alpha = alpha_value[None, :, None, :]
    weight = weight[None, :, None, :]
    bias_scale = bias_scale_value[None, :, None, :]
    coupling = coupling_value[None, :, None, :, :]
    dt = np.float32(config.dt)
    noise_scale = np.sqrt(np.float32(2.0 * config.temperature * config.dt) * alpha)
    sampled = 0
    first_step: dict[str, torch.Tensor] = {}
    start = perf_counter()
    for step in range(config.total_steps):
        previous = phase
        restoring = -weight * np.sin(phase - bias_scale * gauge)
        difference = phase[..., :, None] - phase[..., None, :]
        pair = -(coupling * np.sin(difference)).sum(axis=-1) / np.float32(N)
        bias = bias_scale * force
        drift = alpha * (restoring + pair + bias)
        thermal_increment = noise_scale * thermal[step]
        phase = np.ascontiguousarray(phase + drift * dt + thermal_increment)
        if step == 0:
            first_step = {
                key: torch.from_numpy(np.ascontiguousarray(value.copy()))
                for key, value in {
                    "restoring": restoring,
                    "coupling": pair,
                    "bias": np.broadcast_to(bias, phase.shape),
                    "drift": drift,
                    "thermal_increment": thermal_increment,
                    "phase": phase,
                }.items()
            }
        if step >= config.burn_in_steps and (
            (step - config.burn_in_steps) % config.sample_stride == 0
        ):
            omega_accum += (phase - previous).mean(axis=-1) / dt
            z_accum += np.abs(np.mean(np.exp(1j * phase), axis=-1))
            sampled += 1
    raw_omega = omega_accum / np.float32(sampled)
    raw_z = z_accum / np.float32(sampled)
    response = np.ascontiguousarray(
        np.concatenate((raw_omega.mean(axis=(-1, -2)), raw_z.mean(axis=(-1, -2))))
    )
    return GraphRun(
        response=torch.from_numpy(response),
        raw_omega=torch.from_numpy(raw_omega),
        raw_synchronization=torch.from_numpy(raw_z),
        first_step=first_step,
        final_phase=torch.from_numpy(phase),
        sampled_steps=sampled,
        timings={"evolution_seconds": perf_counter() - start},
    )


@dataclass(frozen=True)
class OverdampedScientificSimulator:
    """Context-seeded physical graph model with canonical 20-channel responses.

    A selected profile maps theta [P] to named means/CVs and fixes remaining
    parameters. evaluate() generates disorder and thermal paths, evolves them and
    returns float32 [20]: ten angular velocities then ten synchronization values.
    Inputs must match the resolved device; deterministic CUDA execution is required
    when CUDA is used. Identical context alone is not an all-platform parity claim.
    """
    config: GraphSimulationConfig = GraphSimulationConfig()

    def __post_init__(self) -> None:
        if self.config.model_identity != SCIENTIFIC_IDENTITY:
            raise ValueError("scientific simulator requires the approved scientific identity")

    @property
    def profile(self) -> GraphParameterProfile:
        return parameter_profile(self.config.parameter_profile_id)

    @property
    def device_resolution(self) -> GraphDeviceResolution:
        return resolve_graph_device(self.config.device)

    @property
    def capabilities(self) -> SimulatorCapabilities:
        return SimulatorCapabilities(
            parameter_spec=self.profile.parameter_spec,
            response_spec=SCIENTIFIC_M20.response_spec,
            simulator_identity=(
                f"{SCIENTIFIC_IDENTITY}.{self.profile.identifier}."
                f"{SCIENTIFIC_M20.identifier}.float32"
            ),
            deterministic=False,
            supports_batch=True,
            supports_context=True,
            definition_metadata=self.config.identity_document(),
        )

    def evaluate(
        self,
        theta: torch.Tensor,
        context: SimulationContext | None = None,
    ) -> SimulationResult:
        if context is None:
            raise ValueError("scientific graph evaluation requires SimulationContext")
        device = self.device_resolution.device
        if theta.device != device or theta.dtype is not torch.float32:
            raise TensorContractError("scientific graph theta must match resolved device and float32")
        if device.type == "cuda" and not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "exact scientific CUDA replay requires torch deterministic algorithms"
            )
        start = perf_counter()
        full = self.profile.expand_theta(theta)
        generation_start = perf_counter()
        realization = generate_scientific_realization(full, self.config, context, device=device)
        generation_seconds = perf_counter() - generation_start
        run = run_scientific_torch(realization, self.config)
        return SimulationResult(
            run.response,
            {
                "model_identity": SCIENTIFIC_IDENTITY,
                "simulator_identity": self.capabilities.simulator_identity,
                "parameter_profile": self.profile.identifier,
                "response_profile": SCIENTIFIC_M20.identifier,
                "evaluation_seed": context.evaluation_seed,
                "seed_policy": "mutaco-coordinate-sha256-v1",
                "seed_manifest": dict(realization.seed_manifest),
                "device_requested": self.config.device,
                "device_effective": str(device),
                "device_fallback": self.device_resolution.used_fallback,
                "dtype": "float32",
                "dt": self.config.dt,
                "burn_in_steps": self.config.burn_in_steps,
                "sample_steps": self.config.sample_steps,
                "sample_stride": self.config.sample_stride,
                "total_steps": self.config.total_steps,
                "sampled_steps": run.sampled_steps,
                "node_count": self.config.node_count,
                "batch_size": self.config.batch_size,
                "repeat_size": self.config.repeat_size,
                "generation_seconds": generation_seconds,
                "evolution_seconds": run.timings["evolution_seconds"],
                "elapsed_seconds": perf_counter() - start,
                "transport_call_count": 1,
            },
        )

    def evaluate_condition_grid(
        self,
        theta: torch.Tensor,
        context: SimulationContext,
        conditions: ScientificConditionGrid,
        *,
        max_conditions_per_chunk: int = 10,
    ) -> tuple[GraphRun, dict[str, object]]:
        """Evaluate a declared grid without changing protocol capabilities."""

        device = self.device_resolution.device
        if theta.device != device or theta.dtype is not torch.float32:
            raise TensorContractError(
                "scientific graph theta must match resolved device and float32"
            )
        if device.type == "cuda" and not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "exact scientific CUDA replay requires torch deterministic algorithms"
            )
        if (
            isinstance(max_conditions_per_chunk, bool)
            or not isinstance(max_conditions_per_chunk, int)
            or max_conditions_per_chunk < len(conditions.gauges)
        ):
            raise ValueError("condition chunk limit must fit at least one force")
        forces_per_chunk = max_conditions_per_chunk // len(conditions.gauges)
        started = perf_counter()
        full = self.profile.expand_theta(theta)
        generation_seconds = 0.0
        chunks: list[GraphRun] = []
        seed_manifest: dict[str, int] = {}
        for start in range(0, len(conditions.force_ticks), forces_per_chunk):
            chunk_conditions = ScientificConditionGrid(
                force_ticks=conditions.force_ticks[start : start + forces_per_chunk],
                force_step=conditions.force_step,
                gauges=conditions.gauges,
                identifier=f"{conditions.identifier}.chunk_v1",
            )
            generation_started = perf_counter()
            realization = generate_scientific_realization(
                full,
                self.config,
                context,
                device=device,
                conditions=chunk_conditions,
            )
            generation_seconds += perf_counter() - generation_started
            seed_manifest.update(realization.seed_manifest)
            chunks.append(run_scientific_torch(realization, self.config))
        condition_sizes = [chunk.raw_omega.shape[0] for chunk in chunks]
        omega_parts = [
            chunk.response[:size] for chunk, size in zip(chunks, condition_sizes, strict=True)
        ]
        synchronization_parts = [
            chunk.response[size:] for chunk, size in zip(chunks, condition_sizes, strict=True)
        ]
        run = GraphRun(
            response=torch.cat((*omega_parts, *synchronization_parts)).contiguous(),
            raw_omega=torch.cat([chunk.raw_omega for chunk in chunks], dim=0).contiguous(),
            raw_synchronization=torch.cat(
                [chunk.raw_synchronization for chunk in chunks], dim=0
            ).contiguous(),
            first_step={
                key: torch.cat([chunk.first_step[key] for chunk in chunks], dim=0).contiguous()
                for key in chunks[0].first_step
            },
            final_phase=torch.cat([chunk.final_phase for chunk in chunks], dim=0).contiguous(),
            sampled_steps=chunks[0].sampled_steps,
            timings={
                "evolution_seconds": sum(
                    chunk.timings["evolution_seconds"] for chunk in chunks
                )
            },
        )
        return run, {
            "schema_version": "mutaco-graph-condition-grid-evaluation-v1",
            "simulator_identity": self.capabilities.simulator_identity,
            "condition_grid": conditions.document(),
            "evaluation_seed": context.evaluation_seed,
            "seed_policy": "mutaco-coordinate-sha256-v1",
            "supplemental_thermal_namespace": "thermal_noise_dense_force_v1",
            "seed_manifest": seed_manifest,
            "chunking": {
                "policy": "force_major_fixed_condition_cap_v1",
                "max_conditions_per_chunk": max_conditions_per_chunk,
                "force_chunk_count": len(chunks),
                "condition_counts": condition_sizes,
            },
            "device_effective": str(device),
            "dtype": "float32",
            "generation_seconds": generation_seconds,
            "evolution_seconds": run.timings["evolution_seconds"],
            "elapsed_seconds": perf_counter() - started,
            "transport_call_count": 1,
        }

    def evaluate_batch(
        self,
        theta: torch.Tensor,
        contexts: Sequence[SimulationContext],
    ) -> SimulationResult:
        if theta.ndim != 2 or theta.shape[0] != len(contexts):
            raise TensorContractError("batched graph evaluation requires one context per H row")
        values = torch.stack(
            tuple(self.evaluate(row, context).values for row, context in zip(theta, contexts))
        )
        return SimulationResult(
            values.contiguous(),
            {
                "model_identity": SCIENTIFIC_IDENTITY,
                "batch_size_H": theta.shape[0],
                "evaluation_seeds": tuple(context.evaluation_seed for context in contexts),
            },
        )
