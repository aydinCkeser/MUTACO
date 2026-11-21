"""Composite Monte Carlo NEI on joint latent multi-output GP responses."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch

from mutaco.identity import stable_hash
from mutaco.posterior import JointLatentPosteriorSummary
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.response import FrozenResponseTransform
from mutaco.types import ParameterSpec


COMPOSITE_MC_NEI_PHYSICAL_MAE = "composite_mc_nei_physical_mae_v1"
NEI_RAW_RECOVERY = "best_ranked_finite_unevaluated_raw_v1"
NEI_BASELINE_FACTORIZATION = "promoted_float64_cholesky_zero_jitter_v1"


class JointLatentPosteriorProvider(Protocol):
    model_id: str
    transform_id: str

    def joint_posterior(self, theta: torch.Tensor) -> JointLatentPosteriorSummary: ...


class NEIAcquisitionError(RuntimeError):
    """NEI cannot produce a valid finite unevaluated candidate."""


@dataclass(frozen=True)
class NEIBaselineFactorizationDiagnostic:
    policy_id: str
    input_dtype: str
    factor_dtype: str
    dimension: int
    promotion_occurred: bool
    jitter_added: float
    relative_jitter: float
    cholesky_attempts: int

    def document(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "input_dtype": self.input_dtype,
            "factor_dtype": self.factor_dtype,
            "dimension": self.dimension,
            "promotion_occurred": self.promotion_occurred,
            "jitter_added": self.jitter_added,
            "relative_jitter": self.relative_jitter,
            "cholesky_attempts": self.cholesky_attempts,
        }


@dataclass(frozen=True)
class NEIConfig:
    n_mc: int
    raw_samples: int
    restarts: int
    max_iterations: int
    learning_rate: float
    seed_deriver: SeedDeriver
    raw_coordinate: SeedCoordinate
    mc_coordinate: SeedCoordinate
    optimizer_coordinate: SeedCoordinate

    def __post_init__(self) -> None:
        for name in ("n_mc", "raw_samples", "restarts", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.restarts > self.raw_samples:
            raise ValueError("NEI restarts cannot exceed raw samples")
        if not isinstance(self.learning_rate, float) or not (
            self.learning_rate > 0.0 and torch.isfinite(torch.tensor(self.learning_rate))
        ):
            raise ValueError("NEI learning_rate must be finite and positive")

    def scientific_document(self) -> dict[str, object]:
        return {
            "policy_id": COMPOSITE_MC_NEI_PHYSICAL_MAE,
            "n_mc": self.n_mc,
            "raw_samples": self.raw_samples,
            "restarts": self.restarts,
            "max_iterations": self.max_iterations,
            "learning_rate": self.learning_rate,
            "raw_seed_coordinate": _coordinate_document(self.raw_coordinate),
            "mc_seed_coordinate": _coordinate_document(self.mc_coordinate),
            "optimizer_seed_coordinate": _coordinate_document(
                self.optimizer_coordinate
            ),
            "raw_seed": self.seed_deriver.seed_for(self.raw_coordinate),
            "mc_seed": self.seed_deriver.seed_for(self.mc_coordinate),
            "optimizer_seed": self.seed_deriver.seed_for(
                self.optimizer_coordinate
            ),
            "joint_sampling": "baseline_cached_conditional_candidate_v1",
            "baseline_factorization": NEI_BASELINE_FACTORIZATION,
            "response_samples": "latent_standardized_then_physical",
            "objective": "physical_response_l1_mae",
            "raw_recovery": NEI_RAW_RECOVERY,
        }


@dataclass(frozen=True)
class NEIRestartDiagnostic:
    raw_index: int
    status: str
    initial_value: float
    final_value: float | None
    accepted: bool
    iterations: int
    failure_type: str | None = None
    failure_explanation: str | None = None

    def document(self) -> dict[str, object]:
        return {
            "raw_index": self.raw_index,
            "status": self.status,
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "accepted": self.accepted,
            "iterations": self.iterations,
            "failure_type": self.failure_type,
            "failure_explanation": self.failure_explanation,
        }


@dataclass(frozen=True)
class NEIProposalResult:
    candidate: torch.Tensor
    value: torch.Tensor
    candidate_source: str
    selected_raw_index: int
    raw_candidates: torch.Tensor
    raw_values: torch.Tensor
    raw_valid: tuple[bool, ...]
    raw_failures: tuple[str | None, ...]
    restarts: tuple[NEIRestartDiagnostic, ...]
    base_sample_identity: str
    base_sample_shape: tuple[int, int]
    baseline_factorization: NEIBaselineFactorizationDiagnostic
    baseline_size: int
    posterior_calls: int
    elapsed_seconds: float
    config: NEIConfig

    def provenance(self) -> dict[str, object]:
        return {
            "policy_id": COMPOSITE_MC_NEI_PHYSICAL_MAE,
            "candidate": self.candidate.detach().cpu().tolist(),
            "value": float(self.value.detach().cpu()),
            "candidate_source": self.candidate_source,
            "selected_raw_index": self.selected_raw_index,
            "raw_candidates": self.raw_candidates.detach().cpu().tolist(),
            "raw_values": self.raw_values.detach().cpu().tolist(),
            "raw_valid": self.raw_valid,
            "raw_failures": self.raw_failures,
            "restarts": tuple(item.document() for item in self.restarts),
            "base_sample_identity": self.base_sample_identity,
            "base_sample_shape": self.base_sample_shape,
            "baseline_factorization": self.baseline_factorization.document(),
            "baseline_size": self.baseline_size,
            "posterior_calls": self.posterior_calls,
            "elapsed_seconds": self.elapsed_seconds,
            "configuration": self.config.scientific_document(),
            "simulator_calls": 0,
        }


@dataclass(frozen=True)
class NEICandidateEvaluation:
    values: torch.Tensor
    base_sample_identity: str
    base_sample_shape: tuple[int, int]
    baseline_factorization: NEIBaselineFactorizationDiagnostic
    baseline_size: int
    posterior_calls: int
    config: NEIConfig

    def provenance(self) -> dict[str, object]:
        return {
            "policy_id": COMPOSITE_MC_NEI_PHYSICAL_MAE,
            "values": self.values.detach().cpu().tolist(),
            "base_sample_identity": self.base_sample_identity,
            "base_sample_shape": self.base_sample_shape,
            "baseline_factorization": self.baseline_factorization.document(),
            "baseline_size": self.baseline_size,
            "posterior_calls": self.posterior_calls,
            "configuration": self.config.scientific_document(),
            "candidate_count": int(self.values.shape[0]),
            "simulator_calls": 0,
        }


def _coordinate_document(coordinate: SeedCoordinate) -> dict[str, object]:
    return {
        "namespace": coordinate.namespace,
        "coordinates": coordinate.coordinates,
    }


def _psd_factor(
    covariance: torch.Tensor, *, tolerance_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Factor a symmetric conditional covariance for Gaussian sampling.

    The eigenvalue guard uses tolerance_dtype when provided, preserving the source
    precision's roundoff scale after promotion. Materially negative eigenvalues
    are rejected; only values within that guard are clipped to zero before taking
    square roots. This sampling factor is not a repair of the fitted GP covariance.
    """
    symmetric = (0.5 * (covariance + covariance.transpose(-1, -2))).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eps = torch.finfo(tolerance_dtype or covariance.dtype).eps
    scale = eigenvalues.abs().amax().clamp_min(1.0)
    tolerance = 8.0 * covariance.shape[-1] * eps * scale
    if bool(eigenvalues[0] < -tolerance):
        raise NEIAcquisitionError("conditional latent covariance is materially indefinite")
    return (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)).contiguous()


def _promoted_baseline_cholesky(
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, NEIBaselineFactorizationDiagnostic]:
    """Factor the baseline latent covariance once in float64, without jitter.

    Symmetrization and precision promotion may resolve rounding loss for a truly
    positive-definite matrix. They cannot make an exactly singular matrix positive
    definite: failed Cholesky is an explicit NEI initialization error, not permission
    to add noise, perturb baseline points or change the covariance model.
    """

    symmetric = (0.5 * (covariance + covariance.transpose(-1, -2))).contiguous()
    promoted = symmetric.to(dtype=torch.float64).contiguous()
    factor, info = torch.linalg.cholesky_ex(promoted, check_errors=False)
    if int(info.detach().cpu()) != 0:
        raise NEIAcquisitionError(
            "baseline latent covariance is not positive definite"
        )
    return factor.contiguous(), NEIBaselineFactorizationDiagnostic(
        policy_id=NEI_BASELINE_FACTORIZATION,
        input_dtype=str(covariance.dtype),
        factor_dtype=str(factor.dtype),
        dimension=int(covariance.shape[-1]),
        promotion_occurred=covariance.dtype is not torch.float64,
        jitter_added=0.0,
        relative_jitter=0.0,
        cholesky_attempts=1,
    )


def _exact_duplicate(candidate: torch.Tensor, baseline: torch.Tensor) -> bool:
    return any(torch.equal(candidate, row) for row in baseline)


class _NEIEvaluator:
    """Monte Carlo expected physical-MAE improvement using joint latent draws.

    The ordered evaluated baseline [N,P] and target [M] share one frozen transform.
    A seeded bank of standard normals is reused across candidate queries, so
    changes in acquisition value do not reflect fresh Monte Carlo draws. Baseline
    and candidate errors are computed after returning sampled responses to physical
    units. This composite-response acquisition is distinct from scalar qLogNEI.
    """
    def __init__(
        self,
        provider: JointLatentPosteriorProvider,
        baseline: torch.Tensor,
        transform: FrozenResponseTransform,
        target_physical: torch.Tensor,
        config: NEIConfig,
    ) -> None:
        if baseline.ndim != 2 or baseline.shape[0] < 2:
            raise ValueError("NEI baseline must have shape [N,P] with N>=2")
        if target_physical.shape != (transform.response_dim,):
            raise ValueError("NEI physical target has the wrong response dimension")
        if (
            baseline.dtype != transform.dtype
            or baseline.device != transform.device
            or target_physical.dtype != transform.dtype
            or target_physical.device != transform.device
        ):
            raise ValueError("NEI baseline, target, and transform must share dtype/device")
        if provider.transform_id != transform.metadata.transform_id:
            raise ValueError("NEI surrogate and response transform identities differ")
        self.provider = provider
        self.baseline = baseline
        self.transform = transform
        self.target = target_physical
        self.config = config
        self.baseline_posterior = provider.joint_posterior(baseline)
        self.posterior_calls = 1
        dimension = self.baseline_posterior.flattened_dimension
        response_dim = self.baseline_posterior.response_dim
        generator = config.seed_deriver.generator(
            config.mc_coordinate, device=baseline.device
        )
        self.base_samples = torch.randn(
            (config.n_mc, dimension + response_dim),
            dtype=baseline.dtype,
            device=baseline.device,
            generator=generator,
        ).contiguous()
        self.base_sample_identity = stable_hash(
            "nei-base-samples-v1",
            {
                "coordinate": _coordinate_document(config.mc_coordinate),
                "seed": config.seed_deriver.seed_for(config.mc_coordinate),
                "shape": tuple(self.base_samples.shape),
                "dtype": str(self.base_samples.dtype),
                "values": self.base_samples.detach().cpu().tolist(),
            },
        )
        self.baseline_cholesky, self.baseline_factorization = (
            _promoted_baseline_cholesky(
                self.baseline_posterior.covariance.detach()
            )
        )
        self.baseline_mean = (
            self.baseline_posterior.mean.detach().reshape(-1).double()
        )
        baseline_flat = (
            self.baseline_mean
            + self.base_samples[:, :dimension].double()
            @ self.baseline_cholesky.transpose(-1, -2)
        )
        self.baseline_samples = baseline_flat.reshape(
            config.n_mc,
            self.baseline_posterior.point_count,
            response_dim,
        ).detach()
        baseline_physical = self.baseline_samples * transform.scale.double()
        baseline_physical = baseline_physical + transform.mean.double()
        self.sampled_incumbent = (
            torch.abs(baseline_physical - target_physical.double())
            .mean(dim=-1)
            .amin(dim=-1)
        ).detach()

    def value(self, candidate: torch.Tensor) -> torch.Tensor:
        """Estimate nonnegative improvement at one physical candidate [P].

        Condition the candidate tasks on the same sampled baseline values using the
        joint GP cross-covariance and its Schur complement. Average positive reductions
        from each sample's best baseline physical MAE. Exact baseline duplicates return
        zero; gradient refinement reuses the same Monte Carlo draws without
        resampling or evaluating a physical simulator.
        """
        if candidate.shape != (self.baseline.shape[1],):
            raise ValueError("NEI candidate must have shape [P]")
        if _exact_duplicate(candidate, self.baseline):
            return torch.zeros((), dtype=candidate.dtype, device=candidate.device)
        joint = self.provider.joint_posterior(
            torch.cat((self.baseline, candidate.unsqueeze(0)), dim=0).contiguous()
        )
        self.posterior_calls += 1
        if (
            joint.model_id != self.baseline_posterior.model_id
            or joint.transform_id != self.baseline_posterior.transform_id
            or joint.response_dim != self.baseline_posterior.response_dim
        ):
            raise NEIAcquisitionError("candidate joint posterior identity drift")
        baseline_dimension = self.baseline_posterior.flattened_dimension
        response_dim = joint.response_dim
        cross = joint.covariance[:baseline_dimension, baseline_dimension:].double()
        candidate_covariance = joint.covariance[
            baseline_dimension:, baseline_dimension:
        ].double()
        weights = torch.cholesky_solve(cross, self.baseline_cholesky)
        centered = (
            self.baseline_samples.reshape(self.config.n_mc, -1)
            - self.baseline_mean
        )
        conditional_mean = joint.mean[-1].double() + centered @ weights
        conditional_covariance = (
            candidate_covariance - cross.transpose(-1, -2) @ weights
        )
        factor = _psd_factor(
            conditional_covariance, tolerance_dtype=joint.covariance.dtype
        )
        candidate_samples = conditional_mean + (
            self.base_samples[:, baseline_dimension:].double()
            @ factor.transpose(-1, -2)
        )
        if candidate_samples.shape != (self.config.n_mc, response_dim):
            raise NEIAcquisitionError("candidate latent sample shape is invalid")
        candidate_physical = (
            candidate_samples * self.transform.scale.double()
            + self.transform.mean.double()
        )
        candidate_error = torch.abs(
            candidate_physical - self.target.double()
        ).mean(dim=-1)
        value = torch.relu(self.sampled_incumbent - candidate_error).mean()
        if not bool(torch.isfinite(value)):
            raise NEIAcquisitionError("NEI value is nonfinite")
        return value.to(dtype=candidate.dtype)


def _raw_candidates(
    parameter_spec: ParameterSpec,
    *,
    config: NEIConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    seed = config.seed_deriver.seed_for(config.raw_coordinate)
    unit = torch.quasirandom.SobolEngine(
        parameter_spec.dimension, scramble=True, seed=seed
    ).draw(config.raw_samples, dtype=dtype)
    lower = torch.tensor(parameter_spec.lower, dtype=dtype)
    upper = torch.tensor(parameter_spec.upper, dtype=dtype)
    return (lower + unit * (upper - lower)).to(device=device).contiguous()


def propose_composite_mc_nei(
    provider: JointLatentPosteriorProvider,
    *,
    baseline: torch.Tensor,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
    parameter_spec: ParameterSpec,
    config: NEIConfig,
) -> NEIProposalResult:
    """Optimize q=1 composite NEI with deterministic raw recovery."""

    started = perf_counter()
    evaluator = _NEIEvaluator(
        provider, baseline, transform, target_physical, config
    )
    raw = _raw_candidates(
        parameter_spec,
        config=config,
        dtype=baseline.dtype,
        device=baseline.device,
    )
    raw_values = torch.full(
        (config.raw_samples,),
        float("-inf"),
        dtype=baseline.dtype,
        device=baseline.device,
    )
    valid: list[bool] = []
    failures: list[str | None] = []
    for index in range(config.raw_samples):
        candidate = raw[index]
        if _exact_duplicate(candidate, baseline):
            valid.append(False)
            failures.append("exact_evaluated_duplicate")
            continue
        try:
            with torch.no_grad():
                raw_values[index] = evaluator.value(candidate)
        except Exception as exc:
            valid.append(False)
            failures.append(f"{type(exc).__name__}: {exc}")
        else:
            is_valid = bool(torch.isfinite(raw_values[index]))
            valid.append(is_valid)
            failures.append(None if is_valid else "nonfinite_raw_value")
    ranking = sorted(
        (index for index, item in enumerate(valid) if item),
        key=lambda index: (-float(raw_values[index].detach()), index),
    )
    if not ranking:
        raise NEIAcquisitionError("NEI raw pool contains no valid unevaluated candidate")

    lower = torch.tensor(
        parameter_spec.lower, dtype=baseline.dtype, device=baseline.device
    )
    upper = torch.tensor(
        parameter_spec.upper, dtype=baseline.dtype, device=baseline.device
    )
    accepted: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
    diagnostics: list[NEIRestartDiagnostic] = []
    for raw_index in ranking[: config.restarts]:
        initial_value = float(raw_values[raw_index].detach())
        candidate = raw[raw_index].detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([candidate], lr=config.learning_rate)
        final_value: torch.Tensor | None = None
        try:
            for _ in range(config.max_iterations):
                optimizer.zero_grad(set_to_none=True)
                value = evaluator.value(candidate)
                (-value).backward()
                optimizer.step()
                with torch.no_grad():
                    candidate.clamp_(lower, upper)
            with torch.no_grad():
                final_value = evaluator.value(candidate)
            accepted_restart = bool(
                torch.isfinite(final_value)
                and float(final_value) >= initial_value
                and not _exact_duplicate(candidate.detach(), baseline)
            )
            diagnostics.append(
                NEIRestartDiagnostic(
                    raw_index=raw_index,
                    status="completed",
                    initial_value=initial_value,
                    final_value=float(final_value.detach()),
                    accepted=accepted_restart,
                    iterations=config.max_iterations,
                )
            )
            if accepted_restart:
                accepted.append(
                    (
                        -float(final_value.detach()),
                        raw_index,
                        candidate.detach().clone().contiguous(),
                        final_value.detach().clone(),
                    )
                )
        except Exception as exc:
            diagnostics.append(
                NEIRestartDiagnostic(
                    raw_index=raw_index,
                    status="failed",
                    initial_value=initial_value,
                    final_value=(
                        None if final_value is None else float(final_value.detach())
                    ),
                    accepted=False,
                    iterations=config.max_iterations,
                    failure_type=type(exc).__name__,
                    failure_explanation=str(exc),
                )
            )

    if accepted:
        _, selected_raw, selected, selected_value = min(
            accepted, key=lambda item: (item[0], item[1])
        )
        source = "local_refinement"
    else:
        selected_raw = ranking[0]
        selected = raw[selected_raw].detach().clone().contiguous()
        selected_value = raw_values[selected_raw].detach().clone()
        source = "raw_recovery"

    return NEIProposalResult(
        candidate=selected,
        value=selected_value,
        candidate_source=source,
        selected_raw_index=selected_raw,
        raw_candidates=raw,
        raw_values=raw_values,
        raw_valid=tuple(valid),
        raw_failures=tuple(failures),
        restarts=tuple(diagnostics),
        base_sample_identity=evaluator.base_sample_identity,
        base_sample_shape=tuple(evaluator.base_samples.shape),
        baseline_factorization=evaluator.baseline_factorization,
        baseline_size=baseline.shape[0],
        posterior_calls=evaluator.posterior_calls,
        elapsed_seconds=perf_counter() - started,
        config=config,
    )


def evaluate_composite_mc_nei_candidates(
    provider: JointLatentPosteriorProvider,
    *,
    baseline: torch.Tensor,
    candidates: torch.Tensor,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
    config: NEIConfig,
) -> NEICandidateEvaluation:
    """Evaluate an ordered candidate set with one common latent base sample."""

    if candidates.ndim != 2 or candidates.shape[1] != baseline.shape[1]:
        raise ValueError("NEI candidate matrix must have shape [K,P]")
    if candidates.shape[0] < 1:
        raise ValueError("NEI candidate evaluation requires at least one candidate")
    evaluator = _NEIEvaluator(
        provider, baseline, transform, target_physical, config
    )
    values = []
    with torch.no_grad():
        for candidate in candidates:
            values.append(evaluator.value(candidate))
    return NEICandidateEvaluation(
        values=torch.stack(values).contiguous(),
        base_sample_identity=evaluator.base_sample_identity,
        base_sample_shape=tuple(evaluator.base_samples.shape),
        baseline_factorization=evaluator.baseline_factorization,
        baseline_size=baseline.shape[0],
        posterior_calls=evaluator.posterior_calls,
        config=config,
    )
