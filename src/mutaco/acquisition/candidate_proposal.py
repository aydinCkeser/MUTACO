"""Manuscript Step 3(a--c): screen, choose a branch, refine and select a point.

Finite strict-interior candidates minimize the matched log barrier; otherwise
the paper policy maximizes latent uncertainty. Raw ranking and refinement
selection preserve deterministic ties. The runner owns duplicate recovery and
simulator evaluation after this module returns a proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Mapping, Protocol

import botorch
import gpytorch
import torch

from mutaco.identity import stable_hash
from mutaco.policies import ExplorationPolicy, PolicyClassification, surface_record
from mutaco.posterior import LatentPosteriorSummary
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.response import StandardizedTarget
from mutaco.types import ParameterSpec

from .confidence_geometry import FixedBeta, JointChiSquareResponseDimensionBeta
from .acquisition_objectives import MatchedBarrierObjective, evaluate_exploration


class RawCandidatePolicy(str, Enum):
    SCRAMBLED_SOBOL = "scrambled_sobol"
    IID_UNIFORM = "iid_uniform"


RAW_CANDIDATE_POLICIES: dict[RawCandidatePolicy, PolicyClassification] = {
    RawCandidatePolicy.SCRAMBLED_SOBOL: PolicyClassification.DEFAULT,
    RawCandidatePolicy.IID_UNIFORM: PolicyClassification.DIAGNOSTIC_ONLY,
}


class ProposalStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class AcquisitionBranch(str, Enum):
    BARRIER = "barrier"
    EXPLORATION = "exploration"
    NONE = "none"


class BranchReason(str, Enum):
    STRICT_INTERIOR_AVAILABLE = "strict_interior_available"
    NO_STRICT_INTERIOR = "no_strict_interior"
    RAW_POSTERIOR_FAILED = "raw_posterior_failed"
    RAW_POSTERIOR_SCORING_FAILED = "raw_posterior_scoring_failed"
    NO_FINITE_EXPLORATION = "no_finite_exploration"


class ReturnedCandidateSource(str, Enum):
    REFINED = "refined"
    RAW_FALLBACK = "raw_fallback"
    RAW_DUPLICATE_RECOVERY = "raw_duplicate_recovery"
    NONE = "none"


class ProposalFailureReason(str, Enum):
    RAW_POSTERIOR_ERROR = "raw_posterior_error"
    RAW_POSTERIOR_SCORING_ERROR = "raw_posterior_scoring_error"
    NO_FINITE_EXPLORATION_CANDIDATE = "no_finite_exploration_candidate"


class FallbackReason(str, Enum):
    NO_VALID_NONWORSENING_REFINEMENT = "no_valid_nonworsening_refinement"
    ALL_RESTARTS_FAILED = "all_restarts_failed"


class RestartStatus(str, Enum):
    COMPLETED = "completed"
    TERMINATED = "terminated"
    FAILED = "failed"


class RestartStopReason(str, Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    BACKTRACK_EXHAUSTED = "backtrack_exhausted"
    NONFINITE_OBJECTIVE = "nonfinite_objective"
    NONFINITE_GRADIENT = "nonfinite_gradient"
    NONFINITE_PARAMETER = "nonfinite_parameter"
    POSTERIOR_ERROR = "posterior_error"


class LatentPosteriorProvider(Protocol):
    """Public surrogate surface used by proposal code."""

    model_id: str
    transform_id: str
    parameter_dim: int
    dtype: torch.dtype
    device: torch.device

    def posterior(self, theta: torch.Tensor) -> LatentPosteriorSummary: ...


@dataclass(frozen=True)
class ProposalConfig:
    candidate_seed: int
    seed_coordinate: SeedCoordinate
    raw_policy: RawCandidatePolicy = RawCandidatePolicy.SCRAMBLED_SOBOL
    exploration_policy: ExplorationPolicy = ExplorationPolicy.OPERATOR_NORM_UNCERTAINTY
    beta: float = 2.0
    beta_policy: Mapping[str, object] | None = None
    alpha: float = 0.1
    tau_interior: float = 1e-8
    raw_samples: int = 64
    restarts: int = 4
    max_iterations: int = 50
    learning_rate: float = 0.05
    convergence_tolerance: float = 1e-8
    max_backtracks: int = 5
    backtrack_contraction: float = 0.5
    confidence_radius_policy: str = "operator_norm_radius_v1"
    infeasible_fallback_policy: str = "posterior_uncertainty_v1"

    @classmethod
    def from_seed_deriver(
        cls,
        seed_deriver: SeedDeriver,
        seed_coordinate: SeedCoordinate,
        **settings: object,
    ) -> "ProposalConfig":
        return cls(
            candidate_seed=seed_deriver.seed_for(seed_coordinate),
            seed_coordinate=seed_coordinate,
            **settings,
        )

    def __post_init__(self) -> None:
        if isinstance(self.candidate_seed, bool) or not isinstance(self.candidate_seed, int):
            raise TypeError("candidate_seed must be an integer")
        if self.candidate_seed < 0:
            raise ValueError("candidate_seed must be nonnegative")
        if not isinstance(self.seed_coordinate, SeedCoordinate):
            raise TypeError("seed_coordinate must be a SeedCoordinate")
        try:
            object.__setattr__(self, "raw_policy", RawCandidatePolicy(self.raw_policy))
        except ValueError as exc:
            raise ValueError(f"unknown raw-candidate policy: {self.raw_policy!r}") from exc
        try:
            object.__setattr__(
                self, "exploration_policy", ExplorationPolicy(self.exploration_policy)
            )
        except ValueError as exc:
            raise ValueError(
                f"unknown exploration policy: {self.exploration_policy!r}"
            ) from exc
        if self.confidence_radius_policy not in {
            "operator_norm_radius_v1",
            "trace_effective_dimension_radius_diagnostic_v1",
        }:
            raise ValueError("unknown confidence-radius policy")
        if self.infeasible_fallback_policy not in {
            "posterior_uncertainty_v1",
            "minimum_normalized_target_miss_diagnostic_v1",
        }:
            raise ValueError("unknown infeasible-fallback policy")
        for name, value in (
            ("raw_samples", self.raw_samples),
            ("restarts", self.restarts),
            ("max_iterations", self.max_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_backtracks, bool)
            or not isinstance(self.max_backtracks, int)
            or self.max_backtracks < 0
        ):
            raise ValueError("max_backtracks must be a nonnegative integer")
        for name, value, positive in (
            ("beta", self.beta, True),
            ("alpha", self.alpha, True),
            ("tau_interior", self.tau_interior, False),
            ("learning_rate", self.learning_rate, True),
            ("convergence_tolerance", self.convergence_tolerance, False),
        ):
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                relation = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {relation}")
        if (
            not math.isfinite(self.backtrack_contraction)
            or not 0 < self.backtrack_contraction < 1
        ):
            raise ValueError("backtrack_contraction must be finite and strictly between 0 and 1")
        if self.beta_policy is not None:
            if not isinstance(self.beta_policy, Mapping):
                raise TypeError("beta_policy must be a mapping")
            document = dict(self.beta_policy)
            dimension = document.get("response_dimension")
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise ValueError("beta_policy response_dimension must be an integer")
            resolved = JointChiSquareResponseDimensionBeta.from_document(
                document, expected_response_dimension=dimension
            )
            if not math.isclose(self.beta, resolved.value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("beta does not match beta_policy resolved value")
            object.__setattr__(self, "beta_policy", document)

    def scientific_document(self) -> dict[str, object]:
        return {
            "raw_policy": self.raw_policy.value,
            "raw_policy_classification": RAW_CANDIDATE_POLICIES[self.raw_policy].value,
            "exploration_policy": self.exploration_policy.value,
            "candidate_seed": self.candidate_seed,
            "seed_coordinate": {
                "namespace": self.seed_coordinate.namespace,
                "coordinates": self.seed_coordinate.coordinates,
            },
            "beta": self.beta,
            "beta_policy": None if self.beta_policy is None else dict(self.beta_policy),
            "alpha": self.alpha,
            "tau_interior": self.tau_interior,
            "raw_samples": self.raw_samples,
            "restarts": self.restarts,
            "max_iterations": self.max_iterations,
            "learning_rate": self.learning_rate,
            "convergence_tolerance": self.convergence_tolerance,
            "max_backtracks": self.max_backtracks,
            "backtrack_contraction": self.backtrack_contraction,
            "confidence_radius_policy": self.confidence_radius_policy,
            "infeasible_fallback_policy": self.infeasible_fallback_policy,
        }


@dataclass(frozen=True)
class CandidatePool:
    """Ordered physical raw candidates theta [H,P] and generation provenance.

    Rows supply the raw indices used for deterministic rankings and tie-breaking;
    columns follow parameter_names and the closed physical bounds. This record
    retains theta without cloning and must not be used to reorder the candidate
    population after its identity has been computed.
    """
    theta: torch.Tensor
    pool_id: str
    policy_id: str
    classification: str
    parameter_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    seed_coordinate: SeedCoordinate
    seed_value: int
    generation_device: str = "cpu"

    def provenance(self) -> dict[str, object]:
        return {
            "pool_id": self.pool_id,
            "policy_id": self.policy_id,
            "classification": self.classification,
            "parameter_names": self.parameter_names,
            "bounds": (self.lower, self.upper),
            "candidate_count": self.theta.shape[0],
            "parameter_dim": self.theta.shape[1],
            "dtype": str(self.theta.dtype).removeprefix("torch."),
            "generation_device": self.generation_device,
            "output_device": str(self.theta.device),
            "seed_coordinate": {
                "namespace": self.seed_coordinate.namespace,
                "coordinates": self.seed_coordinate.coordinates,
            },
            "seed_value": self.seed_value,
        }


@dataclass(frozen=True)
class RawRanking:
    rank: int
    raw_index: int
    score: float
    margin: float | None


@dataclass(frozen=True)
class RestartDiagnostic:
    restart_index: int
    raw_index: int
    status: RestartStatus
    stop_reason: RestartStopReason
    iterations: int
    posterior_calls: int
    accepted_steps: int
    backtracks: int
    initial_score: float
    final_score: float | None
    explanation: str | None = None

    def provenance(self) -> dict[str, object]:
        return {
            "restart_index": self.restart_index,
            "raw_index": self.raw_index,
            "status": self.status.value,
            "stop_reason": self.stop_reason.value,
            "iterations": self.iterations,
            "posterior_calls": self.posterior_calls,
            "accepted_steps": self.accepted_steps,
            "backtracks": self.backtracks,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class DuplicateDiagnostic:
    exact_duplicate: bool
    exact_duplicate_indices: tuple[int, ...]
    nearest_training_index: int
    nearest_training_distance: float
    distance_coordinates: str = "physical"
    distance_metric: str = "euclidean_l2"
    tolerance: None = None

    def provenance(self) -> dict[str, object]:
        return {
            "exact_duplicate": self.exact_duplicate,
            "exact_duplicate_indices": self.exact_duplicate_indices,
            "nearest_training_index": self.nearest_training_index,
            "nearest_training_distance": self.nearest_training_distance,
            "distance_coordinates": self.distance_coordinates,
            "distance_metric": self.distance_metric,
            "tolerance": self.tolerance,
            "selection_changed": False,
        }


@dataclass(frozen=True)
class DuplicateRecoveryDiagnostic:
    policy_id: str
    original_candidate_id: str
    original_candidate_source: str
    original_duplicate_indices: tuple[int, ...]
    recovered_raw_rank: int
    recovered_raw_index: int
    recovered_raw_score: float
    additional_posterior_calls: int = 0
    additional_simulator_calls: int = 0
    additional_gp_fits: int = 0
    additional_random_draws: int = 0

    def provenance(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "original_candidate_id": self.original_candidate_id,
            "original_candidate_source": self.original_candidate_source,
            "original_duplicate_indices": self.original_duplicate_indices,
            "recovered_raw_rank": self.recovered_raw_rank,
            "recovered_raw_index": self.recovered_raw_index,
            "recovered_raw_score": self.recovered_raw_score,
            "additional_posterior_calls": self.additional_posterior_calls,
            "additional_simulator_calls": self.additional_simulator_calls,
            "additional_gp_fits": self.additional_gp_fits,
            "additional_random_draws": self.additional_random_draws,
        }


@dataclass(frozen=True)
class PosteriorCallCounts:
    raw_pool: int
    per_restart: tuple[int, ...]
    total: int

    def provenance(self) -> dict[str, object]:
        return {
            "raw_pool": self.raw_pool,
            "per_restart": self.per_restart,
            "total": self.total,
        }


@dataclass(frozen=True)
class ProposalResult:
    status: ProposalStatus
    candidate: torch.Tensor | None
    candidate_id: str | None
    candidate_source: ReturnedCandidateSource
    branch: AcquisitionBranch
    branch_reason: BranchReason
    branch_direction: str
    pool: CandidatePool
    target_id: str
    model_id: str
    transform_id: str
    configuration: dict[str, object]
    raw_counts: dict[str, int | None]
    raw_rankings: tuple[RawRanking, ...]
    restarts: tuple[RestartDiagnostic, ...]
    posterior_calls: PosteriorCallCounts
    duplicate: DuplicateDiagnostic | None
    failure_reason: ProposalFailureReason | None
    failure_explanation: str | None
    fallback_reason: FallbackReason | None
    fallback_explanation: str | None
    execution: dict[str, object]
    timings: dict[str, float]
    simulator_calls: int = 0
    duplicate_recovery: DuplicateRecoveryDiagnostic | None = None

    def provenance(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "candidate_id": self.candidate_id,
            "candidate_source": self.candidate_source.value,
            "branch": self.branch.value,
            "branch_reason": self.branch_reason.value,
            "branch_direction": self.branch_direction,
            "pool": self.pool.provenance(),
            "target_id": self.target_id,
            "model_id": self.model_id,
            "transform_id": self.transform_id,
            "configuration": self.configuration,
            "raw_counts": self.raw_counts,
            "raw_rankings": tuple(
                {
                    "rank": item.rank,
                    "raw_index": item.raw_index,
                    "score": item.score,
                    "margin": item.margin,
                }
                for item in self.raw_rankings
            ),
            "restarts": tuple(item.provenance() for item in self.restarts),
            "posterior_calls": self.posterior_calls.provenance(),
            "duplicate": self.duplicate.provenance() if self.duplicate else None,
            "duplicate_recovery": (
                self.duplicate_recovery.provenance()
                if self.duplicate_recovery else None
            ),
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "failure_explanation": self.failure_explanation,
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
            "fallback_explanation": self.fallback_explanation,
            "execution": self.execution,
            "timings": self.timings,
            "simulator_calls": self.simulator_calls,
            "dtype": (
                str(self.candidate.dtype).removeprefix("torch.")
                if self.candidate is not None
                else str(self.pool.theta.dtype).removeprefix("torch.")
            ),
            "device": (
                str(self.candidate.device)
                if self.candidate is not None
                else str(self.pool.theta.device)
            ),
        }


@dataclass(frozen=True)
class _RestartOutcome:
    candidate: torch.Tensor | None
    score: torch.Tensor | None
    diagnostic: RestartDiagnostic


def _bounds(
    parameter_spec: ParameterSpec, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("candidate bounds require float32 or float64")
    if any(axis.transformation != "identity" for axis in parameter_spec.axes):
        raise ValueError("Prompt 4B supports identity parameter transformations only")
    bounds = torch.tensor(
        (parameter_spec.lower, parameter_spec.upper), dtype=dtype, device=device
    ).contiguous()
    if bounds.shape != (2, parameter_spec.dimension):
        raise ValueError(f"candidate bounds must have shape [2,{parameter_spec.dimension}]")
    if not bool(torch.isfinite(bounds).all()):
        raise ValueError("candidate bounds must be finite")
    if not bool((bounds[0] < bounds[1]).all()):
        raise ValueError("candidate lower bounds must be strictly below upper bounds")
    return bounds


def generate_candidate_pool(
    *,
    parameter_spec: ParameterSpec,
    config: ProposalConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> CandidatePool:
    """Return ordered physical candidates [C,P] with policy/seed identity.

    C=config.raw_samples; P is parameter dimension. Generate the declared Sobol
    or local-IID pool on CPU, affine-map to bounds, and transfer exactly once to
    the requested dtype/device. Neither a GP fit nor simulator work occurs.
    """

    output_device = torch.device(device)
    cpu_bounds = _bounds(parameter_spec, dtype=dtype, device=torch.device("cpu"))
    if config.raw_policy is RawCandidatePolicy.SCRAMBLED_SOBOL:
        unit = torch.quasirandom.SobolEngine(
            parameter_spec.dimension, scramble=True, seed=config.candidate_seed
        ).draw(config.raw_samples, dtype=dtype)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.candidate_seed)
        unit = torch.rand(
            (config.raw_samples, parameter_spec.dimension),
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
    physical_cpu = (
        cpu_bounds[0] + unit * (cpu_bounds[1] - cpu_bounds[0])
    ).contiguous()
    names = tuple(axis.name for axis in parameter_spec.axes)
    semantic = {
        "policy_id": config.raw_policy.value,
        "classification": RAW_CANDIDATE_POLICIES[config.raw_policy].value,
        "parameter_names": names,
        "bounds": (parameter_spec.lower, parameter_spec.upper),
        "dtype": str(dtype).removeprefix("torch."),
        "candidate_seed": config.candidate_seed,
        "seed_coordinate": {
            "namespace": config.seed_coordinate.namespace,
            "coordinates": config.seed_coordinate.coordinates,
        },
        "theta": physical_cpu.tolist(),
    }
    pool_id = stable_hash("candidate-pool-v1", semantic)
    theta = physical_cpu.to(device=output_device).contiguous()
    if theta.shape != (config.raw_samples, parameter_spec.dimension):
        raise ValueError("generated candidate pool has the wrong shape")
    if not bool(torch.isfinite(theta).all()):
        raise ValueError("generated candidate pool must be finite")
    output_bounds = _bounds(parameter_spec, dtype=dtype, device=output_device)
    if bool(((theta < output_bounds[0]) | (theta > output_bounds[1])).any()):
        raise ValueError("generated candidate pool lies outside declared bounds")
    return CandidatePool(
        theta=theta,
        pool_id=pool_id,
        policy_id=config.raw_policy.value,
        classification=RAW_CANDIDATE_POLICIES[config.raw_policy].value,
        parameter_names=names,
        lower=parameter_spec.lower,
        upper=parameter_spec.upper,
        seed_coordinate=config.seed_coordinate,
        seed_value=config.candidate_seed,
    )


def _target_identity(target: StandardizedTarget) -> str:
    return stable_hash(
        "standardized-target-v1",
        {
            "values": target.values.detach().cpu().tolist(),
            "response_names": target.response_names,
            "transform_id": target.transform_id,
            "dtype": str(target.values.dtype).removeprefix("torch."),
        },
    )


def _execution_provenance(
    surrogate: LatentPosteriorProvider, pool: CandidatePool
) -> dict[str, object]:
    environment_name = (
        os.environ.get("CONDA_DEFAULT_ENV")
        or os.environ.get("VIRTUAL_ENV")
        or Path(sys.prefix).name
    )
    return {
        "interpreter_path": sys.executable,
        "python_version": platform.python_version(),
        "environment_name": environment_name,
        "torch_version": str(torch.__version__),
        "botorch_version": str(botorch.__version__),
        "gpytorch_version": str(gpytorch.__version__),
        "dtype": str(surrogate.dtype).removeprefix("torch."),
        "output_device": str(surrogate.device),
        "candidate_generation_device": pool.generation_device,
        "bounded_host_extraction": {
            "policy": "bounded-control-identity-extraction-v1",
            "categories": (
                "scalar_control",
                "stable_indices",
                "finite_diagnostics",
                "duplicate_diagnostics",
                "canonical_identity_materialization",
            ),
            "cpu_objective_or_optimizer": False,
            "full_posterior_transfer": False,
            "selection_changed": False,
        },
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(surrogate.device)
            if surrogate.device.type == "cuda"
            else None
        ),
    }


def _ordered_finite_indices(values: torch.Tensor, *, minimize: bool) -> torch.Tensor:
    """Rank finite [C] scores; stable sorting retains raw order on exact ties."""
    finite_indices = torch.nonzero(torch.isfinite(values), as_tuple=False).flatten()
    if finite_indices.numel() == 0:
        return finite_indices
    finite_values = values[finite_indices]
    order = torch.argsort(finite_values, descending=not minimize, stable=True)
    return finite_indices[order]


def _ranking(
    indices: torch.Tensor, values: torch.Tensor, margin: torch.Tensor | None
) -> tuple[RawRanking, ...]:
    return tuple(
        RawRanking(
            rank=rank,
            raw_index=int(raw_index),
            score=float(values[raw_index].detach().cpu()),
            margin=(
                float(margin[raw_index].detach().cpu()) if margin is not None else None
            ),
        )
        for rank, raw_index in enumerate(indices.detach().cpu().tolist())
    )


def _score(
    surrogate: LatentPosteriorProvider,
    theta: torch.Tensor,
    *,
    branch: AcquisitionBranch,
    target: StandardizedTarget,
    config: ProposalConfig,
) -> tuple[torch.Tensor, bool]:
    """Score one physical point, preserving gradients for the selected branch.

    Barrier minimizes J in its strict interior. Default exploration maximizes
    uncertainty; the explicitly named target-miss diagnostic instead minimizes.
    The validity flag also rejects a nonfinite objective.
    """
    posterior = surrogate.posterior(theta.contiguous())
    if branch is AcquisitionBranch.BARRIER:
        evaluated = MatchedBarrierObjective(
            alpha=config.alpha,
            tau_interior=config.tau_interior,
            beta=FixedBeta(config.beta),
            radius_policy=config.confidence_radius_policy,
        ).evaluate(posterior, target)
        valid = bool(evaluated.valid_domain.item() and torch.isfinite(evaluated.values).item())
        return evaluated.values, valid
    if config.infeasible_fallback_policy == "minimum_normalized_target_miss_diagnostic_v1":
        geometry = MatchedBarrierObjective(
            alpha=config.alpha,
            tau_interior=config.tau_interior,
            beta=FixedBeta(config.beta),
            radius_policy=config.confidence_radius_policy,
        ).evaluate(posterior, target)
        covariance_scale = (
            posterior.trace
            if config.confidence_radius_policy == "trace_effective_dimension_radius_diagnostic_v1"
            else posterior.largest_eigenvalue
        )
        values = geometry.exploitation.square() / ((config.beta**2) * covariance_scale)
        return values, bool(torch.isfinite(values).item())
    evaluated = evaluate_exploration(posterior, config.exploration_policy)
    return evaluated.values, bool(torch.isfinite(evaluated.values).item())


def _barrier_restart(
    surrogate: LatentPosteriorProvider,
    start: torch.Tensor,
    *,
    raw_index: int,
    restart_index: int,
    initial_score: torch.Tensor,
    bounds: torch.Tensor,
    target: StandardizedTarget,
    config: ProposalConfig,
) -> _RestartOutcome:
    """Refine one [P] barrier start by projected gradient descent on J.

    Backtrack a bounded step until its score is finite and strictly interior;
    acceptance tests feasibility, not an Armijo or monotone-decrease condition.
    Retain the best score with strict improvement, independently of the current
    iterate. Stop on move/score tolerance, iteration/backtrack limits or an
    invalid posterior/gradient; return retained state plus termination details.
    """
    current = start.detach().clone().contiguous()
    best = current
    best_score = initial_score.detach().clone()
    posterior_calls = accepted_steps = backtracks = 0
    status = RestartStatus.COMPLETED
    stop_reason = RestartStopReason.MAX_ITERATIONS
    explanation: str | None = None
    iterations = 0
    for iteration in range(config.max_iterations):
        iterations = iteration + 1
        point = current.detach().clone().requires_grad_(True)
        try:
            value, valid = _score(
                surrogate,
                point,
                branch=AcquisitionBranch.BARRIER,
                target=target,
                config=config,
            )
            posterior_calls += 1
        except Exception as exc:
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.POSTERIOR_ERROR
            explanation = f"{type(exc).__name__}: {exc}"
            break
        if not valid or not bool(torch.isfinite(value).item()):
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.NONFINITE_OBJECTIVE
            break
        try:
            (gradient,) = torch.autograd.grad(value, point)
        except Exception as exc:
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.POSTERIOR_ERROR
            explanation = f"{type(exc).__name__}: {exc}"
            break
        if not bool(torch.isfinite(gradient).all()):
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.NONFINITE_GRADIENT
            break
        full_step = -config.learning_rate * gradient
        accepted = False
        trial_score: torch.Tensor | None = None
        trial_point: torch.Tensor | None = None
        for backtrack in range(config.max_backtracks + 1):
            scale = config.backtrack_contraction**backtrack
            trial = torch.maximum(
                torch.minimum(current + scale * full_step, bounds[1]), bounds[0]
            ).detach().contiguous()
            if not bool(torch.isfinite(trial).all()):
                continue
            try:
                candidate_score, candidate_valid = _score(
                    surrogate,
                    trial,
                    branch=AcquisitionBranch.BARRIER,
                    target=target,
                    config=config,
                )
                posterior_calls += 1
            except Exception as exc:
                status = RestartStatus.FAILED
                stop_reason = RestartStopReason.POSTERIOR_ERROR
                explanation = f"{type(exc).__name__}: {exc}"
                break
            if candidate_valid and bool(torch.isfinite(candidate_score).item()):
                accepted = True
                trial_score = candidate_score.detach()
                trial_point = trial
                backtracks += backtrack
                break
        if status is RestartStatus.FAILED:
            break
        if not accepted or trial_score is None or trial_point is None:
            status = RestartStatus.TERMINATED
            stop_reason = RestartStopReason.BACKTRACK_EXHAUSTED
            backtracks += config.max_backtracks
            break
        move = torch.linalg.vector_norm(trial_point - current)
        score_change = torch.abs(trial_score - value.detach())
        current = trial_point
        accepted_steps += 1
        if bool(trial_score < best_score):
            best = trial_point
            best_score = trial_score
        if bool(
            (move <= config.convergence_tolerance)
            | (score_change <= config.convergence_tolerance)
        ):
            stop_reason = RestartStopReason.CONVERGED
            break
    diagnostic = RestartDiagnostic(
        restart_index=restart_index,
        raw_index=raw_index,
        status=status,
        stop_reason=stop_reason,
        iterations=iterations,
        posterior_calls=posterior_calls,
        accepted_steps=accepted_steps,
        backtracks=backtracks,
        initial_score=float(initial_score.detach().cpu()),
        final_score=float(best_score.detach().cpu()),
        explanation=explanation,
    )
    return _RestartOutcome(best, best_score, diagnostic)


def _exploration_restart(
    surrogate: LatentPosteriorProvider,
    start: torch.Tensor,
    *,
    raw_index: int,
    restart_index: int,
    initial_score: torch.Tensor,
    bounds: torch.Tensor,
    target: StandardizedTarget,
    config: ProposalConfig,
) -> _RestartOutcome:
    """Refine one [P] exploration start using bounded projected Adam.

    Backpropagate -U for the paper's uncertainty maximization (the target-miss
    diagnostic instead minimizes its own score). Each Adam step is projected
    onto physical bounds. Pair pre-step scores with pre-step points, retain
    strict improvements, then score the final point if the restart did not fail.
    Move/score tolerance or the iteration limit terminates normal refinement;
    invalid values/gradients/posteriors retain their existing diagnostic status.
    """
    point = start.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([point], lr=config.learning_rate)
    best = point.detach().clone().contiguous()
    best_score = initial_score.detach().clone()
    posterior_calls = accepted_steps = 0
    status = RestartStatus.COMPLETED
    stop_reason = RestartStopReason.MAX_ITERATIONS
    explanation: str | None = None
    iterations = 0
    previous_score: torch.Tensor | None = None
    minimize = (
        config.infeasible_fallback_policy
        == "minimum_normalized_target_miss_diagnostic_v1"
    )
    for iteration in range(config.max_iterations):
        iterations = iteration + 1
        optimizer.zero_grad(set_to_none=True)
        try:
            value, valid = _score(
                surrogate,
                point,
                branch=AcquisitionBranch.EXPLORATION,
                target=target,
                config=config,
            )
            posterior_calls += 1
        except Exception as exc:
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.POSTERIOR_ERROR
            explanation = f"{type(exc).__name__}: {exc}"
            break
        if not valid or not bool(torch.isfinite(value).item()):
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.NONFINITE_OBJECTIVE
            break
        try:
            (value if minimize else -value).backward()
        except Exception as exc:
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.POSTERIOR_ERROR
            explanation = f"{type(exc).__name__}: {exc}"
            break
        if point.grad is None or not bool(torch.isfinite(point.grad).all()):
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.NONFINITE_GRADIENT
            break
        before = point.detach().clone()
        optimizer.step()
        with torch.no_grad():
            point.copy_(torch.maximum(torch.minimum(point, bounds[1]), bounds[0]))
        if not bool(torch.isfinite(point).all()):
            status = RestartStatus.FAILED
            stop_reason = RestartStopReason.NONFINITE_PARAMETER
            break
        accepted_steps += 1
        move = torch.linalg.vector_norm(point.detach() - before)
        score_change = (
            torch.abs(value.detach() - previous_score)
            if previous_score is not None
            else None
        )
        previous_score = value.detach()
        if bool(
            value.detach() < best_score
            if minimize
            else value.detach() > best_score
        ):
            best = before.contiguous()
            best_score = value.detach()
        converged_score = (
            bool(score_change <= config.convergence_tolerance)
            if score_change is not None
            else False
        )
        if bool(move <= config.convergence_tolerance) or converged_score:
            stop_reason = RestartStopReason.CONVERGED
            break
    if status is not RestartStatus.FAILED:
        try:
            final_score, final_valid = _score(
                surrogate,
                point.detach().contiguous(),
                branch=AcquisitionBranch.EXPLORATION,
                target=target,
                config=config,
            )
            posterior_calls += 1
            if final_valid and bool(
                final_score < best_score if minimize else final_score > best_score
            ):
                best = point.detach().clone().contiguous()
                best_score = final_score.detach()
        except Exception as exc:
            status = RestartStatus.TERMINATED
            stop_reason = RestartStopReason.POSTERIOR_ERROR
            explanation = f"{type(exc).__name__}: {exc}"
    diagnostic = RestartDiagnostic(
        restart_index=restart_index,
        raw_index=raw_index,
        status=status,
        stop_reason=stop_reason,
        iterations=iterations,
        posterior_calls=posterior_calls,
        accepted_steps=accepted_steps,
        backtracks=0,
        initial_score=float(initial_score.detach().cpu()),
        final_score=float(best_score.detach().cpu()),
        explanation=explanation,
    )
    return _RestartOutcome(best, best_score, diagnostic)


def _duplicate_diagnostic(
    candidate: torch.Tensor, evaluated_theta: torch.Tensor
) -> DuplicateDiagnostic:
    equality = torch.all(candidate.unsqueeze(0) == evaluated_theta, dim=-1)
    exact_indices_tensor = torch.nonzero(equality, as_tuple=False).flatten()
    distances = torch.linalg.vector_norm(
        evaluated_theta - candidate.unsqueeze(0), ord=2, dim=-1
    )
    nearest = int(torch.argmin(distances).detach().cpu())
    return DuplicateDiagnostic(
        exact_duplicate=bool(equality.any().detach().cpu()),
        exact_duplicate_indices=tuple(
            int(item) for item in exact_indices_tensor.detach().cpu().tolist()
        ),
        nearest_training_index=nearest,
        nearest_training_distance=float(distances[nearest].detach().cpu()),
    )


def recover_best_ranked_unevaluated_raw(
    proposal: ProposalResult,
    *,
    evaluated_theta: torch.Tensor,
    parameter_spec: ParameterSpec,
    config: ProposalConfig,
    policy_id: str,
) -> ProposalResult | None:
    """Return the first unevaluated point in existing finite raw-ranking order.

    Equality with evaluated_theta [N,P] is exact, not distance-tolerant. Copy the
    selected [P] pool row and update proposal identity/diagnostics without a new
    posterior or simulator call. Return None when every ranked row is already
    evaluated; the runner then reports exhaustion. Invalid recovery input raises.
    """

    if (
        proposal.status is not ProposalStatus.SUCCESS
        or proposal.candidate is None
        or proposal.candidate_id is None
        or proposal.duplicate is None
        or not proposal.duplicate.exact_duplicate
    ):
        raise ValueError("duplicate recovery requires a successful exact duplicate")
    for ranking in proposal.raw_rankings:
        candidate = proposal.pool.theta[ranking.raw_index].detach().clone().contiguous()
        duplicate = _duplicate_diagnostic(candidate, evaluated_theta)
        if duplicate.exact_duplicate:
            continue
        source = ReturnedCandidateSource.RAW_DUPLICATE_RECOVERY
        candidate_id = _candidate_identity(
            candidate,
            source=source,
            branch=proposal.branch,
            pool=proposal.pool,
            target_id=proposal.target_id,
            model_id=proposal.model_id,
            transform_id=proposal.transform_id,
            parameter_spec=parameter_spec,
            config=config,
        )
        recovery = DuplicateRecoveryDiagnostic(
            policy_id=policy_id,
            original_candidate_id=proposal.candidate_id,
            original_candidate_source=proposal.candidate_source.value,
            original_duplicate_indices=proposal.duplicate.exact_duplicate_indices,
            recovered_raw_rank=ranking.rank,
            recovered_raw_index=ranking.raw_index,
            recovered_raw_score=ranking.score,
        )
        return replace(
            proposal,
            candidate=candidate,
            candidate_id=candidate_id,
            candidate_source=source,
            duplicate=duplicate,
            duplicate_recovery=recovery,
        )
    return None


def _candidate_identity(
    candidate: torch.Tensor,
    *,
    source: ReturnedCandidateSource,
    branch: AcquisitionBranch,
    pool: CandidatePool,
    target_id: str,
    model_id: str,
    transform_id: str,
    parameter_spec: ParameterSpec,
    config: ProposalConfig,
) -> str:
    return stable_hash(
        "proposal-candidate-v1",
        {
            "candidate": candidate.detach().cpu().tolist(),
            "source": source.value,
            "branch": branch.value,
            "pool_id": pool.pool_id,
            "target_id": target_id,
            "model_id": model_id,
            "transform_id": transform_id,
            "parameter_names": tuple(axis.name for axis in parameter_spec.axes),
            "bounds": (parameter_spec.lower, parameter_spec.upper),
            "dtype": str(candidate.dtype).removeprefix("torch."),
            "configuration": config.scientific_document(),
        },
    )


def _validate_inputs(
    surrogate: LatentPosteriorProvider,
    *,
    parameter_spec: ParameterSpec,
    target: StandardizedTarget,
    evaluated_theta: torch.Tensor,
) -> torch.Tensor:
    if parameter_spec.dimension != surrogate.parameter_dim:
        raise ValueError("parameter specification does not match surrogate dimension")
    bounds = _bounds(parameter_spec, dtype=surrogate.dtype, device=surrogate.device)
    if target.transform_id != surrogate.transform_id:
        raise ValueError("target and surrogate transform identities do not match")
    # The bounds capture the current GPU for an unindexed CUDA selector.
    expected_device = surrogate.device
    if expected_device == torch.device("cuda"):
        expected_device = bounds.device
    if target.values.dtype != surrogate.dtype or target.values.device != expected_device:
        raise ValueError("target must match surrogate dtype and device")
    if not isinstance(evaluated_theta, torch.Tensor):
        raise TypeError("evaluated_theta must be a torch.Tensor")
    if evaluated_theta.ndim != 2 or evaluated_theta.shape[1] != parameter_spec.dimension:
        raise ValueError(
            f"evaluated_theta must have shape [N,{parameter_spec.dimension}]"
        )
    if evaluated_theta.shape[0] < 1:
        raise ValueError("evaluated_theta must contain at least one training point")
    if (
        evaluated_theta.dtype != surrogate.dtype
        or evaluated_theta.device != expected_device
    ):
        raise ValueError("evaluated_theta must match surrogate dtype and device")
    if not evaluated_theta.is_contiguous() or not bool(torch.isfinite(evaluated_theta).all()):
        raise ValueError("evaluated_theta must be contiguous and finite")
    if bool(
        ((evaluated_theta < bounds[0]) | (evaluated_theta > bounds[1])).any()
    ):
        raise ValueError("evaluated_theta lies outside declared parameter bounds")
    return bounds


def _screen_and_rank_raw_candidates(
    raw_posterior: LatentPosteriorSummary,
    target: StandardizedTarget,
    config: ProposalConfig,
) -> tuple[
    AcquisitionBranch, BranchReason, torch.Tensor, torch.Tensor, bool,
    int, int | None, tuple[RawRanking, ...],
]:
    """Choose and rank the branch from a raw [C,M] mean/[C,M,M] covariance.

    Return (branch, reason, values [C], ranked_indices [K], minimize,
    barrier_finite_count, exploration_finite_count_or_None, rankings).
    K counts finite branch scores, in stable best-first raw order. Barrier
    requires a finite J with margin>tau_interior (Delta>epsilon_Delta); boundary
    compatibility alone does not suffice. Otherwise use configured exploration,
    maximized by default or minimized for the explicit target-miss diagnostic.
    Preserve tensor ownership/dtype/device and consume no random numbers.
    """
    barrier = MatchedBarrierObjective(
        alpha=config.alpha,
        tau_interior=config.tau_interior,
        beta=FixedBeta(config.beta),
        radius_policy=config.confidence_radius_policy,
    ).evaluate(raw_posterior, target)
    barrier_values = barrier.values
    barrier_indices = _ordered_finite_indices(barrier_values, minimize=True)
    barrier_valid_count = int(barrier_indices.numel())
    if barrier_indices.numel() > 0:
        branch = AcquisitionBranch.BARRIER
        branch_reason = BranchReason.STRICT_INTERIOR_AVAILABLE
        values = barrier_values
        indices = barrier_indices
        margin: torch.Tensor | None = barrier.margin
        minimize = True
        exploration_finite_count: int | None = None
    else:
        targeted = (
            config.infeasible_fallback_policy
            == "minimum_normalized_target_miss_diagnostic_v1"
        )
        if targeted:
            covariance_scale = (
                raw_posterior.trace
                if config.confidence_radius_policy
                == "trace_effective_dimension_radius_diagnostic_v1"
                else raw_posterior.largest_eigenvalue
            )
            exploration_values = barrier.exploitation.square() / (
                (config.beta**2) * covariance_scale
            )
        else:
            exploration_values = evaluate_exploration(
                raw_posterior, config.exploration_policy
            ).values
        branch = AcquisitionBranch.EXPLORATION
        branch_reason = BranchReason.NO_STRICT_INTERIOR
        values = exploration_values
        indices = _ordered_finite_indices(values, minimize=targeted)
        margin = None
        minimize = targeted
        exploration_finite_count = int(indices.numel())
    rankings = _ranking(indices, values, margin)
    return (
        branch, branch_reason, values, indices, minimize,
        barrier_valid_count, exploration_finite_count, rankings,
    )


def _select_nonworsening_refined_outcome(
    outcomes: list[_RestartOutcome],
    *,
    bounds: torch.Tensor,
    raw_best_score: torch.Tensor,
    minimize: bool,
) -> _RestartOutcome | None:
    """Select the earliest best valid refinement relative to the global raw best.

    Barrier scores are minimized; uncertainty scores are maximized. Equality to
    the raw best is eligible (<= or >=), whereas replacing the current selection
    requires strict improvement. Failed, missing, nonfinite or out-of-box outcomes
    are skipped in supplied order. Return the original outcome object, or None
    so the caller can retain its raw fallback; no optimization or RNG use occurs.
    """
    selected_outcome: _RestartOutcome | None = None
    for outcome in outcomes:
        if (
            outcome.diagnostic.status is RestartStatus.FAILED
            or outcome.candidate is None
            or outcome.score is None
            or not bool(torch.isfinite(outcome.score).item())
            or not bool(torch.isfinite(outcome.candidate).all())
            or bool(
                (
                    (outcome.candidate < bounds[0])
                    | (outcome.candidate > bounds[1])
                ).any()
            )
        ):
            continue
        nonworsening = (
            bool(outcome.score <= raw_best_score)
            if minimize
            else bool(outcome.score >= raw_best_score)
        )
        if not nonworsening:
            continue
        if selected_outcome is None:
            selected_outcome = outcome
            continue
        strictly_better = (
            bool(outcome.score < selected_outcome.score)
            if minimize
            else bool(outcome.score > selected_outcome.score)
        )
        if strictly_better:
            selected_outcome = outcome
        # Exact equal scores retain the existing outcome. Outcomes are emitted
        # in raw-rank order and then restart order, so this implements the
        # accepted deterministic tie rule without a tolerance.
    return selected_outcome


def propose_candidate(
    surrogate: LatentPosteriorProvider,
    *,
    parameter_spec: ParameterSpec,
    target: StandardizedTarget,
    evaluated_theta: torch.Tensor,
    config: ProposalConfig,
) -> ProposalResult:
    """Screen, refine and select one physical [P] candidate (Steps 3(a--c)).

    Generate the named-seed [C,P] pool, obtain latent moments, rank finite scores,
    and refine up to config.restarts starts. Keep a valid result no worse than
    the raw best, otherwise fall back to that raw best. Exact ties retain raw
    ranking/restart order. Inputs and returned tensors use surrogate dtype/device.
    Report posterior/refinement failures in ProposalResult where handled below;
    input-contract and other uncaught errors propagate to the runner. This
    function records exact duplication but leaves recovery to Step 3(d) there.
    """

    total_start = perf_counter()
    bounds = _validate_inputs(
        surrogate,
        parameter_spec=parameter_spec,
        target=target,
        evaluated_theta=evaluated_theta,
    )
    generation_start = perf_counter()
    pool = generate_candidate_pool(
        parameter_spec=parameter_spec,
        config=config,
        dtype=surrogate.dtype,
        device=surrogate.device,
    )
    generation_seconds = perf_counter() - generation_start
    target_id = _target_identity(target)
    raw_start = perf_counter()
    try:
        raw_posterior = surrogate.posterior(pool.theta)
    except Exception as exc:
        timings = {
            "candidate_generation_seconds": generation_seconds,
            "raw_scoring_seconds": perf_counter() - raw_start,
            "refinement_seconds": 0.0,
            "total_seconds": perf_counter() - total_start,
        }
        return ProposalResult(
            status=ProposalStatus.FAILED,
            candidate=None,
            candidate_id=None,
            candidate_source=ReturnedCandidateSource.NONE,
            branch=AcquisitionBranch.NONE,
            branch_reason=BranchReason.RAW_POSTERIOR_FAILED,
            branch_direction="none",
            pool=pool,
            target_id=target_id,
            model_id=surrogate.model_id,
            transform_id=surrogate.transform_id,
            configuration=config.scientific_document(),
            raw_counts={
                "candidate_count": config.raw_samples,
                "strict_interior_finite": None,
                "exploration_finite": None,
            },
            raw_rankings=(),
            restarts=(),
            posterior_calls=PosteriorCallCounts(raw_pool=1, per_restart=(), total=1),
            duplicate=None,
            failure_reason=ProposalFailureReason.RAW_POSTERIOR_ERROR,
            failure_explanation=f"{type(exc).__name__}: {exc}",
            fallback_reason=None,
            fallback_explanation=None,
            execution=_execution_provenance(surrogate, pool),
            timings=timings,
        )
    try:
        (
            branch, branch_reason, values, indices, minimize,
            barrier_valid_count, exploration_finite_count, rankings,
        ) = _screen_and_rank_raw_candidates(raw_posterior, target, config)
    except Exception as exc:
        timings = {
            "candidate_generation_seconds": generation_seconds,
            "raw_scoring_seconds": perf_counter() - raw_start,
            "refinement_seconds": 0.0,
            "total_seconds": perf_counter() - total_start,
        }
        return ProposalResult(
            status=ProposalStatus.FAILED,
            candidate=None,
            candidate_id=None,
            candidate_source=ReturnedCandidateSource.NONE,
            branch=AcquisitionBranch.NONE,
            branch_reason=BranchReason.RAW_POSTERIOR_SCORING_FAILED,
            branch_direction="none",
            pool=pool,
            target_id=target_id,
            model_id=raw_posterior.model_id,
            transform_id=raw_posterior.transform_id,
            configuration=config.scientific_document(),
            raw_counts={
                "candidate_count": config.raw_samples,
                "strict_interior_finite": None,
                "exploration_finite": None,
            },
            raw_rankings=(),
            restarts=(),
            posterior_calls=PosteriorCallCounts(raw_pool=1, per_restart=(), total=1),
            duplicate=None,
            failure_reason=ProposalFailureReason.RAW_POSTERIOR_SCORING_ERROR,
            failure_explanation=f"{type(exc).__name__}: {exc}",
            fallback_reason=None,
            fallback_explanation=None,
            execution=_execution_provenance(surrogate, pool),
            timings=timings,
        )
    raw_seconds = perf_counter() - raw_start
    if indices.numel() == 0:
        timings = {
            "candidate_generation_seconds": generation_seconds,
            "raw_scoring_seconds": raw_seconds,
            "refinement_seconds": 0.0,
            "total_seconds": perf_counter() - total_start,
        }
        return ProposalResult(
            status=ProposalStatus.FAILED,
            candidate=None,
            candidate_id=None,
            candidate_source=ReturnedCandidateSource.NONE,
            branch=AcquisitionBranch.EXPLORATION,
            branch_reason=BranchReason.NO_FINITE_EXPLORATION,
            branch_direction=(
                "minimize"
                if config.infeasible_fallback_policy
                == "minimum_normalized_target_miss_diagnostic_v1"
                else "maximize"
            ),
            pool=pool,
            target_id=target_id,
            model_id=raw_posterior.model_id,
            transform_id=raw_posterior.transform_id,
            configuration=config.scientific_document(),
            raw_counts={
                "candidate_count": config.raw_samples,
                "strict_interior_finite": barrier_valid_count,
                "exploration_finite": exploration_finite_count,
            },
            raw_rankings=(),
            restarts=(),
            posterior_calls=PosteriorCallCounts(raw_pool=1, per_restart=(), total=1),
            duplicate=None,
            failure_reason=ProposalFailureReason.NO_FINITE_EXPLORATION_CANDIDATE,
            failure_explanation="the exploration branch contained no finite raw score",
            fallback_reason=None,
            fallback_explanation=None,
            execution=_execution_provenance(surrogate, pool),
            timings=timings,
        )
    restart_count = min(config.restarts, int(indices.numel()))
    selected_indices = indices[:restart_count]
    raw_best_index = int(indices[0].detach().cpu())
    raw_best = pool.theta[raw_best_index].detach().clone().contiguous()
    raw_best_score = values[raw_best_index].detach().clone()
    refinement_start = perf_counter()
    outcomes: list[_RestartOutcome] = []
    for restart_index, raw_index_tensor in enumerate(selected_indices):
        raw_index = int(raw_index_tensor.detach().cpu())
        start = pool.theta[raw_index]
        initial_score = values[raw_index]
        if branch is AcquisitionBranch.BARRIER:
            outcome = _barrier_restart(
                surrogate,
                start,
                raw_index=raw_index,
                restart_index=restart_index,
                initial_score=initial_score,
                bounds=bounds,
                target=target,
                config=config,
            )
        else:
            outcome = _exploration_restart(
                surrogate,
                start,
                raw_index=raw_index,
                restart_index=restart_index,
                initial_score=initial_score,
                bounds=bounds,
                target=target,
                config=config,
            )
        outcomes.append(outcome)
    refinement_seconds = perf_counter() - refinement_start
    selected_outcome = _select_nonworsening_refined_outcome(
        outcomes, bounds=bounds, raw_best_score=raw_best_score, minimize=minimize,
    )
    if selected_outcome is not None:
        assert selected_outcome.candidate is not None
        candidate = selected_outcome.candidate
        source = ReturnedCandidateSource.REFINED
    else:
        candidate = raw_best
        source = ReturnedCandidateSource.RAW_FALLBACK
    all_failed = all(
        outcome.diagnostic.status is RestartStatus.FAILED for outcome in outcomes
    )
    fallback_reason = (
        None
        if source is ReturnedCandidateSource.REFINED
        else (
            FallbackReason.ALL_RESTARTS_FAILED
            if all_failed
            else FallbackReason.NO_VALID_NONWORSENING_REFINEMENT
        )
    )
    fallback_explanation = (
        None
        if fallback_reason is None
        else (
            "every configured restart failed; returned the best ranked raw candidate"
            if fallback_reason is FallbackReason.ALL_RESTARTS_FAILED
            else "no restart produced a valid non-worsening refined candidate"
        )
    )
    duplicate = _duplicate_diagnostic(candidate, evaluated_theta)
    candidate_id = _candidate_identity(
        candidate,
        source=source,
        branch=branch,
        pool=pool,
        target_id=target_id,
        model_id=raw_posterior.model_id,
        transform_id=raw_posterior.transform_id,
        parameter_spec=parameter_spec,
        config=config,
    )
    diagnostics = tuple(outcome.diagnostic for outcome in outcomes)
    per_restart = tuple(item.posterior_calls for item in diagnostics)
    calls = PosteriorCallCounts(
        raw_pool=1,
        per_restart=per_restart,
        total=1 + sum(per_restart),
    )
    timings = {
        "candidate_generation_seconds": generation_seconds,
        "raw_scoring_seconds": raw_seconds,
        "refinement_seconds": refinement_seconds,
        "total_seconds": perf_counter() - total_start,
    }
    return ProposalResult(
        status=ProposalStatus.SUCCESS,
        candidate=candidate,
        candidate_id=candidate_id,
        candidate_source=source,
        branch=branch,
        branch_reason=branch_reason,
        branch_direction="minimize" if minimize else "maximize",
        pool=pool,
        target_id=target_id,
        model_id=raw_posterior.model_id,
        transform_id=raw_posterior.transform_id,
        configuration=config.scientific_document(),
        raw_counts={
            "candidate_count": config.raw_samples,
            "strict_interior_finite": barrier_valid_count,
            "exploration_finite": exploration_finite_count,
        },
        raw_rankings=rankings,
        restarts=diagnostics,
        posterior_calls=calls,
        duplicate=duplicate,
        failure_reason=None,
        failure_explanation=None,
        fallback_reason=fallback_reason,
        fallback_explanation=fallback_explanation,
        execution=_execution_provenance(surrogate, pool),
        timings=timings,
    )


def candidate_policy_support_markdown() -> str:
    descriptions = {
        RawCandidatePolicy.SCRAMBLED_SOBOL: (
            "Seeded scrambled Sobol points mapped from [0,1]^P to physical bounds"
        ),
        RawCandidatePolicy.IID_UNIFORM: (
            "Seeded IID uniform points mapped from [0,1]^P to physical bounds"
        ),
    }
    lines = [
        "# Prompt 4B candidate-policy support",
        "",
        "This table is generated from `mutaco.acquisition`; do not edit it independently.",
        "",
        "| Identifier | Classification | Surface | Behavior | Generation device |",
        "|---|---|---|---|---|",
    ]
    for policy, classification in RAW_CANDIDATE_POLICIES.items():
        lines.append(
            f"| `{policy.value}` | {classification.value} | "
            f"`{surface_record(policy.value).surface.value}` | "
            f"{descriptions[policy]} | CPU, then one recorded transfer |"
        )
    lines.extend(
        [
            "",
            "Latin hypercube and caller-supplied raw pools are rejected as unsupported.",
        ]
    )
    return "\n".join(lines) + "\n"
