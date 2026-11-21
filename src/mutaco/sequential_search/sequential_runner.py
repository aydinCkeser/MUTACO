"""Run the manuscript's initial-design, GP, acquisition and terminal stages.

The simulator is an adapter boundary. Observation commitment, named seeds and
failure/accounting records belong here; mathematical primitives stay in their
scientific modules. A failed run returns the work already committed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import platform
from time import perf_counter
from typing import Any, Callable, Mapping

import torch

from mutaco.acquisition import (
    JointChiSquareResponseDimensionBeta,
    ProposalConfig,
    ProposalResult,
    ProposalStatus,
    propose_candidate,
    recover_best_ranked_unevaluated_raw,
)
from mutaco.design import (
    InitialDesignPolicy,
    InitialDesignResult,
    generate_initial_design,
    validate_caller_design,
)
from mutaco.identity import (
    BEST_RANKED_UNEVALUATED_RAW_POLICY,
    ExecutionIdentity,
    ScientificIdentity,
    SequentialScientificIntent,
    canonical_json_bytes,
    sequential_observation_identity,
    strict_metadata_document,
)
from mutaco.identity.sequential import (
    FIXED_FULL_STREAM_CRN_POLICY,
    PAIRED_EVALUATION_SEED_POLICY,
)
from mutaco.policies import RecommendationPolicy
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.recommendation import (
    EvaluatedDesigns,
    PhysicalL2RecommendationResult,
    PhysicalMAERecommendationResult,
    RecommendationResult,
    recommend_continuous_physical_l2,
    recommend_continuous_physical_mae,
    recommend_evaluated,
    recommend_evaluated_physical_l2,
)
from mutaco.response import FrozenResponseTransform, StandardizedTarget
from mutaco.simulator import (
    RetryableSimulationError,
    SimulationContext,
    SimulatorCapabilityDriftError,
    SimulatorCapabilitySnapshot,
    SimulatorMetadataContractError,
    SimulatorProtocol,
    TensorContractError,
    checked_evaluate,
)
from mutaco.surrogate import (
    ExactMultiTaskGPSurrogate,
    GPFitConfig,
    GPFitDiagnostic,
    GPFitFailureKind,
    GPFitStatus,
    GPWarmStartState,
    SurrogateTrainingData,
    fit_exact_multitask_gp_with_recovery,
)


class RunnerStatus(str, Enum):
    COMPLETED = "completed"
    CONFIGURATION_INVALID = "configuration_invalid"
    SIMULATOR_DEFINITION_INVALID = "simulator_definition_invalid"
    SIMULATOR_METADATA_INVALID = "simulator_metadata_invalid"
    SIMULATOR_CAPABILITY_DRIFT = "simulator_capability_drift"
    INITIAL_DESIGN_FAILED = "initial_design_failed"
    SIMULATOR_RUNTIME_FAILED = "simulator_runtime_failed"
    SIMULATOR_CONTRACT_FAILED = "simulator_contract_failed"
    SIMULATOR_ATTEMPTS_EXHAUSTED = "simulator_attempts_exhausted"
    SIMULATOR_TOTAL_ATTEMPT_BUDGET_EXHAUSTED = (
        "simulator_total_attempt_budget_exhausted"
    )
    RESPONSE_TRANSFORM_FAILED = "response_transform_failed"
    INITIAL_GP_FIT_FAILED = "initial_gp_fit_failure"
    GP_FIT_FAILED = "gp_fit_failed"
    PROPOSAL_FAILED = "proposal_failed"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    RECOMMENDATION_FAILED = "recommendation_failed"


class ObservationPhase(str, Enum):
    INITIAL = "initial"
    SEQUENTIAL = "sequential"


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    RUNTIME_FAILURE = "runtime_failure"
    CONTRACT_FAILURE = "contract_failure"
    METADATA_FAILURE = "metadata_failure"


class ProgressKind(str, Enum):
    RUN_STARTED = "run_started"
    INITIAL_DESIGN_READY = "initial_design_ready"
    SIMULATOR_ATTEMPT_STARTED = "simulator_attempt_started"
    SIMULATOR_ATTEMPT_FINISHED = "simulator_attempt_finished"
    TRANSFORM_FITTED = "transform_fitted"
    GP_FIT_STARTED = "gp_fit_started"
    GP_FIT_FINISHED = "gp_fit_finished"
    PROPOSAL_STARTED = "proposal_started"
    PROPOSAL_FINISHED = "proposal_finished"
    RECOMMENDATION_STARTED = "recommendation_started"
    RUN_FINISHED = "run_finished"


@dataclass(frozen=True)
class RunnerProgressEvent:
    kind: ProgressKind
    phase: str
    logical_index: int | None
    sequential_round: int | None
    message: str
    elapsed_seconds: float
    details: Mapping[str, Any]

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "phase": self.phase,
            "logical_index": self.logical_index,
            "sequential_round": self.sequential_round,
            "message": self.message,
            "elapsed_seconds": self.elapsed_seconds,
            "details": dict(self.details),
        }


ProgressCallback = Callable[[RunnerProgressEvent], None]


@dataclass(frozen=True)
class TerminalRecommendationContext:
    """Explicit terminal identity and seed authority for direct core dispatch."""

    policy_id: str
    target_id: str
    target_index: int
    seed_deriver: SeedDeriver

    def __post_init__(self) -> None:
        RecommendationPolicy(self.policy_id)
        if not self.target_id.strip():
            raise ValueError("terminal target_id must be nonempty")
        if (
            isinstance(self.target_index, bool)
            or not isinstance(self.target_index, int)
            or self.target_index < 0
        ):
            raise ValueError("terminal target_index must be a nonnegative integer")

    def document(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_id": self.target_id,
            "target_index": self.target_index,
            "seed_policy": "mutaco-coordinate-sha256-v1",
            "seed_family_id": self.seed_deriver.scientific_identity.value,
            "root_seed": self.seed_deriver.root_seed,
        }


@dataclass(frozen=True)
class SequentialRunnerConfig:
    """Scientific intent plus execution limits for one sequential run.

    The planned observation count is initial points plus sequential rounds.
    Retry/total-attempt limits bound backend attempts separately: retries do not
    create new logical observations. Snapshot counts retain committed states;
    terminal context supplies the separately identified recommendation handoff.
    """
    intent: SequentialScientificIntent
    device: torch.device = torch.device("cpu")
    max_attempts_per_logical_evaluation: int = 1
    max_total_simulator_attempts: int | None = None
    snapshot_observation_counts: tuple[int, ...] = ()
    terminal_recommendation_context: TerminalRecommendationContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        for name, value in (
            (
                "max_attempts_per_logical_evaluation",
                self.max_attempts_per_logical_evaluation,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_total_simulator_attempts is not None and (
            isinstance(self.max_total_simulator_attempts, bool)
            or not isinstance(self.max_total_simulator_attempts, int)
            or self.max_total_simulator_attempts < 1
        ):
            raise ValueError("max_total_simulator_attempts must be a positive integer")
        counts = tuple(self.snapshot_observation_counts)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in counts
        ):
            raise TypeError("snapshot observation counts must be integers")
        if tuple(sorted(set(counts))) != counts:
            raise ValueError("snapshot observation counts must be unique and increasing")
        if any(
            value < self.intent.initial_point_count
            or value > self.planned_observations
            for value in counts
        ):
            raise ValueError("snapshot observation count lies outside the run")
        object.__setattr__(self, "snapshot_observation_counts", counts)

    @property
    def planned_observations(self) -> int:
        return self.intent.initial_point_count + self.intent.sequential_rounds

    @property
    def effective_total_attempt_budget(self) -> int:
        return (
            self.planned_observations * self.max_attempts_per_logical_evaluation
            if self.max_total_simulator_attempts is None
            else self.max_total_simulator_attempts
        )

    def execution_document(self) -> dict[str, Any]:
        document = {
            "device": str(self.device),
            "max_attempts_per_logical_evaluation": (
                self.max_attempts_per_logical_evaluation
            ),
            "max_total_simulator_attempts": self.effective_total_attempt_budget,
            "retry_policy": "explicit_retryable_simulation_error_v1",
            "snapshot_observation_counts": self.snapshot_observation_counts,
        }
        if self.terminal_recommendation_context is not None:
            document["terminal_recommendation_context"] = (
                self.terminal_recommendation_context.document()
            )
        return document


@dataclass(frozen=True)
class SimulationAttemptRecord:
    attempt_index: int
    status: AttemptStatus
    elapsed_seconds: float
    failure_type: str | None
    failure_reason: str | None
    failure_explanation: str | None

    def provenance(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "status": self.status.value,
            "elapsed_seconds": self.elapsed_seconds,
            "failure_type": self.failure_type,
            "failure_reason": self.failure_reason,
            "failure_explanation": self.failure_explanation,
        }


@dataclass(frozen=True)
class ObservationRecord:
    """One committed physical observation and its evaluation provenance.

    theta [P] and response_physical [M] follow the declared axis order. Initial
    records have no standardized response until the initial transform is frozen;
    sequential records acquire it before commitment. The frozen dataclass does
    not make tensor storage immutable. Failed attempts remain in attempts, while
    this record represents a successful logical observation, not one backend call.
    """
    observation_id: str
    phase: ObservationPhase
    logical_index: int
    phase_index: int
    theta: torch.Tensor
    response_physical: torch.Tensor
    response_standardized: torch.Tensor | None
    simulator_metadata: Mapping[str, Any]
    context: SimulationContext
    attempts: tuple[SimulationAttemptRecord, ...]
    elapsed_seconds: float

    def provenance(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "phase": self.phase.value,
            "logical_index": self.logical_index,
            "phase_index": self.phase_index,
            "theta": self.theta.detach().cpu().tolist(),
            "response_physical": self.response_physical.detach().cpu().tolist(),
            "response_standardized": (
                self.response_standardized.detach().cpu().tolist()
                if self.response_standardized is not None
                else None
            ),
            "simulator_metadata": strict_metadata_document(self.simulator_metadata),
            "context": {
                "scientific_run_id": self.context.scientific_run_id,
                "purpose": self.context.purpose,
                "logical_index": self.context.logical_index,
                "repeat_index": self.context.repeat_index,
                "evaluation_seed": self.context.evaluation_seed,
            },
            "attempts": tuple(item.provenance() for item in self.attempts),
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class FitRecord:
    training_version: int
    diagnostic: GPFitDiagnostic

    def provenance(self) -> dict[str, Any]:
        return {
            "training_version": self.training_version,
            "status": self.diagnostic.status.value,
            "fit_id": self.diagnostic.fit_id,
            "training_data_id": self.diagnostic.training_data_id,
            "model_id": self.diagnostic.model_id,
            "elapsed_seconds": self.diagnostic.elapsed_seconds,
            "warnings": self.diagnostic.warnings,
            "failure_type": self.diagnostic.failure_type,
            "failure_message": self.diagnostic.failure_message,
            "failure_kind": (
                None
                if self.diagnostic.failure_kind is None
                else self.diagnostic.failure_kind.value
            ),
            "failure_stage": self.diagnostic.failure_stage,
            "exception_chain": self.diagnostic.exception_chain,
            "optimizer_result": self.diagnostic.optimizer_result,
            "validation_posterior_calls": (
                self.diagnostic.validation_posterior_calls
            ),
            "posterior_validation_diagnostics": (
                self.diagnostic.posterior_validation_diagnostics
            ),
            "configuration": self.diagnostic.configuration,
            "execution": self.diagnostic.execution,
            "recovery_attempts": self.diagnostic.recovery_attempts,
        }


@dataclass(frozen=True)
class SequentialRunResult:
    status: RunnerStatus
    failure_explanation: str | None
    scientific_identity: ScientificIdentity
    execution_identity: ExecutionIdentity
    capability_snapshot: SimulatorCapabilitySnapshot | None
    configuration: dict[str, Any]
    initial_design: InitialDesignResult | None
    transform: FrozenResponseTransform | None
    standardized_target: StandardizedTarget | None
    observations: tuple[ObservationRecord, ...]
    fits: tuple[FitRecord, ...]
    proposals: tuple[ProposalResult, ...]
    recommendation: (
        RecommendationResult
        | PhysicalL2RecommendationResult
        | PhysicalMAERecommendationResult
        | None
    )
    logical_accounting: dict[str, Any]
    physical_accounting: dict[str, Any]
    timings: dict[str, float]
    progress_callback_failures: tuple[str, ...]
    terminal_surrogate: ExactMultiTaskGPSurrogate | None = None
    observation_snapshots: Mapping[int, "SequentialObservationSnapshot"] | None = None

    def provenance(self) -> dict[str, Any]:
        recommendation = None
        if self.recommendation is not None:
            if isinstance(
                self.recommendation,
                (PhysicalL2RecommendationResult, PhysicalMAERecommendationResult),
            ):
                recommendation = self.recommendation.document()
            else:
                recommendation = {
                    "policy_id": self.recommendation.policy_id,
                    "classification": self.recommendation.classification,
                    "direction": self.recommendation.direction,
                    "selected_index": self.recommendation.selected_index,
                    "selected_observation_id": (
                        self.recommendation.selected_observation_id
                    ),
                    "selected_theta": (
                        self.recommendation.selected_theta.detach().cpu().tolist()
                    ),
                    "score": float(self.recommendation.score.detach().cpu()),
                    "scores": self.recommendation.scores.detach().cpu().tolist(),
                    "posterior_model_id": self.recommendation.posterior_model_id,
                    "transform_id": self.recommendation.transform_id,
                    "simulator_calls": self.recommendation.simulator_calls,
                }
        return {
            "status": self.status.value,
            "failure_explanation": self.failure_explanation,
            "scientific_identity": self.scientific_identity.value,
            "execution_identity": self.execution_identity.value,
            "configuration": self.configuration,
            "capability_snapshot": (
                {
                    "capability_id": self.capability_snapshot.capability_id,
                    "document": strict_metadata_document(
                        self.capability_snapshot.document
                    ),
                }
                if self.capability_snapshot is not None
                else None
            ),
            "initial_design": (
                self.initial_design.provenance()
                if self.initial_design is not None
                else None
            ),
            "transform": (
                self.transform.metadata.provenance_document()
                if self.transform is not None
                else None
            ),
            "standardized_target": (
                self.standardized_target.values.detach().cpu().tolist()
                if self.standardized_target is not None
                else None
            ),
            "observations": tuple(item.provenance() for item in self.observations),
            "fits": tuple(item.provenance() for item in self.fits),
            "proposals": tuple(item.provenance() for item in self.proposals),
            "recommendation": recommendation,
            "logical_accounting": self.logical_accounting,
            "physical_accounting": self.physical_accounting,
            "timings": self.timings,
            "progress_callback_failures": self.progress_callback_failures,
            "observation_snapshots": {
                str(count): snapshot.provenance()
                for count, snapshot in (self.observation_snapshots or {}).items()
            },
        }


@dataclass(frozen=True)
class SequentialObservationSnapshot:
    observation_count: int
    observations: tuple[ObservationRecord, ...]
    fits: tuple[FitRecord, ...]
    proposals: tuple[ProposalResult, ...]
    surrogate: ExactMultiTaskGPSurrogate

    def provenance(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "observation_ids": tuple(
                item.observation_id for item in self.observations
            ),
            "fit_id": self.fits[-1].diagnostic.fit_id,
            "model_id": self.surrogate.model_id,
            "proposal_count": len(self.proposals),
            "simulator_calls": 0,
            "gp_fits": 0,
            "random_draws": 0,
        }


@dataclass(frozen=True)
class _EvaluationResult:
    observation: ObservationRecord | None
    status: RunnerStatus | None
    explanation: str | None
    attempts: tuple[SimulationAttemptRecord, ...]


def _response_names(intent: SequentialScientificIntent) -> tuple[str, ...]:
    return tuple(axis.name for axis in intent.response_spec.axes)


def _emit(
    callback: ProgressCallback | None,
    failures: list[str],
    *,
    start: float,
    kind: ProgressKind,
    phase: str,
    logical_index: int | None,
    sequential_round: int | None,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    event = RunnerProgressEvent(
        kind=kind,
        phase=phase,
        logical_index=logical_index,
        sequential_round=sequential_round,
        message=message,
        elapsed_seconds=perf_counter() - start,
        details={} if details is None else dict(details),
    )
    try:
        callback(event)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")


def _capabilities_match_intent(
    intent: SequentialScientificIntent,
    snapshot: SimulatorCapabilitySnapshot,
) -> None:
    document = strict_metadata_document(snapshot.document)
    if document["simulator_identity"] != intent.simulator_identity:
        raise ValueError("simulator identity does not match scientific intent")
    if canonical_json_bytes(document["definition_metadata"]) != canonical_json_bytes(
        strict_metadata_document(intent.simulator_definition_metadata)
    ):
        raise ValueError("simulator definition metadata does not match scientific intent")
    parameter_names = tuple(item["name"] for item in document["parameter_spec"])
    response_names = tuple(item["name"] for item in document["response_spec"])
    if parameter_names != tuple(axis.name for axis in intent.parameter_spec.axes):
        raise ValueError("simulator parameter specification does not match intent")
    if response_names != tuple(axis.name for axis in intent.response_spec.axes):
        raise ValueError("simulator response specification does not match intent")
    if canonical_json_bytes(document["parameter_spec"]) != canonical_json_bytes(
        intent.scientific_document()["specifications"]["parameters"]
    ):
        raise ValueError("simulator parameter declaration does not match intent")
    if canonical_json_bytes(document["response_spec"]) != canonical_json_bytes(
        intent.scientific_document()["specifications"]["responses"]
    ):
        raise ValueError("simulator response declaration does not match intent")


def _contract_device(requested: torch.device) -> torch.device:
    """Resolve an unindexed CUDA request for exact simulator and design placement checks."""
    if requested.type == "cuda" and requested.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return requested


def _initial_design(
    config: SequentialRunnerConfig,
    seed_deriver: SeedDeriver,
) -> InitialDesignResult:
    intent = config.intent
    if intent.initial_design_policy == InitialDesignPolicy.CALLER_SUPPLIED.value:
        assert intent.caller_initial_points is not None
        theta = torch.tensor(
            intent.caller_initial_points,
            dtype=intent.dtype,
            device=config.device,
        ).contiguous()
        result = validate_caller_design(
            theta,
            parameter_spec=intent.parameter_spec,
            dtype=intent.dtype,
            device=_contract_device(config.device),
        )
        if result.theta.shape[0] != intent.initial_point_count:
            raise ValueError("caller initial point count does not match intent")
        return result
    return generate_initial_design(
        parameter_spec=intent.parameter_spec,
        point_count=intent.initial_point_count,
        seed_deriver=seed_deriver,
        seed_coordinate=SeedCoordinate("initial_design", ("initial", 0)),
        dtype=intent.dtype,
        device=config.device,
        policy=intent.initial_design_policy,
    )


def _evaluate(
    simulator: SimulatorProtocol,
    *,
    snapshot: SimulatorCapabilitySnapshot,
    theta: torch.Tensor,
    observation_id: str,
    phase: ObservationPhase,
    logical_index: int,
    phase_index: int,
    seed_deriver: SeedDeriver,
    config: SequentialRunnerConfig,
    accounting: dict[str, Any],
    run_start: float,
    callback: ProgressCallback | None,
    callback_failures: list[str],
) -> _EvaluationResult:
    """Obtain one raw response [M] for a physical parameter vector [P].

    Resolve one evaluation seed for this logical observation; retries keep it
    and vary the attempt index in the effective context. Only declared retryable
    failures retry, within both attempt limits. Update attempt accounting here,
    but leave observation-list commitment to the caller. Successful records own
    detached copies of theta/response and are not yet standardized.
    """
    purpose = phase.value
    intent = config.intent
    coordinate = SeedCoordinate(
        "simulator_evaluation",
        (purpose, logical_index, 0),
    )
    if intent.evaluation_seed_policy == FIXED_FULL_STREAM_CRN_POLICY:
        assert intent.fixed_evaluation_seed is not None
        evaluation_seed = intent.fixed_evaluation_seed
    elif intent.evaluation_seed_policy == PAIRED_EVALUATION_SEED_POLICY:
        assert intent.evaluation_seed_family_id is not None
        evaluation_seed = SeedDeriver(
            ScientificIdentity(intent.evaluation_seed_family_id), intent.root_seed
        ).seed_for(coordinate)
    else:
        evaluation_seed = seed_deriver.seed_for(coordinate)
    attempt_records: list[SimulationAttemptRecord] = []
    logical_start = perf_counter()
    for attempt_index in range(config.max_attempts_per_logical_evaluation):
        try:
            snapshot.validate_current(simulator.capabilities)
        except SimulatorCapabilityDriftError as exc:
            return _EvaluationResult(
                None,
                RunnerStatus.SIMULATOR_CAPABILITY_DRIFT,
                str(exc),
                tuple(attempt_records),
            )
        if (
            accounting["simulator_attempts"]
            >= config.effective_total_attempt_budget
        ):
            return _EvaluationResult(
                None,
                RunnerStatus.SIMULATOR_TOTAL_ATTEMPT_BUDGET_EXHAUSTED,
                "the total checked-evaluation attempt budget is exhausted",
                tuple(attempt_records),
            )
        context = SimulationContext(
            scientific_run_id=config.intent.identity().value,
            purpose=purpose,
            logical_index=logical_index,
            repeat_index=0,
            evaluation_seed=evaluation_seed,
            attempt_index=attempt_index,
        )
        accounting["simulator_attempts"] += 1
        if attempt_index > 0:
            accounting["simulator_retry_attempts"] += 1
        _emit(
            callback,
            callback_failures,
            start=run_start,
            kind=ProgressKind.SIMULATOR_ATTEMPT_STARTED,
            phase=purpose,
            logical_index=logical_index,
            sequential_round=(phase_index if phase is ObservationPhase.SEQUENTIAL else None),
            message=f"{purpose} evaluation {phase_index} attempt {attempt_index} started",
            details={
                "attempt_index": attempt_index,
                "total_attempts": accounting["simulator_attempts"],
                "total_budget": config.effective_total_attempt_budget,
            },
        )
        attempt_start = perf_counter()
        try:
            result = checked_evaluate(
                simulator,
                theta,
                device=_contract_device(config.device),
                dtype=config.intent.dtype,
                context=(
                    context
                    if bool(snapshot.document["supports_context"])
                    else None
                ),
            )
            elapsed = perf_counter() - attempt_start
            record = SimulationAttemptRecord(
                attempt_index=attempt_index,
                status=AttemptStatus.SUCCESS,
                elapsed_seconds=elapsed,
                failure_type=None,
                failure_reason=None,
                failure_explanation=None,
            )
            attempt_records.append(record)
            accounting["simulator_successes"] += 1
            accounting["simulator_seconds"] += elapsed
            _emit(
                callback,
                callback_failures,
                start=run_start,
                kind=ProgressKind.SIMULATOR_ATTEMPT_FINISHED,
                phase=purpose,
                logical_index=logical_index,
                sequential_round=(
                    phase_index if phase is ObservationPhase.SEQUENTIAL else None
                ),
                message=f"{purpose} evaluation {phase_index} succeeded",
                details={"attempt_index": attempt_index, "status": "success"},
            )
            return _EvaluationResult(
                ObservationRecord(
                    observation_id=observation_id,
                    phase=phase,
                    logical_index=logical_index,
                    phase_index=phase_index,
                    theta=theta.detach().clone().contiguous(),
                    response_physical=result.values.detach().clone().contiguous(),
                    response_standardized=None,
                    simulator_metadata=result.metadata,
                    context=context,
                    attempts=tuple(attempt_records),
                    elapsed_seconds=perf_counter() - logical_start,
                ),
                None,
                None,
                tuple(attempt_records),
            )
        except RetryableSimulationError as exc:
            elapsed = perf_counter() - attempt_start
            accounting["simulator_failures"] += 1
            accounting["simulator_seconds"] += elapsed
            attempt_records.append(
                SimulationAttemptRecord(
                    attempt_index=attempt_index,
                    status=AttemptStatus.RETRYABLE_FAILURE,
                    elapsed_seconds=elapsed,
                    failure_type=type(exc).__name__,
                    failure_reason=exc.reason.value,
                    failure_explanation=exc.explanation,
                )
            )
            _emit(
                callback,
                callback_failures,
                start=run_start,
                kind=ProgressKind.SIMULATOR_ATTEMPT_FINISHED,
                phase=purpose,
                logical_index=logical_index,
                sequential_round=(
                    phase_index if phase is ObservationPhase.SEQUENTIAL else None
                ),
                message=f"{purpose} evaluation {phase_index} requested a retry",
                details={
                    "attempt_index": attempt_index,
                    "status": "retryable_failure",
                    "reason": exc.reason.value,
                },
            )
            if attempt_index + 1 >= config.max_attempts_per_logical_evaluation:
                return _EvaluationResult(
                    None,
                    RunnerStatus.SIMULATOR_ATTEMPTS_EXHAUSTED,
                    str(exc),
                    tuple(attempt_records),
                )
        except SimulatorMetadataContractError as exc:
            elapsed = perf_counter() - attempt_start
            accounting["simulator_failures"] += 1
            accounting["simulator_seconds"] += elapsed
            attempt_records.append(
                SimulationAttemptRecord(
                    attempt_index=attempt_index,
                    status=AttemptStatus.METADATA_FAILURE,
                    elapsed_seconds=elapsed,
                    failure_type=type(exc).__name__,
                    failure_reason="metadata_contract",
                    failure_explanation=str(exc),
                )
            )
            return _EvaluationResult(
                None,
                RunnerStatus.SIMULATOR_METADATA_INVALID,
                str(exc),
                tuple(attempt_records),
            )
        except TensorContractError as exc:
            elapsed = perf_counter() - attempt_start
            accounting["simulator_failures"] += 1
            accounting["simulator_seconds"] += elapsed
            attempt_records.append(
                SimulationAttemptRecord(
                    attempt_index=attempt_index,
                    status=AttemptStatus.CONTRACT_FAILURE,
                    elapsed_seconds=elapsed,
                    failure_type=type(exc).__name__,
                    failure_reason="tensor_contract",
                    failure_explanation=str(exc),
                )
            )
            return _EvaluationResult(
                None,
                RunnerStatus.SIMULATOR_CONTRACT_FAILED,
                str(exc),
                tuple(attempt_records),
            )
        except Exception as exc:
            elapsed = perf_counter() - attempt_start
            accounting["simulator_failures"] += 1
            accounting["simulator_seconds"] += elapsed
            attempt_records.append(
                SimulationAttemptRecord(
                    attempt_index=attempt_index,
                    status=AttemptStatus.RUNTIME_FAILURE,
                    elapsed_seconds=elapsed,
                    failure_type=type(exc).__name__,
                    failure_reason="unclassified_runtime",
                    failure_explanation=str(exc),
                )
            )
            return _EvaluationResult(
                None,
                RunnerStatus.SIMULATOR_RUNTIME_FAILED,
                f"{type(exc).__name__}: {exc}",
                tuple(attempt_records),
            )
    return _EvaluationResult(
        None,
        RunnerStatus.SIMULATOR_ATTEMPTS_EXHAUSTED,
        "logical simulator attempts are exhausted",
        tuple(attempt_records),
    )


@dataclass(frozen=True)
class _FrozenResponsePreparation:
    transform: FrozenResponseTransform | None
    standardized_target: StandardizedTarget | None
    message: str
    failure_explanation: str | None = None


def _prepare_frozen_responses(
    observations: list[ObservationRecord],
    *,
    observation_ids: tuple[str, ...],
    intent: SequentialScientificIntent,
    device: torch.device,
) -> _FrozenResponsePreparation:
    """Freeze the initial [N,M] responses and standardize records/target.

    For the paper policy, mean and floored sample standard deviation (ddof=1)
    are fitted once in the declared dtype/device and response order. Replace
    records in the supplied list with standardized copies; do not refit these
    statistics when sequential observations arrive. The identity-policy branch
    is an explicit alternative, not the paper's z-score transform.
    Retain partial transform/observation state on failure, as the runner does
    for all work already committed before the failed preparation operation.
    """
    transform: FrozenResponseTransform | None = None
    standardized_target: StandardizedTarget | None = None
    transform_message = ""
    try:
        physical_initial = torch.stack(
            [item.response_physical for item in observations],
            dim=0,
        ).contiguous()
        transform_settings = strict_metadata_document(intent.transform_settings)
        transform_policy = str(
            transform_settings.get("policy", "initial_design_zscore_v1")
        )
        if transform_policy == "initial_design_zscore_v1":
            if int(transform_settings.get("ddof", 1)) != 1:
                raise ValueError("initial-design response standardization requires ddof=1")
            transform = FrozenResponseTransform.fit(
                physical_initial,
                response_spec=intent.response_spec,
                observation_ids=observation_ids,
                scale_floor=float(transform_settings.get("scale_floor", 1e-12)),
            )
            transform_message = "frozen response transform fitted"
        elif transform_policy == "identity_physical_v1":
            if transform_settings != {"policy": "identity_physical_v1"}:
                raise ValueError("identity response transform accepts only its policy identifier")
            transform = FrozenResponseTransform.identity(
                physical_initial,
                response_spec=intent.response_spec,
                observation_ids=observation_ids,
            )
            transform_message = "identity response transform materialized"
        else:
            raise ValueError(f"unknown response transform policy: {transform_policy!r}")
        response_names = _response_names(intent)
        standardized_initial = transform.to_standardized(
            physical_initial,
            response_names=response_names,
        )
        observations[:] = [
            replace(item, response_standardized=standardized_initial[index].clone())
            for index, item in enumerate(observations)
        ]
        target_physical = torch.tensor(
            intent.physical_target,
            dtype=intent.dtype,
            device=device,
        ).contiguous()
        standardized_target = transform.standardize_target(
            target_physical,
            response_names=response_names,
        )
    except Exception as exc:
        return _FrozenResponsePreparation(
            transform, standardized_target, transform_message,
            f"{type(exc).__name__}: {exc}",
        )
    return _FrozenResponsePreparation(transform, standardized_target, transform_message)


def _validate_pre_run_options(
    intent: SequentialScientificIntent,
    seed_deriver: SeedDeriver,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate options in execution order and return their strict documents."""
    gp_settings = strict_metadata_document(intent.gp_settings)
    acquisition_settings = strict_metadata_document(intent.acquisition_settings)
    beta_policy = acquisition_settings.get("beta_policy")
    if beta_policy is not None:
        if not isinstance(beta_policy, dict):
            raise ValueError("beta_policy must be a strict mapping")
        JointChiSquareResponseDimensionBeta.from_document(
            beta_policy,
            expected_response_dimension=intent.response_spec.dimension,
        )
    GPFitConfig.from_seed_deriver(
        seed_deriver,
        SeedCoordinate("gp_fit", ("training_version", 0)),
        **gp_settings,
    )
    ProposalConfig.from_seed_deriver(
        seed_deriver,
        SeedCoordinate("raw_candidates", ("sequential_round", 0)),
        **acquisition_settings,
    )
    RecommendationPolicy(intent.recommendation_policy)
    return gp_settings, acquisition_settings


def _collect_initial_observations(
    simulator: SimulatorProtocol,
    *,
    snapshot: SimulatorCapabilitySnapshot,
    initial_design: InitialDesignResult,
    seed_deriver: SeedDeriver,
    config: SequentialRunnerConfig,
    observations: list[ObservationRecord],
    logical: dict[str, int],
    physical: dict[str, Any],
    run_start: float,
    progress_callback: ProgressCallback | None,
    callback_failures: list[str],
) -> tuple[RunnerStatus, str | None] | None:
    """Evaluate design rows in order and append successful raw observations.

    Mutate observations and logical counters only after each successful result;
    _evaluate separately accounts for attempts/retries. Return None on complete
    collection, or (status, explanation) while retaining earlier committed rows.
    Frozen response standardization follows collection, not each initial call.
    """
    intent = config.intent
    for index in range(intent.initial_point_count):
        evaluated = _evaluate(
            simulator,
            snapshot=snapshot,
            theta=initial_design.theta[index].contiguous(),
            observation_id=initial_design.observation_ids[index],
            phase=ObservationPhase.INITIAL,
            logical_index=index,
            phase_index=index,
            seed_deriver=seed_deriver,
            config=config,
            accounting=physical,
            run_start=run_start,
            callback=progress_callback,
            callback_failures=callback_failures,
        )
        if evaluated.observation is None:
            assert evaluated.status is not None
            return evaluated.status, evaluated.explanation
        observations.append(evaluated.observation)
        logical["committed_observations"] += 1
        logical["initial_observations"] += 1

    return None


def _prepare_gp_training_data(
    observations: list[ObservationRecord],
    *,
    intent: SequentialScientificIntent,
    transform: FrozenResponseTransform,
) -> SurrogateTrainingData:
    """Stack committed physical theta [N,P] and standardized responses [N,M].

    Preserve observation/response order and the frozen transform identity.
    Stacking allocates tensors without mutating records; all responses must
    already be standardized. SurrogateTrainingData validates dtype/device and
    bounds. Parameter normalization and task expansion occur in _build_model.
    """
    standardized_responses = tuple(
        item.response_standardized for item in observations
    )
    if any(item is None for item in standardized_responses):
        raise ValueError(
            "every committed observation must be standardized before fitting"
        )
    data = SurrogateTrainingData(
        theta=torch.stack([item.theta for item in observations]).contiguous(),
        response_standardized=torch.stack(
            [item for item in standardized_responses if item is not None]
        ).contiguous(),
        parameter_spec=intent.parameter_spec,
        observation_ids=tuple(item.observation_id for item in observations),
        transform_id=transform.metadata.transform_id,
    )
    return data


def _standardize_and_commit_sequential_observation(
    observation: ObservationRecord,
    *,
    transform: FrozenResponseTransform,
    intent: SequentialScientificIntent,
    observations: list[ObservationRecord],
    logical: dict[str, int],
) -> str | None:
    """Apply the frozen transform to a raw [M] response, then commit a copy.

    Append the replaced record and increment logical counters only after a
    successful transform; keep the caller's raw record unchanged. Return an
    error string before commitment if standardization fails. The preceding
    simulator attempt has already been counted and is not rolled back.
    """
    try:
        standardized = transform.to_standardized(
            observation.response_physical,
            response_names=_response_names(intent),
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    observations.append(
        replace(
            observation,
            response_standardized=standardized.detach().clone().contiguous(),
        )
    )
    logical["committed_observations"] += 1
    logical["sequential_observations"] += 1
    return None


def _record_gp_fit_result(
    version: int,
    diagnostic: GPFitDiagnostic,
    *,
    fits: list[FitRecord],
    physical: dict[str, Any],
    run_start: float,
    progress_callback: ProgressCallback | None,
    callback_failures: list[str],
) -> None:
    """Record a returned diagnostic and physical work before judging fit usability."""
    fits.append(FitRecord(version, diagnostic))
    physical["gp_fit_attempts"] += sum(
        bool(item.get("fit_executed", False))
        for item in diagnostic.recovery_attempts
    )
    physical["gp_fit_seconds"] += diagnostic.elapsed_seconds
    physical["gp_fit_validation_posterior_calls"] += (
        diagnostic.validation_posterior_calls
    )
    _emit(
        progress_callback,
        callback_failures,
        start=run_start,
        kind=ProgressKind.GP_FIT_FINISHED,
        phase="fit",
        logical_index=None,
        sequential_round=version,
        message=f"GP fit for training version {version}: {diagnostic.status.value}",
        details={
            "fit_id": diagnostic.fit_id,
            "model_id": diagnostic.model_id,
            "elapsed_seconds": diagnostic.elapsed_seconds,
        },
    )


def run_sequential(
    simulator: SimulatorProtocol,
    *,
    config: SequentialRunnerConfig,
    progress_callback: ProgressCallback | None = None,
) -> SequentialRunResult:
    """Execute manuscript Steps 1--4 for one target-matching search.

    1. Generate the configured initial design, collect ordered physical
       responses, then freeze their componentwise normalization and the target.
    2. Assemble standardized training data and fit/validate the correlated GP.
    3. For each remaining logical observation, screen/refine a proposal, recover
       an exact duplicate if configured, evaluate, standardize/commit, and refit.
    4. Recommend from the final surrogate under the selected terminal policy.
       The paper policy minimizes continuous physical posterior-mean MAE.

    Configuration/capability checks and coordinate-addressed seeds precede
    design generation so invalid setup fails before a simulator attempt. The
    paper profile uses 16 initial observations; sizes/policies are supplied by
    config, not imposed here. Retries have separate physical-attempt limits.

    Return observations, diagnostics, identities and accounting, including
    committed partial state on failure; callback failures are recorded separately.
    Initial normalization is never refitted. Each sequential commitment is
    followed by a GP update, so the last update already supplies the final GP
    (or the initial fit does when there are no sequential rounds). Terminal
    recommendation adds neither a GP fit nor a simulator observation.
    """

    # Operational setup protects identities, partial-state reporting and budgets
    # before the manuscript's scientific steps can consume simulator work.
    run_start = perf_counter()
    intent = config.intent
    science = intent.identity()
    execution = ExecutionIdentity.create(
        science,
        {
            **config.execution_document(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
        },
    )
    callback_failures: list[str] = []
    observations: list[ObservationRecord] = []
    fits: list[FitRecord] = []
    proposals: list[ProposalResult] = []
    observation_snapshots: dict[int, SequentialObservationSnapshot] = {}
    initial_design: InitialDesignResult | None = None
    transform: FrozenResponseTransform | None = None
    standardized_target: StandardizedTarget | None = None
    recommendation: (
        RecommendationResult
        | PhysicalL2RecommendationResult
        | PhysicalMAERecommendationResult
        | None
    ) = None
    surrogate: ExactMultiTaskGPSurrogate | None = None
    snapshot: SimulatorCapabilitySnapshot | None = None
    logical = {
        "planned_observations": config.planned_observations,
        "committed_observations": 0,
        "initial_observations": 0,
        "sequential_observations": 0,
        "training_data_versions": 0,
        "committed_fit_events": 0,
        "committed_proposal_events": 0,
        "committed_recommendation_events": 0,
    }
    physical = {
        "simulator_attempts": 0,
        "simulator_successes": 0,
        "simulator_failures": 0,
        "simulator_retry_attempts": 0,
        "simulator_seconds": 0.0,
        "gp_fit_attempts": 0,
        "gp_fit_successes": 0,
        "gp_fit_validation_posterior_calls": 0,
        "gp_fit_seconds": 0.0,
        "proposal_calls": 0,
        "proposal_raw_posterior_calls": 0,
        "proposal_refinement_posterior_calls": 0,
        "proposal_total_posterior_calls": 0,
        "proposal_seconds": 0.0,
        "recommendation_posterior_calls": 0,
        "recommendation_seconds": 0.0,
    }

    def finish(status: RunnerStatus, explanation: str | None) -> SequentialRunResult:
        total = perf_counter() - run_start
        timings = {
            "simulator_seconds": float(physical["simulator_seconds"]),
            "gp_fit_seconds": float(physical["gp_fit_seconds"]),
            "proposal_seconds": float(physical["proposal_seconds"]),
            "recommendation_seconds": float(physical["recommendation_seconds"]),
            "total_seconds": total,
        }
        _emit(
            progress_callback,
            callback_failures,
            start=run_start,
            kind=ProgressKind.RUN_FINISHED,
            phase="terminal",
            logical_index=None,
            sequential_round=None,
            message=f"run finished with status {status.value}",
            details={"status": status.value},
        )
        return SequentialRunResult(
            status=status,
            failure_explanation=explanation,
            scientific_identity=science,
            execution_identity=execution,
            capability_snapshot=snapshot,
            configuration={
                "scientific_intent": intent.scientific_document(),
                "execution": config.execution_document(),
            },
            initial_design=initial_design,
            transform=transform,
            standardized_target=standardized_target,
            observations=tuple(observations),
            fits=tuple(fits),
            proposals=tuple(proposals),
            recommendation=recommendation,
            logical_accounting=dict(logical),
            physical_accounting=dict(physical),
            timings=timings,
            progress_callback_failures=tuple(callback_failures),
            terminal_surrogate=surrogate,
            observation_snapshots=dict(observation_snapshots),
        )

    _emit(
        progress_callback,
        callback_failures,
        start=run_start,
        kind=ProgressKind.RUN_STARTED,
        phase="setup",
        logical_index=None,
        sequential_round=None,
        message="sequential run started",
        details={
            "scientific_identity": science.value,
            "device": str(config.device),
            "planned_observations": config.planned_observations,
        },
    )
    try:
        snapshot = SimulatorCapabilitySnapshot.capture(simulator.capabilities)
        _capabilities_match_intent(intent, snapshot)
        seed_deriver = SeedDeriver(
            (
                ScientificIdentity(intent.diagnostic_internal_seed_family_id)
                if intent.diagnostic_internal_seed_family_id is not None
                else science
            ),
            intent.root_seed,
        )
        gp_settings, acquisition_settings = _validate_pre_run_options(intent, seed_deriver)
    except (SimulatorMetadataContractError, ValueError, TypeError) as exc:
        status = (
            RunnerStatus.SIMULATOR_DEFINITION_INVALID
            if snapshot is None
            else RunnerStatus.CONFIGURATION_INVALID
        )
        return finish(status, f"{type(exc).__name__}: {exc}")

    # Step 1: generate the initial design, then collect its raw responses.
    try:
        initial_design = _initial_design(config, seed_deriver)
    except Exception as exc:
        return finish(
            RunnerStatus.INITIAL_DESIGN_FAILED,
            f"{type(exc).__name__}: {exc}",
        )
    _emit(
        progress_callback,
        callback_failures,
        start=run_start,
        kind=ProgressKind.INITIAL_DESIGN_READY,
        phase="initial",
        logical_index=None,
        sequential_round=None,
        message=f"initial design contains {intent.initial_point_count} points",
        details={"design_id": initial_design.design_id},
    )

    initial_failure = _collect_initial_observations(
        simulator,
        snapshot=snapshot,
        initial_design=initial_design,
        seed_deriver=seed_deriver,
        config=config,
        observations=observations,
        logical=logical,
        physical=physical,
        run_start=run_start,
        progress_callback=progress_callback,
        callback_failures=callback_failures,
    )
    if initial_failure is not None:
        return finish(*initial_failure)

    # Step 1 (continued): freeze initial statistics; later rows use this transform.
    prepared = _prepare_frozen_responses(
        observations,
        observation_ids=initial_design.observation_ids,
        intent=intent,
        device=config.device,
    )
    transform = prepared.transform
    standardized_target = prepared.standardized_target
    if prepared.failure_explanation is not None:
        return finish(RunnerStatus.RESPONSE_TRANSFORM_FAILED, prepared.failure_explanation)
    _emit(
        progress_callback,
        callback_failures,
        start=run_start,
        kind=ProgressKind.TRANSFORM_FITTED,
        phase="initial",
        logical_index=None,
        sequential_round=None,
        message=prepared.message,
        details={"transform_id": transform.metadata.transform_id},
    )

    surrogate: ExactMultiTaskGPSurrogate | None = None
    warm_start_history: list[GPWarmStartState] = []

    # Step 2: this same fit path is reused after each Step 3 dataset augmentation.
    def fit_version(
        version: int,
    ) -> tuple[
        ExactMultiTaskGPSurrogate | None,
        str | None,
        GPFitFailureKind | None,
    ]:
        """Fit the current committed dataset using its training-version seed address.

        The initial fit and each sequential update take this same path. Recovery sees
        saved states newest first, while successful fits alone advance committed-fit
        accounting and the bounded history. A fitting failure leaves observations
        already committed; it does not undo or repeat their simulator evaluations.
        """
        _emit(
            progress_callback,
            callback_failures,
            start=run_start,
            kind=ProgressKind.GP_FIT_STARTED,
            phase="fit",
            logical_index=None,
            sequential_round=version,
            message=f"GP fit for training version {version} started",
        )
        try:
            data = _prepare_gp_training_data(
                observations, intent=intent, transform=transform,
            )
            fit_config = GPFitConfig.from_seed_deriver(
                seed_deriver,
                SeedCoordinate("gp_fit", ("training_version", version)),
                **gp_settings,
            )
            result = fit_exact_multitask_gp_with_recovery(
                data,
                config=fit_config,
                warm_start_states=tuple(reversed(warm_start_history)),
                training_version=version,
            )
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", None
        _record_gp_fit_result(
            version, result.diagnostic,
            fits=fits,
            physical=physical,
            run_start=run_start,
            progress_callback=progress_callback,
            callback_failures=callback_failures,
        )
        if result.diagnostic.status is not GPFitStatus.SUCCESS or result.surrogate is None:
            return (
                None,
                result.diagnostic.failure_message,
                result.diagnostic.failure_kind,
            )
        physical["gp_fit_successes"] += 1
        if fit_config.max_warm_start_states > 0:
            warm_start_history.append(result.surrogate.warm_start_state())
            if len(warm_start_history) > fit_config.max_warm_start_states:
                warm_start_history[:] = warm_start_history[
                    -fit_config.max_warm_start_states :
                ]
        logical["training_data_versions"] += 1
        logical["committed_fit_events"] += 1
        return result.surrogate, None, None

    surrogate, fit_failure, fit_failure_kind = fit_version(0)
    if surrogate is None:
        return finish(
            (
                RunnerStatus.INITIAL_GP_FIT_FAILED
                if fit_failure_kind is GPFitFailureKind.INITIAL_GP_FIT_FAILURE
                else RunnerStatus.GP_FIT_FAILED
            ),
            fit_failure,
        )
    if len(observations) in config.snapshot_observation_counts:
        observation_snapshots[len(observations)] = SequentialObservationSnapshot(
            len(observations), tuple(observations), tuple(fits), tuple(proposals), surrogate
        )

    # Step 3: screening/refinement -> duplicate recovery -> evaluation/commit/refit.
    for round_index in range(intent.sequential_rounds):
        proposal_config = ProposalConfig.from_seed_deriver(
            seed_deriver,
            SeedCoordinate("raw_candidates", ("sequential_round", round_index)),
            **acquisition_settings,
        )
        _emit(
            progress_callback,
            callback_failures,
            start=run_start,
            kind=ProgressKind.PROPOSAL_STARTED,
            phase="sequential",
            logical_index=intent.initial_point_count + round_index,
            sequential_round=round_index,
            message=f"proposal for sequential round {round_index} started",
        )
        proposal_start = perf_counter()
        physical["proposal_calls"] += 1
        try:
            proposal = propose_candidate(
                surrogate,
                parameter_spec=intent.parameter_spec,
                target=standardized_target,
                evaluated_theta=torch.stack(
                    [item.theta for item in observations]
                ).contiguous(),
                config=proposal_config,
            )
        except Exception as exc:
            physical["proposal_seconds"] += perf_counter() - proposal_start
            return finish(
                RunnerStatus.PROPOSAL_FAILED,
                f"{type(exc).__name__}: {exc}",
            )
        physical["proposal_seconds"] += perf_counter() - proposal_start
        proposals.append(proposal)
        physical["proposal_raw_posterior_calls"] += proposal.posterior_calls.raw_pool
        physical["proposal_refinement_posterior_calls"] += sum(
            proposal.posterior_calls.per_restart
        )
        physical["proposal_total_posterior_calls"] += proposal.posterior_calls.total
        if proposal.status is not ProposalStatus.SUCCESS or proposal.candidate is None:
            return finish(
                RunnerStatus.PROPOSAL_FAILED,
                (
                    proposal.failure_explanation
                    or (
                        proposal.failure_reason.value
                        if proposal.failure_reason is not None
                        else "proposal failed"
                    )
                ),
            )
        if proposal.duplicate is not None and proposal.duplicate.exact_duplicate:
            if intent.duplicate_policy == BEST_RANKED_UNEVALUATED_RAW_POLICY:
                recovered = recover_best_ranked_unevaluated_raw(
                    proposal,
                    evaluated_theta=torch.stack(
                        [item.theta for item in observations]
                    ).contiguous(),
                    parameter_spec=intent.parameter_spec,
                    config=proposal_config,
                    policy_id=intent.duplicate_policy,
                )
                if recovered is None:
                    return finish(
                        RunnerStatus.DUPLICATE_CANDIDATE,
                        "no finite unevaluated raw candidate remained after exact duplicate",
                    )
                proposal = recovered
                proposals[-1] = proposal
            else:
                return finish(
                    RunnerStatus.DUPLICATE_CANDIDATE,
                    "Prompt 4B returned an exact evaluated-point duplicate",
                )
        logical["committed_proposal_events"] += 1
        _emit(
            progress_callback,
            callback_failures,
            start=run_start,
            kind=ProgressKind.PROPOSAL_FINISHED,
            phase="sequential",
            logical_index=intent.initial_point_count + round_index,
            sequential_round=round_index,
            message=(
                f"round {round_index} proposal used {proposal.branch.value}/"
                f"{proposal.candidate_source.value}"
            ),
            details={
                "branch": proposal.branch.value,
                "branch_reason": proposal.branch_reason.value,
                "candidate_source": proposal.candidate_source.value,
                "candidate_id": proposal.candidate_id,
            },
        )
        assert proposal.candidate_id is not None
        logical_index = intent.initial_point_count + round_index
        observation_id = sequential_observation_identity(
            scientific_identity=science,
            global_logical_index=logical_index,
            sequential_round_index=round_index,
            candidate_id=proposal.candidate_id,
            theta=proposal.candidate,
        )
        evaluated = _evaluate(
            simulator,
            snapshot=snapshot,
            theta=proposal.candidate.detach().clone().contiguous(),
            observation_id=observation_id,
            phase=ObservationPhase.SEQUENTIAL,
            logical_index=logical_index,
            phase_index=round_index,
            seed_deriver=seed_deriver,
            config=config,
            accounting=physical,
            run_start=run_start,
            callback=progress_callback,
            callback_failures=callback_failures,
        )
        if evaluated.observation is None:
            assert evaluated.status is not None
            return finish(evaluated.status, evaluated.explanation)
        # Step 3(e): physical work is already counted; commit only after transform.
        commit_failure = _standardize_and_commit_sequential_observation(
            evaluated.observation,
            transform=transform,
            intent=intent,
            observations=observations,
            logical=logical,
        )
        if commit_failure is not None:
            return finish(RunnerStatus.RESPONSE_TRANSFORM_FAILED, commit_failure)
        surrogate, fit_failure, _ = fit_version(round_index + 1)
        if surrogate is None:
            return finish(RunnerStatus.GP_FIT_FAILED, fit_failure)
        if len(observations) in config.snapshot_observation_counts:
            observation_snapshots[len(observations)] = SequentialObservationSnapshot(
                len(observations), tuple(observations), tuple(fits), tuple(proposals), surrogate
            )

    # Step 4: the last refit already conditions on every committed observation.
    # Terminal policy dispatch below uses that GP, without another simulator call.
    _emit(
        progress_callback,
        callback_failures,
        start=run_start,
        kind=ProgressKind.RECOMMENDATION_STARTED,
        phase="recommendation",
        logical_index=None,
        sequential_round=None,
        message="final recommendation started",
    )
    recommendation_start = perf_counter()
    try:
        designs = EvaluatedDesigns(
            theta=torch.stack([item.theta for item in observations]).contiguous(),
            observation_ids=tuple(item.observation_id for item in observations),
        )
        terminal_context = config.terminal_recommendation_context
        selected_recommendation_policy = RecommendationPolicy(
            (
                terminal_context.policy_id
                if terminal_context is not None
                else intent.recommendation_policy
            )
        )
        if selected_recommendation_policy in {
            RecommendationPolicy.FINAL_POSTERIOR_STANDARDIZED_L1,
            RecommendationPolicy.EVALUATED_POSTERIOR_EXPECTED_L2,
        }:
            posterior = (
                surrogate.posterior_mean(designs.theta)
                if selected_recommendation_policy
                is RecommendationPolicy.FINAL_POSTERIOR_STANDARDIZED_L1
                else surrogate.posterior(designs.theta)
            )
            physical["recommendation_posterior_calls"] += 1
            recommendation = recommend_evaluated(
                designs,
                posterior,
                standardized_target,
                policy=intent.recommendation_policy,
            )
        else:
            physical_target = torch.tensor(
                intent.physical_target,
                dtype=transform.dtype,
                device=transform.device,
            ).contiguous()
            if (
                selected_recommendation_policy
                is RecommendationPolicy.EVALUATED_POSTERIOR_MEAN_PHYSICAL_L2
            ):
                recommendation = recommend_evaluated_physical_l2(
                    surrogate=surrogate,
                    transform=transform,
                    target_physical=physical_target,
                    target_id=(
                        terminal_context.target_id
                        if terminal_context is not None
                        else science.value
                    ),
                    theta=designs.theta,
                    observation_ids=designs.observation_ids,
                )
            else:
                recommender = (
                    recommend_continuous_physical_mae
                    if selected_recommendation_policy
                    is RecommendationPolicy.CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L1_MAE
                    else recommend_continuous_physical_l2
                )
                recommendation = recommender(
                    surrogate=surrogate,
                    transform=transform,
                    target_physical=physical_target,
                    target_id=(
                        terminal_context.target_id
                        if terminal_context is not None
                        else science.value
                    ),
                    parameter_spec=intent.parameter_spec,
                    seed_deriver=(
                        terminal_context.seed_deriver
                        if terminal_context is not None
                        else seed_deriver
                    ),
                    target_index=(
                        terminal_context.target_index
                        if terminal_context is not None
                        else 0
                    ),
                )
            physical["recommendation_posterior_calls"] += (
                recommendation.posterior_calls
            )
    except Exception as exc:
        physical["recommendation_seconds"] += perf_counter() - recommendation_start
        return finish(
            RunnerStatus.RECOMMENDATION_FAILED,
            f"{type(exc).__name__}: {exc}",
        )
    physical["recommendation_seconds"] += perf_counter() - recommendation_start
    logical["committed_recommendation_events"] = 1
    return finish(RunnerStatus.COMPLETED, None)
