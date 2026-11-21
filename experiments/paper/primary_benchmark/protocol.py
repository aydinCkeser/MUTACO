"""Strict schemas for the primary MUTACO--NEI--Sobol benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from mutaco import stable_hash
from experiments.paper.protocol.validation import PaperContractError


PRIMARY_PROTOCOL_ID = "mutaco-primary-benchmark-population-v1"
PRIMARY_RECORD_SCHEMA = "mutaco-primary-target-method-record-v1"
PRIMARY_DECLARATION_SCHEMA = "mutaco-primary-population-declaration-v1"
PRIMARY_METHODS = ("mutaco", "nei_bo", "sobol")
PRIMARY_KNOWN_METHODS = (*PRIMARY_METHODS, "scalar_qlognei")
PRIMARY_DECISION_TIMING_KEYS = frozenset(
    {"simulator", "gp_fit", "acquisition", "terminal", "other", "total"}
)


def _nonempty(value: str, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise PaperContractError(f"{label} must be nonempty")
    return result


def _positive_seed(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PaperContractError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class PrimaryComparisonConfig:
    simulator_definition_identity: str
    simulator_configuration_identity: str
    assessment_fidelity: str
    design_campaign_seed: int
    inference_campaign_seed: int
    terminal_campaign_seed: int
    assessment_campaign_seed: int
    protocol_id: str = PRIMARY_PROTOCOL_ID
    record_schema: str = PRIMARY_RECORD_SCHEMA
    assessment_policy: str = "paired_held_out_assessment_v1"
    metric_schema: str = "primary_physical_response_metrics_v1"

    def __post_init__(self) -> None:
        for name in (
            "simulator_definition_identity",
            "simulator_configuration_identity",
            "assessment_fidelity",
        ):
            _nonempty(getattr(self, name), name)
        for name in (
            "design_campaign_seed",
            "inference_campaign_seed",
            "terminal_campaign_seed",
            "assessment_campaign_seed",
        ):
            _positive_seed(getattr(self, name), name)
        if self.protocol_id != PRIMARY_PROTOCOL_ID:
            raise PaperContractError("unsupported primary protocol")
        if self.record_schema != PRIMARY_RECORD_SCHEMA:
            raise PaperContractError("unsupported primary record schema")
        if self.assessment_policy != "paired_held_out_assessment_v1":
            raise PaperContractError("unsupported primary assessment policy")

    def document(self) -> dict[str, Any]:
        body = {
            "protocol_id": self.protocol_id,
            "record_schema": self.record_schema,
            "assessment_policy": self.assessment_policy,
            "metric_schema": self.metric_schema,
            "simulator_definition_identity": self.simulator_definition_identity,
            "simulator_configuration_identity": self.simulator_configuration_identity,
            "assessment_fidelity": self.assessment_fidelity,
            "design_campaign_seed": self.design_campaign_seed,
            "inference_campaign_seed": self.inference_campaign_seed,
            "terminal_campaign_seed": self.terminal_campaign_seed,
            "assessment_campaign_seed": self.assessment_campaign_seed,
        }
        return {
            **body,
            "configuration_identity": stable_hash(
                "primary-comparison-config-v1", body
            ),
        }

    @property
    def identity(self) -> str:
        return str(self.document()["configuration_identity"])


@dataclass(frozen=True)
class PrimaryMethodDeclaration:
    method: str
    configuration_identity: str
    search_policy_identifier: str
    terminal_policy_identifier: str

    def __post_init__(self) -> None:
        if self.method not in PRIMARY_KNOWN_METHODS:
            raise PaperContractError(f"unsupported primary method {self.method!r}")
        for name in (
            "configuration_identity",
            "search_policy_identifier",
            "terminal_policy_identifier",
        ):
            _nonempty(getattr(self, name), name)

    def document(self) -> dict[str, str]:
        return {
            "method": self.method,
            "configuration_identity": self.configuration_identity,
            "search_policy_identifier": self.search_policy_identifier,
            "terminal_policy_identifier": self.terminal_policy_identifier,
        }


def primary_population_call_preview(target_count: int, budget: int) -> dict[str, int]:
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        raise PaperContractError("primary target count must be positive")
    if budget not in {40, 100}:
        raise PaperContractError("primary budget must be 40 or 100")
    search = target_count * len(PRIMARY_METHODS) * budget
    assessment = target_count * len(PRIMARY_METHODS)
    return {
        "target_construction_calls": 0,
        "maximum_search_calls": search,
        "maximum_assessment_calls": assessment,
        "maximum_total_calls": search + assessment,
        "terminal_calls": 0,
        "scientific_calls_executed": 0,
    }


@dataclass(frozen=True)
class PrimaryPopulationDeclaration:
    campaign_id: str
    comparison: PrimaryComparisonConfig
    study_identity: str
    target_ledger_hash: str
    target_record_identities: tuple[str, ...]
    budget: int
    methods: tuple[PrimaryMethodDeclaration, ...]
    output_root: str

    def __post_init__(self) -> None:
        for name in ("campaign_id", "study_identity", "target_ledger_hash", "output_root"):
            _nonempty(getattr(self, name), name)
        if self.budget not in {40, 100}:
            raise PaperContractError("primary population budget must be 40 or 100")
        if (
            not self.target_record_identities
            or len(set(self.target_record_identities)) != len(self.target_record_identities)
            or any(not item for item in self.target_record_identities)
        ):
            raise PaperContractError("primary target identities must be unique and nonempty")
        if tuple(item.method for item in self.methods) != PRIMARY_METHODS:
            raise PaperContractError(
                "primary methods must be exactly mutaco, nei_bo, sobol in canonical order"
            )

    def document(self) -> dict[str, Any]:
        body = {
            "schema_version": PRIMARY_DECLARATION_SCHEMA,
            "state": "unexecuted",
            "execution_requires_approval": True,
            "campaign_id": self.campaign_id,
            "study_identity": self.study_identity,
            "comparison": self.comparison.document(),
            "target_ledger_hash": self.target_ledger_hash,
            "target_record_identities": list(self.target_record_identities),
            "budget": self.budget,
            "methods": [item.document() for item in self.methods],
            "output_root": self.output_root,
            "call_preview": primary_population_call_preview(
                len(self.target_record_identities), self.budget
            ),
        }
        return {
            **body,
            "declaration_identity": stable_hash(
                "primary-population-declaration-v1", body
            ),
        }


def load_primary_population_declaration(
    document: Mapping[str, Any],
) -> PrimaryPopulationDeclaration:
    if document.get("schema_version") != PRIMARY_DECLARATION_SCHEMA:
        raise PaperContractError("primary declaration schema is unsupported")
    if document.get("state") != "unexecuted" or document.get("execution_requires_approval") is not True:
        raise PaperContractError("primary declaration must remain approval-gated")
    comparison_document = document.get("comparison")
    if not isinstance(comparison_document, Mapping):
        raise PaperContractError("primary comparison is missing")
    comparison = PrimaryComparisonConfig(
        simulator_definition_identity=str(
            comparison_document.get("simulator_definition_identity", "")
        ),
        simulator_configuration_identity=str(
            comparison_document.get("simulator_configuration_identity", "")
        ),
        assessment_fidelity=str(comparison_document.get("assessment_fidelity", "")),
        design_campaign_seed=comparison_document.get("design_campaign_seed"),
        inference_campaign_seed=comparison_document.get("inference_campaign_seed"),
        terminal_campaign_seed=comparison_document.get("terminal_campaign_seed"),
        assessment_campaign_seed=comparison_document.get("assessment_campaign_seed"),
        protocol_id=str(comparison_document.get("protocol_id", "")),
        record_schema=str(comparison_document.get("record_schema", "")),
        assessment_policy=str(comparison_document.get("assessment_policy", "")),
        metric_schema=str(comparison_document.get("metric_schema", "")),
    )
    if comparison_document.get("configuration_identity") != comparison.identity:
        raise PaperContractError("primary comparison identity mismatch")
    raw_methods = document.get("methods")
    if not isinstance(raw_methods, list):
        raise PaperContractError("primary methods must be a list")
    declaration = PrimaryPopulationDeclaration(
        campaign_id=str(document.get("campaign_id", "")),
        comparison=comparison,
        study_identity=str(document.get("study_identity", "")),
        target_ledger_hash=str(document.get("target_ledger_hash", "")),
        target_record_identities=tuple(
            str(item) for item in document.get("target_record_identities", ())
        ),
        budget=document.get("budget"),
        methods=tuple(
            PrimaryMethodDeclaration(
                method=str(item.get("method", "")),
                configuration_identity=str(item.get("configuration_identity", "")),
                search_policy_identifier=str(item.get("search_policy_identifier", "")),
                terminal_policy_identifier=str(item.get("terminal_policy_identifier", "")),
            )
            for item in raw_methods
            if isinstance(item, Mapping)
        ),
        output_root=str(document.get("output_root", "")),
    )
    if declaration.document() != dict(document):
        raise PaperContractError("primary declaration is not canonical")
    return declaration


@dataclass(frozen=True)
class PrimaryTrajectoryEntry:
    evaluation_index: int
    theta: tuple[float, ...]
    response_physical: tuple[float, ...]
    observed_physical_mae: float
    best_observed_mae_so_far: float
    simulator_context: Mapping[str, Any]
    simulator_wall_seconds: float
    gp_fit_status: str | None = None
    gp_fit_seconds: float | None = None
    acquisition_value: float | None = None
    acquisition_seconds: float | None = None
    candidate_source: str | None = None
    acquisition_diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.evaluation_index, bool) or self.evaluation_index < 1:
            raise PaperContractError("trajectory evaluation index must be positive")
        if not self.theta or not self.response_physical:
            raise PaperContractError("trajectory theta and response must be nonempty")
        numbers = (
            *self.theta,
            *self.response_physical,
            self.observed_physical_mae,
            self.best_observed_mae_so_far,
            self.simulator_wall_seconds,
        )
        if any(not math.isfinite(float(value)) for value in numbers):
            raise PaperContractError("trajectory values must be finite")
        if self.observed_physical_mae < 0 or self.best_observed_mae_so_far < 0:
            raise PaperContractError("trajectory MAE values must be nonnegative")

    def document(self) -> dict[str, Any]:
        return {
            "evaluation_index": self.evaluation_index,
            "theta": list(self.theta),
            "response_physical": list(self.response_physical),
            "observed_physical_mae": self.observed_physical_mae,
            "best_observed_mae_so_far": self.best_observed_mae_so_far,
            "simulator_context": dict(self.simulator_context),
            "simulator_wall_seconds": self.simulator_wall_seconds,
            "gp_fit_status": self.gp_fit_status,
            "gp_fit_seconds": self.gp_fit_seconds,
            "acquisition_value": self.acquisition_value,
            "acquisition_seconds": self.acquisition_seconds,
            "candidate_source": self.candidate_source,
            "acquisition_diagnostics": (
                None
                if self.acquisition_diagnostics is None
                else dict(self.acquisition_diagnostics)
            ),
        }


@dataclass(frozen=True)
class PrimaryDecisionOutcome:
    method: str
    status: str
    configuration_identity: str
    search_policy_identifier: str
    terminal_policy_identifier: str
    budget: int
    attempted_search_calls: int
    successful_search_calls: int
    trajectory: tuple[PrimaryTrajectoryEntry, ...]
    recommended_theta: tuple[float, ...] | None
    terminal_metadata: Mapping[str, Any] | None
    timing: Mapping[str, float]
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_explanation: str | None = None

    def __post_init__(self) -> None:
        if self.method not in PRIMARY_KNOWN_METHODS:
            raise PaperContractError("primary outcome has an unsupported method")
        if set(self.timing) != PRIMARY_DECISION_TIMING_KEYS:
            raise PaperContractError("primary outcome timing fields are incomplete")
        timing = {key: float(value) for key, value in self.timing.items()}
        if any(not math.isfinite(value) or value < 0.0 for value in timing.values()):
            raise PaperContractError("primary outcome timing must be finite and nonnegative")
        component_sum = sum(
            timing[key]
            for key in ("simulator", "gp_fit", "acquisition", "terminal", "other")
        )
        if not math.isclose(
            component_sum,
            timing["total"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PaperContractError("primary outcome timing does not sum to total")
        if self.status not in {"recommended", "failed"}:
            raise PaperContractError("primary outcome status is invalid")
        if self.budget not in {40, 100}:
            raise PaperContractError("primary outcome budget is invalid")
        if not (0 <= self.successful_search_calls <= self.attempted_search_calls <= self.budget):
            raise PaperContractError("primary search call counts are invalid")
        if len(self.trajectory) != self.successful_search_calls:
            raise PaperContractError("trajectory length differs from successful calls")
        indices = tuple(item.evaluation_index for item in self.trajectory)
        if indices != tuple(range(1, len(indices) + 1)):
            raise PaperContractError("trajectory evaluation indices are not contiguous")
        if self.trajectory:
            parameter_dimension = len(self.trajectory[0].theta)
            response_dimension = len(self.trajectory[0].response_physical)
            running_best = float("inf")
            for item in self.trajectory:
                if (
                    len(item.theta) != parameter_dimension
                    or len(item.response_physical) != response_dimension
                ):
                    raise PaperContractError("trajectory dimensions drift within a run")
                running_best = min(running_best, item.observed_physical_mae)
                if not math.isclose(
                    item.best_observed_mae_so_far,
                    running_best,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise PaperContractError(
                        "trajectory best-observed MAE is not reconstructable"
                    )
        if self.status == "recommended":
            if (
                self.recommended_theta is None
                or self.terminal_metadata is None
                or self.successful_search_calls != self.budget
            ):
                raise PaperContractError("recommended outcome is incomplete")
            if self.failure_stage is not None:
                raise PaperContractError("recommended outcome cannot contain failure metadata")
            if self.terminal_metadata.get("simulator_calls") != 0:
                raise PaperContractError("primary terminal recommendation must be zero-call")
            if self.terminal_metadata.get("gp_fits") != 0:
                raise PaperContractError("primary terminal recommendation cannot fit a GP")
        elif self.recommended_theta is not None:
            raise PaperContractError("failed outcome cannot contain a recommendation")
        if self.method == "sobol" and any(
            item.gp_fit_status is not None
            or item.gp_fit_seconds is not None
            or item.acquisition_value is not None
            or item.acquisition_seconds is not None
            for item in self.trajectory
        ):
            raise PaperContractError("Sobol trajectory cannot contain GP/acquisition activity")

    def document(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "configuration_identity": self.configuration_identity,
            "search_policy_identifier": self.search_policy_identifier,
            "terminal_policy_identifier": self.terminal_policy_identifier,
            "budget": self.budget,
            "search_calls": {
                "attempted": self.attempted_search_calls,
                "successful": self.successful_search_calls,
            },
            "trajectory": [item.document() for item in self.trajectory],
            "recommended_theta": (
                None if self.recommended_theta is None else list(self.recommended_theta)
            ),
            "terminal_metadata": (
                None if self.terminal_metadata is None else dict(self.terminal_metadata)
            ),
            "timing": dict(self.timing),
            "failure_stage": self.failure_stage,
            "failure_type": self.failure_type,
            "failure_explanation": self.failure_explanation,
        }
