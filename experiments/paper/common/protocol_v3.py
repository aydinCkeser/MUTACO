"""Prospective v3 paper-comparison primitives.

This module implements the common protocol frozen by Decision 0043. It does
not run MUTACO or SBI inference; Prompt 20E's experiment-owned direct-posterior
adapter supplies validated decision outcomes to this persistence boundary.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from mutaco import (
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    canonical_json_bytes,
    stable_hash,
)
from experiments.contracts import PhysicalCallCounts
from experiments.benchmarking.methods.sbi_zoo_failures import (
    SbiZooScoringContractError,
)

from .validation import PaperContractError, validate_method_ids

# Reuse the canonical module-name object, including pickle memoization identity.
PhysicalCallCounts.__module__ = __name__


PROTOCOL_ID = "paper_final_decision_and_paired_assessment_v1"
RECORD_SCHEMA_V2 = "mutaco-paper-target-method-record-v2"
RECORD_SCHEMA_V3 = "mutaco-paper-target-method-record-v3"
ASSESSMENT_POLICY_ID = "paired_held_out_assessment_v1"
METRIC_SCHEMA_ID = "physical_response_and_parameter_errors_v1"
FREEZE_SCHEMA = "mutaco-paper-decision-freeze-v1"

V3_METRIC_FIELDS = (
    "response_error_l1_sum",
    "response_error_mae",
    "response_error_l2",
    "parameter_error_l2",
)
LEGACY_METRIC_FIELDS = (
    "target_error_l1",
    "target_error_l2",
    "design_error_l1",
    "design_error_l2",
)
LEDGER_CATEGORIES = (
    "target_construction_or_reuse",
    "method_inference_or_training",
    "terminal_recommendation",
    "paired_assessment_and_metric",
)


@dataclass(frozen=True)
class AssessmentEvaluationV3:
    """One physical response plus auditable named-randomness evidence."""

    response: tuple[float, ...]
    randomness_evidence: Mapping[str, Any]
    raw_randomness_evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", _finite_vector(self.response, "assessment_response"))
        if not isinstance(self.randomness_evidence, Mapping):
            raise PaperContractError("assessment randomness evidence must be a mapping")
        if self.raw_randomness_evidence is not None and not isinstance(
            self.raw_randomness_evidence, Mapping
        ):
            raise PaperContractError("raw assessment randomness evidence must be a mapping")


def _nonempty(value: str, field: str) -> str:
    resolved = str(value)
    if not resolved.strip():
        raise PaperContractError(f"{field} must be nonempty")
    return resolved


def _seed(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PaperContractError(f"{field} must be a nonnegative integer")
    return value


def _finite_vector(values: Sequence[float], field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise PaperContractError(f"{field} must be a numerical vector")
    try:
        resolved = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise PaperContractError(f"{field} must be a numerical vector") from exc
    if not resolved or not all(math.isfinite(item) for item in resolved):
        raise PaperContractError(f"{field} must contain finite values")
    return resolved


ZERO_CALLS = PhysicalCallCounts(0, 0)


@dataclass(frozen=True)
class PaperComparisonConfigV3:
    """Method-independent prospective comparison configuration."""

    simulator_definition_identity: str
    simulator_configuration_identity: str
    assessment_fidelity: str
    assessment_campaign_seed: int
    inference_campaign_seed: int
    protocol_id: str = PROTOCOL_ID
    record_schema: str = RECORD_SCHEMA_V3
    assessment_policy: str = ASSESSMENT_POLICY_ID
    metric_schema: str = METRIC_SCHEMA_ID

    def __post_init__(self) -> None:
        for field in (
            "simulator_definition_identity",
            "simulator_configuration_identity",
            "assessment_fidelity",
        ):
            _nonempty(getattr(self, field), field)
        _seed(self.assessment_campaign_seed, "assessment_campaign_seed")
        _seed(self.inference_campaign_seed, "inference_campaign_seed")
        if self.protocol_id != PROTOCOL_ID:
            raise PaperContractError("unsupported paper comparison protocol")
        if self.record_schema != RECORD_SCHEMA_V3:
            raise PaperContractError("unsupported prospective record schema")
        if self.assessment_policy != ASSESSMENT_POLICY_ID:
            raise PaperContractError("unsupported paired assessment policy")
        if self.metric_schema != METRIC_SCHEMA_ID:
            raise PaperContractError("unsupported v3 metric schema")

    def document(self) -> dict[str, Any]:
        content = {
            "protocol_id": self.protocol_id,
            "record_schema": self.record_schema,
            "assessment_policy": self.assessment_policy,
            "metric_schema": self.metric_schema,
            "simulator_definition_identity": self.simulator_definition_identity,
            "simulator_configuration_identity": self.simulator_configuration_identity,
            "assessment_fidelity": self.assessment_fidelity,
            "assessment_campaign_seed": self.assessment_campaign_seed,
            "inference_campaign_seed": self.inference_campaign_seed,
        }
        return {
            **content,
            "configuration_identity": stable_hash(
                "paper-comparison-config-v1", content
            ),
        }

    @property
    def identity(self) -> str:
        return str(self.document()["configuration_identity"])


@dataclass(frozen=True)
class DecisionOutcome:
    """One method result before any held-out assessment is constructed."""

    method: str
    status: str
    decision_policy_identifier: str
    method_configuration_identity: str
    inference_budget: int
    inference_calls: PhysicalCallCounts
    recommended_theta: tuple[float, ...] | None = None
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_explanation: str | None = None
    decision_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_method_ids((self.method,))
        _nonempty(self.decision_policy_identifier, "decision_policy_identifier")
        _nonempty(self.method_configuration_identity, "method_configuration_identity")
        if isinstance(self.inference_budget, bool) or not isinstance(
            self.inference_budget, int
        ):
            raise PaperContractError("inference_budget must be a nonnegative integer")
        if self.inference_budget < 0:
            raise PaperContractError("inference_budget must be a nonnegative integer")
        if self.status not in {"recommended", "failed"}:
            raise PaperContractError("decision status must be 'recommended' or 'failed'")
        if self.status == "recommended":
            if self.recommended_theta is None:
                raise PaperContractError("recommended decisions require theta")
            object.__setattr__(
                self,
                "recommended_theta",
                _finite_vector(self.recommended_theta, "recommended_theta"),
            )
        elif self.recommended_theta is not None:
            raise PaperContractError("failed decisions cannot contain a recommendation")
        if self.status == "failed" and self.failure_stage is not None:
            _nonempty(self.failure_stage, "failure_stage")
        if self.status == "recommended" and self.failure_stage is not None:
            raise PaperContractError("recommended decisions cannot contain a failure stage")
        if self.decision_metadata is not None:
            if not isinstance(self.decision_metadata, Mapping):
                raise PaperContractError("decision_metadata must be a mapping")
            object.__setattr__(
                self,
                "decision_metadata",
                deepcopy(dict(self.decision_metadata)),
            )


@dataclass(frozen=True)
class FrozenDecisionBatch:
    """All target-method decisions after one successful persistence barrier."""

    target_record_identity: str
    decisions: tuple[DecisionOutcome, ...]
    persisted_document_json: str
    freeze_identity: str

    def document(self) -> dict[str, Any]:
        value = json.loads(self.persisted_document_json)
        assert isinstance(value, dict)
        return value


def paired_assessment_context(
    config: PaperComparisonConfigV3,
    target_record_identity: str,
    *,
    replicate: int = 0,
) -> dict[str, Any]:
    """Create the method-independent assessment family and coordinate."""

    target_id = _nonempty(target_record_identity, "target_record_identity")
    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or replicate < 0
    ):
        raise PaperContractError("assessment replicate must be a nonnegative integer")
    family_definition = {
        "protocol_id": config.protocol_id,
        "target_record_identity": target_id,
        "simulator_definition_identity": config.simulator_definition_identity,
        "simulator_configuration_identity": config.simulator_configuration_identity,
        "assessment_fidelity": config.assessment_fidelity,
        "assessment_campaign_seed": config.assessment_campaign_seed,
    }
    family = ScientificIdentity.create(family_definition)
    coordinate = SeedCoordinate(
        "paired_held_out_assessment", (target_id, replicate)
    )
    return {
        "policy_identifier": config.assessment_policy,
        "family_identity": family.value,
        "family_definition": family_definition,
        "coordinate": {
            "namespace": coordinate.namespace,
            "coordinates": list(coordinate.coordinates),
        },
        "target_record_identity": target_id,
        "replicate": replicate,
        "evaluation_seed": SeedDeriver(
            family, config.assessment_campaign_seed
        ).seed_for(coordinate),
    }


def method_inference_context(
    config: PaperComparisonConfigV3,
    target_record_identity: str,
    method: str,
    method_configuration_identity: str,
) -> dict[str, Any]:
    """Create a method-addressed inference family without moving old streams."""

    target_id = _nonempty(target_record_identity, "target_record_identity")
    validate_method_ids((method,))
    method_config = _nonempty(
        method_configuration_identity, "method_configuration_identity"
    )
    family_definition = {
        "protocol_id": config.protocol_id,
        "target_record_identity": target_id,
        "method": method,
        "method_configuration_identity": method_config,
        "simulator_definition_identity": config.simulator_definition_identity,
        "simulator_configuration_identity": config.simulator_configuration_identity,
        "inference_campaign_seed": config.inference_campaign_seed,
    }
    family = ScientificIdentity.create(family_definition)
    coordinate = SeedCoordinate("method_inference", (target_id, method))
    return {
        "family_identity": family.value,
        "family_definition": family_definition,
        "coordinate": {
            "namespace": coordinate.namespace,
            "coordinates": list(coordinate.coordinates),
        },
        "root_seed": config.inference_campaign_seed,
        "evaluation_seed": SeedDeriver(
            family, config.inference_campaign_seed
        ).seed_for(coordinate),
    }


def _decision_document(
    config: PaperComparisonConfigV3,
    target_record_identity: str,
    outcome: DecisionOutcome,
) -> dict[str, Any]:
    return {
        "method": outcome.method,
        "status": outcome.status,
        "decision_policy_identifier": outcome.decision_policy_identifier,
        "method_configuration_identity": outcome.method_configuration_identity,
        "recommended_theta": (
            list(outcome.recommended_theta)
            if outcome.recommended_theta is not None
            else None
        ),
        "failure_type": outcome.failure_type,
        "failure_explanation": outcome.failure_explanation,
        "failure_stage": outcome.failure_stage,
        "decision_metadata": deepcopy(outcome.decision_metadata),
        "inference_budget": outcome.inference_budget,
        "inference_calls": outcome.inference_calls.document(),
        "inference_context": method_inference_context(
            config,
            target_record_identity,
            outcome.method,
            outcome.method_configuration_identity,
        ),
        "recommendation_simulator_calls": 0,
    }


def freeze_decisions(
    config: PaperComparisonConfigV3,
    target_record_identity: str,
    decisions: Iterable[DecisionOutcome],
    *,
    persist: Callable[[Mapping[str, Any]], None],
) -> FrozenDecisionBatch:
    """Persist every recommendation/failure before assessment can be created."""

    target_id = _nonempty(target_record_identity, "target_record_identity")
    materialized = tuple(decisions)
    if not materialized:
        raise PaperContractError("at least one method decision is required")
    methods = tuple(item.method for item in materialized)
    validate_method_ids(methods)
    document = {
        "schema_version": FREEZE_SCHEMA,
        "protocol_id": config.protocol_id,
        "record_schema": config.record_schema,
        "configuration_identity": config.identity,
        "target_record_identity": target_id,
        "status": "frozen_before_assessment",
        "decisions": [
            _decision_document(config, target_id, outcome)
            for outcome in materialized
        ],
    }
    freeze_identity = stable_hash("paper-decision-freeze-v1", document)
    persisted = {**document, "freeze_identity": freeze_identity}
    persist(persisted)
    return FrozenDecisionBatch(
        target_id,
        materialized,
        canonical_json_bytes(persisted).decode("utf-8"),
        freeze_identity,
    )


def physical_metrics(
    assessment_response: Sequence[float],
    target_response: Sequence[float],
    recommended_theta: Sequence[float],
    theta_star: Sequence[float] | None,
) -> dict[str, float | None]:
    """Compute all v3 metrics from one physical assessment response."""

    actual = _finite_vector(assessment_response, "assessment_response")
    target = _finite_vector(target_response, "target_response")
    theta = _finite_vector(recommended_theta, "recommended_theta")
    if len(actual) != len(target):
        raise PaperContractError("assessment and target response dimensions differ")
    residual = tuple(left - right for left, right in zip(actual, target))
    l1_sum = sum(abs(item) for item in residual)
    parameter_l2: float | None = None
    if theta_star is not None:
        truth = _finite_vector(theta_star, "theta_star")
        if len(theta) != len(truth):
            raise PaperContractError("recommendation and hidden-theta dimensions differ")
        parameter_l2 = math.sqrt(
            sum((left - right) ** 2 for left, right in zip(theta, truth))
        )
    return {
        "response_error_l1_sum": l1_sum,
        "response_error_mae": l1_sum / len(residual),
        "response_error_l2": math.sqrt(sum(item**2 for item in residual)),
        "parameter_error_l2": parameter_l2,
    }


def _ledger_entry(
    *,
    category: str,
    nominal_calls: int,
    treatment: str,
    counts: PhysicalCallCounts,
    scope: str,
    identity_fields: Mapping[str, Any],
    disposition: str | None = None,
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "category": category,
        "scope": scope,
        "nominal_calls": nominal_calls,
        "budget_treatment": treatment,
        **counts.document(),
    }
    if disposition is not None:
        content["disposition"] = disposition
    return {
        **content,
        "ledger_identity": stable_hash(
            "paper-call-ledger-entry-v1", {**identity_fields, **content}
        ),
    }


def assess_frozen_decisions(
    config: PaperComparisonConfigV3,
    batch: FrozenDecisionBatch,
    *,
    target_response: Sequence[float],
    theta_star: Sequence[float] | None,
    target_construction_counts: PhysicalCallCounts,
    target_disposition: str,
    evaluate: Callable[
        [tuple[float, ...], Mapping[str, Any]],
        Sequence[float] | AssessmentEvaluationV3,
    ],
    replicate: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Assess a persisted decision batch once per valid recommendation."""

    if target_disposition not in {"constructed", "reused"}:
        raise PaperContractError("target disposition must be constructed or reused")
    if target_disposition == "reused" and target_construction_counts.attempted != 0:
        raise PaperContractError("reused targets cannot report new construction calls")
    target = _finite_vector(target_response, "target_response")
    truth = _finite_vector(theta_star, "theta_star") if theta_star is not None else None
    context = paired_assessment_context(
        config, batch.target_record_identity, replicate=replicate
    )
    frozen_by_method = {
        str(item["method"]): item for item in batch.document()["decisions"]
    }
    records = []
    for outcome in batch.decisions:
        decision = frozen_by_method[outcome.method]
        common = {
            "schema_version": config.record_schema,
            "software_schema": config.record_schema,
            "paper_comparison_protocol": config.protocol_id,
            "paper_comparison_configuration_identity": config.identity,
            "assessment_policy": config.assessment_policy,
            "metric_schema": config.metric_schema,
            "simulator_definition_identity": config.simulator_definition_identity,
            "simulator_configuration_identity": config.simulator_configuration_identity,
            "assessment_fidelity": config.assessment_fidelity,
            "target_record_identity": batch.target_record_identity,
            "method": outcome.method,
            "method_configuration_identity": outcome.method_configuration_identity,
            "decision_freeze_identity": batch.freeze_identity,
            "decision": decision,
            "assessment_context": context,
        }
        target_ledger = _ledger_entry(
            category="target_construction_or_reuse",
            nominal_calls=target_construction_counts.successful,
            treatment="outside_method_inference_budget_shared_target",
            counts=target_construction_counts,
            scope="shared_target_record",
            identity_fields={
                "protocol_id": config.protocol_id,
                "target_record_identity": batch.target_record_identity,
            },
            disposition=target_disposition,
        )
        inference_ledger = _ledger_entry(
            category="method_inference_or_training",
            nominal_calls=outcome.inference_budget,
            treatment="inside_method_inference_budget",
            counts=outcome.inference_calls,
            scope="target_method_record",
            identity_fields={
                "protocol_id": config.protocol_id,
                "target_record_identity": batch.target_record_identity,
                "method": outcome.method,
            },
        )
        recommendation_ledger = _ledger_entry(
            category="terminal_recommendation",
            nominal_calls=0,
            treatment="zero_call_model_side_decision",
            counts=ZERO_CALLS,
            scope="target_method_record",
            identity_fields={
                "protocol_id": config.protocol_id,
                "target_record_identity": batch.target_record_identity,
                "method": outcome.method,
                "decision_freeze_identity": batch.freeze_identity,
            },
        )
        if outcome.status == "failed":
            record = {
                **common,
                "status": "failed",
                "failure_stage": outcome.failure_stage or "final_decision",
                "failure_type": outcome.failure_type,
                "failure_explanation": outcome.failure_explanation,
                "recommended_theta": None,
                "assessment_response": None,
                **{field: None for field in V3_METRIC_FIELDS},
                "call_ledger": {
                    "target_construction_or_reuse": target_ledger,
                    "method_inference_or_training": inference_ledger,
                    "terminal_recommendation": recommendation_ledger,
                    "paired_assessment_and_metric": _ledger_entry(
                        category="paired_assessment_and_metric",
                        nominal_calls=0,
                        treatment="outside_method_inference_budget_not_run",
                        counts=ZERO_CALLS,
                        scope="target_method_record",
                        identity_fields={
                            "protocol_id": config.protocol_id,
                            "target_record_identity": batch.target_record_identity,
                            "method": outcome.method,
                            "assessment_family_identity": context["family_identity"],
                        },
                    ),
                },
            }
        else:
            assert outcome.recommended_theta is not None
            try:
                evaluated = evaluate(outcome.recommended_theta, context)
                randomness_evidence = None
                if isinstance(evaluated, AssessmentEvaluationV3):
                    assessed = evaluated.response
                    randomness_evidence = deepcopy(dict(evaluated.randomness_evidence))
                    raw_randomness_evidence = (
                        None
                        if evaluated.raw_randomness_evidence is None
                        else deepcopy(dict(evaluated.raw_randomness_evidence))
                    )
                else:
                    assessed = _finite_vector(evaluated, "assessment_response")
                    raw_randomness_evidence = None
            except Exception as exc:  # typed retention; unrelated methods continue
                record = {
                    **common,
                    "status": "failed",
                    "failure_stage": "paired_held_out_assessment",
                    "failure_type": type(exc).__name__,
                    "failure_explanation": str(exc),
                    "recommended_theta": list(outcome.recommended_theta),
                    "assessment_response": None,
                    **{field: None for field in V3_METRIC_FIELDS},
                    "call_ledger": {
                        "target_construction_or_reuse": target_ledger,
                        "method_inference_or_training": inference_ledger,
                        "terminal_recommendation": recommendation_ledger,
                        "paired_assessment_and_metric": _ledger_entry(
                            category="paired_assessment_and_metric",
                            nominal_calls=1,
                            treatment="outside_method_inference_budget",
                            counts=PhysicalCallCounts(1, 0, 1),
                            scope="target_method_record",
                            identity_fields={
                                "protocol_id": config.protocol_id,
                                "target_record_identity": batch.target_record_identity,
                                "method": outcome.method,
                                "assessment_family_identity": context["family_identity"],
                            },
                        ),
                    },
                }
            else:
                metrics = physical_metrics(
                    assessed, target, outcome.recommended_theta, truth
                )
                record = {
                    **common,
                    "status": "completed",
                    "failure_stage": None,
                    "failure_type": None,
                    "failure_explanation": None,
                    "recommended_theta": list(outcome.recommended_theta),
                    "assessment_response": list(assessed),
                    "assessment_randomness_evidence": randomness_evidence,
                    **(
                        {"_raw_assessment_randomness_evidence": raw_randomness_evidence}
                        if raw_randomness_evidence is not None
                        else {}
                    ),
                    **metrics,
                    "call_ledger": {
                        "target_construction_or_reuse": target_ledger,
                        "method_inference_or_training": inference_ledger,
                        "terminal_recommendation": recommendation_ledger,
                        "paired_assessment_and_metric": _ledger_entry(
                            category="paired_assessment_and_metric",
                            nominal_calls=1,
                            treatment="outside_method_inference_budget",
                            counts=PhysicalCallCounts(1, 1),
                            scope="target_method_record",
                            identity_fields={
                                "protocol_id": config.protocol_id,
                                "target_record_identity": batch.target_record_identity,
                                "method": outcome.method,
                                "assessment_family_identity": context["family_identity"],
                            },
                        ),
                    },
                }
        validate_v3_record(record)
        records.append(record)
    return tuple(records)


def _validate_posterior_sample_map_decision(decision: Mapping[str, Any]) -> None:
    metadata = decision.get("decision_metadata")
    if not isinstance(metadata, Mapping):
        raise PaperContractError("sample-MAP decision requires decision metadata")
    required = {
        "policy_identifier",
        "method_family",
        "decision_label",
        "candidate_count",
        "selected_candidate_index",
        "selected_theta",
        "selected_score",
        "score_identity",
        "score_api",
        "coordinate_system",
        "density_measure",
        "physical_bounds",
        "support_mask_policy",
        "tie_rule",
        "transform_jacobian_policy",
        "refinement",
        "recommendation_simulator_calls",
    }
    if set(metadata) != required:
        raise PaperContractError("sample-MAP decision metadata fields are malformed")
    method = decision.get("method")
    sample_map_methods = {
        "sbi_npe",
        "sbi_snpe",
        "sbi_nle_single_round",
        "sbi_nre_single_round",
    }
    if (
        method not in sample_map_methods
        or metadata.get("method_family") != method
    ):
        raise PaperContractError("sample-MAP method family is not approved")
    score_api = (
        "sbi_direct_posterior_log_prob_physical_v1"
        if method in {"sbi_npe", "sbi_snpe"}
        else "sbi_mcmc_posterior_potential_physical_no_transform_jacobian_v1"
    )
    fixed = {
        "policy_identifier": "posterior_sample_map_physical_v1",
        "decision_label": "sample_map",
        "coordinate_system": "physical_theta",
        "density_measure": "physical_theta_lebesgue",
        "support_mask_policy": "finite_and_closed_physical_box_v1",
        "tie_rule": "earliest_exact_score_tie",
        "transform_jacobian_policy": "none",
        "refinement": "none",
        "recommendation_simulator_calls": 0,
    }
    if metadata.get("score_api") != score_api:
        raise SbiZooScoringContractError("sample-MAP method/score API mismatch")
    if any(metadata.get(key) != value for key, value in fixed.items()):
        raise PaperContractError("sample-MAP decision metadata policy mismatch")
    score_identity = str(metadata.get("score_identity", ""))
    if not score_identity.strip():
        raise SbiZooScoringContractError("sample-MAP score identity is empty")
    zoo_score_identities = {
        "sbi_nle_single_round": "sbi-0.26.1-likelihood_plus_prior-physical-theta-v1",
        "sbi_nre_single_round": "sbi-0.26.1-ratio_plus_prior-physical-theta-v1",
    }
    if method in zoo_score_identities and score_identity != zoo_score_identities[method]:
        raise SbiZooScoringContractError("SBI-zoo sample-MAP score identity mismatch")
    candidate_count = metadata.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 1
    ):
        raise PaperContractError("sample-MAP candidate count must be positive")
    if method in {"sbi_nle_single_round", "sbi_nre_single_round"} and candidate_count != 10_000:
        raise PaperContractError("SBI-zoo sample-MAP requires exactly 10,000 candidates")
    bounds = metadata.get("physical_bounds")
    if not isinstance(bounds, Mapping) or set(bounds) != {"lower", "upper"}:
        raise PaperContractError("sample-MAP physical bounds are malformed")
    lower = _finite_vector(bounds["lower"], "sample_map_lower")
    upper = _finite_vector(bounds["upper"], "sample_map_upper")
    if len(lower) != len(upper) or any(lo >= hi for lo, hi in zip(lower, upper)):
        raise PaperContractError("sample-MAP physical bounds are invalid")
    decision_status = decision.get("status")
    if decision_status not in {"recommended", "failed"}:
        raise PaperContractError("sample-MAP decision status is invalid")
    if decision_status == "recommended":
        selected_index = metadata.get("selected_candidate_index")
        if (
            isinstance(selected_index, bool)
            or not isinstance(selected_index, int)
            or not 0 <= selected_index < candidate_count
        ):
            raise PaperContractError("sample-MAP selected candidate index is invalid")
        theta = _finite_vector(metadata.get("selected_theta"), "sample_map_theta")
        recommended = _finite_vector(
            decision.get("recommended_theta"), "recommended_theta"
        )
        if theta != recommended:
            raise PaperContractError("sample-MAP metadata theta disagrees with decision")
        if len(theta) != len(lower) or any(
            value < lo or value > hi
            for value, lo, hi in zip(theta, lower, upper)
        ):
            raise PaperContractError("sample-MAP selected theta is outside support")
        score = metadata.get("selected_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise PaperContractError("sample-MAP selected score must be finite")
    elif any(
        metadata.get(field) is not None
        for field in ("selected_candidate_index", "selected_theta", "selected_score")
    ):
        raise PaperContractError("failed sample-MAP decision cannot contain a selection")


def _validate_assessment_randomness_evidence(
    record: Mapping[str, Any], evidence: object
) -> None:
    if not isinstance(evidence, Mapping):
        raise PaperContractError("assessment randomness evidence is malformed")
    context = record["assessment_context"]
    if evidence.get("root_evaluation_seed") != context["evaluation_seed"]:
        raise PaperContractError("assessment randomness root differs from paired context")
    identities = evidence.get("coordinate_family_identities")
    hashes = evidence.get("namespace_seed_manifest_hashes")
    if not all(isinstance(value, Mapping) for value in (identities, hashes)):
        raise PaperContractError("assessment randomness namespace evidence is malformed")
    required = {"quenched_disorder", "initial_state", "thermal_noise"}
    if set(identities) != required or set(hashes) != required:
        raise PaperContractError("graph assessment requires disorder, initial, and thermal namespaces")
    for namespace in sorted(required):
        expected_identity = stable_hash(
            "paper-assessment-randomness-coordinate-family-v1",
            {
                "assessment_family_identity": context["family_identity"],
                "root_evaluation_seed": context["evaluation_seed"],
                "namespace": namespace,
            },
        )
        if identities.get(namespace) != expected_identity:
            raise PaperContractError(f"assessment coordinate family mismatch: {namespace}")
        value = hashes.get(namespace)
        if not isinstance(value, str) or not value.strip():
            raise PaperContractError(f"assessment seed manifest hash is missing: {namespace}")
    full_hash = evidence.get("full_seed_manifest_hash")
    if not isinstance(full_hash, str) or not full_hash.strip():
        raise PaperContractError("full assessment seed manifest hash is missing")


def validate_v3_record(record: Mapping[str, Any]) -> None:
    """Strictly validate one prospective record without reinterpreting v2."""

    if record.get("schema_version") != RECORD_SCHEMA_V3:
        raise PaperContractError("v3 validator received another record schema")
    if record.get("software_schema") != RECORD_SCHEMA_V3:
        raise PaperContractError("v3 software schema mismatch")
    if record.get("paper_comparison_protocol") != PROTOCOL_ID:
        raise PaperContractError("v3 paper comparison protocol mismatch")
    if record.get("assessment_policy") != ASSESSMENT_POLICY_ID:
        raise PaperContractError("v3 assessment policy mismatch")
    if record.get("metric_schema") != METRIC_SCHEMA_ID:
        raise PaperContractError("v3 metric schema mismatch")
    leaked = [field for field in LEGACY_METRIC_FIELDS if field in record]
    if leaked:
        raise PaperContractError("v3 record contains ambiguous legacy metric fields")
    for field in (
        "paper_comparison_configuration_identity",
        "simulator_definition_identity",
        "simulator_configuration_identity",
        "assessment_fidelity",
        "target_record_identity",
        "method",
        "method_configuration_identity",
        "decision_freeze_identity",
    ):
        _nonempty(str(record.get(field, "")), field)
    validate_method_ids((str(record["method"]),))
    decision = record.get("decision")
    if not isinstance(decision, Mapping):
        raise PaperContractError("v3 record requires a frozen decision")
    if decision.get("method") != record.get("method"):
        raise PaperContractError("v3 decision method mismatch")
    if decision.get("recommendation_simulator_calls") != 0:
        raise PaperContractError("v3 recommendation must use zero simulator calls")
    if (
        decision.get("status") == "recommended"
        and decision.get("decision_policy_identifier")
        == "posterior_sample_map_physical_v1"
    ):
        _validate_posterior_sample_map_decision(decision)
    context = record.get("assessment_context")
    if not isinstance(context, Mapping):
        raise PaperContractError("v3 record requires an assessment context")
    if context.get("policy_identifier") != ASSESSMENT_POLICY_ID:
        raise PaperContractError("v3 assessment context policy mismatch")
    if context.get("target_record_identity") != record.get("target_record_identity"):
        raise PaperContractError("v3 assessment target mismatch")
    coordinate = context.get("coordinate")
    if (
        not isinstance(coordinate, Mapping)
        or coordinate.get("namespace") != "paired_held_out_assessment"
    ):
        raise PaperContractError("v3 assessment coordinate mismatch")
    replicate = context.get("replicate")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise PaperContractError("v3 assessment replicate is invalid")
    expected_coordinate = [record["target_record_identity"], replicate]
    if coordinate.get("coordinates") != expected_coordinate:
        raise PaperContractError("v3 assessment coordinate is not target addressed")
    family_definition = context.get("family_definition")
    expected_family_fields = {
        "protocol_id": PROTOCOL_ID,
        "target_record_identity": record["target_record_identity"],
        "simulator_definition_identity": record["simulator_definition_identity"],
        "simulator_configuration_identity": record[
            "simulator_configuration_identity"
        ],
        "assessment_fidelity": record["assessment_fidelity"],
    }
    if not isinstance(family_definition, Mapping):
        raise PaperContractError("v3 assessment family definition is missing")
    if set(family_definition) != {
        *expected_family_fields,
        "assessment_campaign_seed",
    }:
        raise PaperContractError("v3 assessment family contains forbidden fields")
    if any(family_definition.get(key) != value for key, value in expected_family_fields.items()):
        raise PaperContractError("v3 assessment family definition mismatch")
    assessment_seed = family_definition.get("assessment_campaign_seed")
    _seed(assessment_seed, "assessment_campaign_seed")
    family = ScientificIdentity.create(family_definition)
    if context.get("family_identity") != family.value:
        raise PaperContractError("v3 assessment family identity mismatch")
    expected_assessment_seed = SeedDeriver(family, assessment_seed).seed_for(
        SeedCoordinate("paired_held_out_assessment", tuple(expected_coordinate))
    )
    if context.get("evaluation_seed") != expected_assessment_seed:
        raise PaperContractError("v3 paired assessment seed mismatch")
    inference = decision.get("inference_context")
    if not isinstance(inference, Mapping):
        raise PaperContractError("v3 decision requires an inference context")
    inference_family_definition = inference.get("family_definition")
    expected_inference_fields = {
        "protocol_id": PROTOCOL_ID,
        "target_record_identity": record["target_record_identity"],
        "method": record["method"],
        "method_configuration_identity": record["method_configuration_identity"],
        "simulator_definition_identity": record["simulator_definition_identity"],
        "simulator_configuration_identity": record[
            "simulator_configuration_identity"
        ],
    }
    if not isinstance(inference_family_definition, Mapping):
        raise PaperContractError("v3 inference family definition is missing")
    if set(inference_family_definition) != {
        *expected_inference_fields,
        "inference_campaign_seed",
    }:
        raise PaperContractError("v3 inference family contains forbidden fields")
    if any(
        inference_family_definition.get(key) != value
        for key, value in expected_inference_fields.items()
    ):
        raise PaperContractError("v3 inference family definition mismatch")
    inference_seed = inference_family_definition.get("inference_campaign_seed")
    _seed(inference_seed, "inference_campaign_seed")
    inference_family = ScientificIdentity.create(inference_family_definition)
    if inference.get("family_identity") != inference_family.value:
        raise PaperContractError("v3 inference family identity mismatch")
    inference_coordinate = inference.get("coordinate")
    expected_inference_coordinate = [record["target_record_identity"], record["method"]]
    if not isinstance(inference_coordinate, Mapping) or inference_coordinate != {
        "namespace": "method_inference",
        "coordinates": expected_inference_coordinate,
    }:
        raise PaperContractError("v3 inference coordinate is not method addressed")
    if inference.get("root_seed") != inference_seed:
        raise PaperContractError("v3 inference root seed mismatch")
    expected_inference_seed = SeedDeriver(
        inference_family, inference_seed
    ).seed_for(
        SeedCoordinate("method_inference", tuple(expected_inference_coordinate))
    )
    if inference.get("evaluation_seed") != expected_inference_seed:
        raise PaperContractError("v3 inference seed mismatch")
    ledger = record.get("call_ledger")
    if not isinstance(ledger, Mapping) or set(ledger) != set(LEDGER_CATEGORIES):
        raise PaperContractError("v3 record requires the exact four-category call ledger")
    for category in LEDGER_CATEGORIES:
        entry = ledger[category]
        if not isinstance(entry, Mapping):
            raise PaperContractError(f"invalid v3 ledger category: {category}")
        if entry.get("category") != category:
            raise PaperContractError(f"v3 ledger category label mismatch: {category}")
        expected_scope = (
            "shared_target_record"
            if category == "target_construction_or_reuse"
            else "target_method_record"
        )
        if entry.get("scope") != expected_scope:
            raise PaperContractError(f"v3 ledger scope mismatch: {category}")
        _nonempty(str(entry.get("ledger_identity", "")), "ledger_identity")
        raw_counts = tuple(
            entry.get(field)
            for field in (
                "attempted_calls",
                "successful_calls",
                "failed_calls",
                "retry_calls",
            )
        )
        if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_counts):
            raise PaperContractError(f"invalid physical call counts for {category}")
        counts = PhysicalCallCounts(*raw_counts)
        nominal = entry.get("nominal_calls")
        if isinstance(nominal, bool) or not isinstance(nominal, int) or nominal < 0:
            raise PaperContractError(f"invalid nominal calls for {category}")
        if not str(entry.get("budget_treatment", "")).strip():
            raise PaperContractError(f"missing budget treatment for {category}")
        if category == "terminal_recommendation" and counts.attempted != 0:
            raise PaperContractError("v3 terminal recommendation must be zero-call")
        identity_fields: dict[str, Any] = {
            "protocol_id": PROTOCOL_ID,
            "target_record_identity": record["target_record_identity"],
        }
        if category != "target_construction_or_reuse":
            identity_fields["method"] = record["method"]
        if category == "terminal_recommendation":
            identity_fields["decision_freeze_identity"] = record[
                "decision_freeze_identity"
            ]
        if category == "paired_assessment_and_metric":
            identity_fields["assessment_family_identity"] = context[
                "family_identity"
            ]
        content = {key: value for key, value in entry.items() if key != "ledger_identity"}
        expected_ledger_identity = stable_hash(
            "paper-call-ledger-entry-v1", {**identity_fields, **content}
        )
        if entry.get("ledger_identity") != expected_ledger_identity:
            raise PaperContractError(f"v3 ledger identity mismatch: {category}")
    status = record.get("status")
    if status not in {"completed", "failed"}:
        raise PaperContractError("v3 record status must be completed or failed")
    assessment_counts = ledger["paired_assessment_and_metric"]
    if status == "completed":
        if assessment_counts.get("successful_calls") != 1:
            raise PaperContractError("completed v3 records require one assessment call")
        if any(record.get(field) is None for field in V3_METRIC_FIELDS[:3]):
            raise PaperContractError("completed v3 records require response metrics")
        l1_sum = float(record["response_error_l1_sum"])
        mae = float(record["response_error_mae"])
        response = record.get("assessment_response")
        if not isinstance(response, list) or not response:
            raise PaperContractError("completed v3 records require assessment response")
        if not math.isclose(mae * len(response), l1_sum, rel_tol=1e-12, abs_tol=1e-12):
            raise PaperContractError("v3 MAE and L1-sum fields disagree")
        randomness = record.get("assessment_randomness_evidence")
        if randomness is not None:
            _validate_assessment_randomness_evidence(record, randomness)
    elif any(record.get(field) is not None for field in V3_METRIC_FIELDS):
        raise PaperContractError("failed v3 records cannot contain performance metrics")


def validate_paper_record_schema(record: Mapping[str, Any]) -> None:
    """Dispatch persisted records without changing historical v2 semantics."""

    schema = record.get("schema_version")
    if schema == RECORD_SCHEMA_V2:
        return
    if schema == RECORD_SCHEMA_V3:
        validate_v3_record(record)
        return
    raise PaperContractError(f"unsupported paper target-method record schema: {schema!r}")
