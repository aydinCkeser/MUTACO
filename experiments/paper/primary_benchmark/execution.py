"""Capped freeze-before-assessment execution for one primary target-budget pair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver, stable_hash
from experiments.paper.protocol.validation import PaperContractError
from .design import PrimarySobolDesign
from .protocol import (
    PrimaryDecisionOutcome,
    PrimaryMethodDeclaration,
)


def primary_method_context(
    declaration: Any,
    target_id: str,
    method: str,
) -> dict[str, Any]:
    declared_methods = tuple(item.method for item in declaration.methods)
    if method not in declared_methods:
        raise PaperContractError("unsupported primary method context")
    family = stable_hash(
        "primary-method-inference-family-v1",
        {
            "protocol_id": declaration.comparison.protocol_id,
            "campaign_id": declaration.campaign_id,
            "study_identity": declaration.study_identity,
            "target_id": target_id,
            "budget": declaration.budget,
            "method": method,
            "comparison_identity": declaration.comparison.identity,
        },
    )
    return {
        "family_identity": family,
        "root_seed": declaration.comparison.inference_campaign_seed,
        "coordinate": {
            "namespace": "method_inference",
            "coordinates": [
                declaration.campaign_id,
                target_id,
                declaration.budget,
                method,
            ],
        },
    }


def primary_assessment_context(
    declaration: Any,
    target_id: str,
    *,
    replicate: int = 0,
) -> dict[str, Any]:
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise PaperContractError("assessment replicate must be nonnegative")
    coordinate = SeedCoordinate(
        "heldout",
        (
            declaration.campaign_id,
            target_id,
            declaration.budget,
            replicate,
            "heldout",
        ),
    )
    family = stable_hash(
        "primary-paired-heldout-family-v1",
        {
            "protocol_id": declaration.comparison.protocol_id,
            "campaign_id": declaration.campaign_id,
            "study_identity": declaration.study_identity,
            "target_id": target_id,
            "budget": declaration.budget,
            "replicate": replicate,
            "comparison_identity": declaration.comparison.identity,
        },
    )
    deriver = SeedDeriver(
        ScientificIdentity(family), declaration.comparison.assessment_campaign_seed
    )
    return {
        "family_identity": family,
        "root_seed": declaration.comparison.assessment_campaign_seed,
        "purpose": "paired_held_out_assessment",
        "logical_index": replicate,
        "evaluation_seed": deriver.seed_for(coordinate),
        "coordinate": {
            "namespace": coordinate.namespace,
            "coordinates": list(coordinate.coordinates),
        },
        "replicate": replicate,
    }


@dataclass
class PrimaryCallCap:
    maximum: int
    attempted: int = 0
    successful: int = 0
    failed: int = 0
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.maximum, bool) or not isinstance(self.maximum, int) or self.maximum < 0:
            raise PaperContractError("primary call cap must be nonnegative")

    def invoke(self, category: str, operation: Callable[[], Any]) -> Any:
        if category not in {"method_search", "paired_heldout_assessment"}:
            raise PaperContractError("primary call category is invalid")
        if self.attempted >= self.maximum:
            raise PaperContractError("primary physical-call cap exhausted")
        counts = self.by_category.setdefault(
            category, {"attempted": 0, "successful": 0, "failed": 0}
        )
        self.attempted += 1
        counts["attempted"] += 1
        try:
            value = operation()
        except Exception:
            self.failed += 1
            counts["failed"] += 1
            raise
        self.successful += 1
        counts["successful"] += 1
        return value

    def document(self) -> dict[str, Any]:
        return {
            "maximum": self.maximum,
            "attempted": self.attempted,
            "successful": self.successful,
            "failed": self.failed,
            "by_category": {key: dict(value) for key, value in self.by_category.items()},
        }


@dataclass(frozen=True)
class PrimaryTargetExecution:
    target_id: str
    budget: int
    shared_design: Mapping[str, Any]
    decision_freeze: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    physical_calls: Mapping[str, Any]
    schema_version: str = "mutaco-primary-target-execution-v1"

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "budget": self.budget,
            "shared_design": dict(self.shared_design),
            "decision_freeze": dict(self.decision_freeze),
            "records": [dict(item) for item in self.records],
            "physical_calls": dict(self.physical_calls),
        }


RunMethod = Callable[
    [
        PrimaryMethodDeclaration,
        Mapping[str, Any],
        PrimarySobolDesign,
        Callable[[Sequence[float], int], Sequence[float]],
    ],
    PrimaryDecisionOutcome,
]


def execute_primary_target(
    *,
    declaration: Any,
    target_id: str,
    target_response: Sequence[float],
    design: PrimarySobolDesign,
    run_method: RunMethod,
    evaluate_search: Callable[
        [str, Sequence[float], Mapping[str, Any]], Sequence[float]
    ],
    persist_freeze: Callable[[Mapping[str, Any]], None],
    evaluate_assessment: Callable[
        [Sequence[float], Mapping[str, Any]], Sequence[float]
    ],
) -> PrimaryTargetExecution:
    if target_id not in declaration.target_record_identities:
        raise PaperContractError("primary target is absent from declaration")
    if design.target_id != target_id or design.budget != declaration.budget:
        raise PaperContractError("primary shared design differs from target declaration")
    if len(target_response) < 1:
        raise PaperContractError("primary target response must be nonempty")
    method_names = tuple(item.method for item in declaration.methods)
    if len(method_names) != 3 or len(set(method_names)) != 3:
        raise PaperContractError("primary execution requires exactly three methods")
    cap = PrimaryCallCap(len(method_names) * (declaration.budget + 1))
    outcomes: list[PrimaryDecisionOutcome] = []

    for method_declaration in declaration.methods:
        method = method_declaration.method
        context = primary_method_context(declaration, target_id, method)
        deriver = SeedDeriver(
            ScientificIdentity(str(context["family_identity"])),
            int(context["root_seed"]),
        )
        seen_indices: set[int] = set()
        evaluated_thetas: list[tuple[float, ...]] = []
        simulator_contexts: list[Mapping[str, Any]] = []
        before_attempted = cap.attempted
        before_successful = cap.successful

        def evaluate(theta: Sequence[float], evaluation_index: int) -> Sequence[float]:
            if (
                isinstance(evaluation_index, bool)
                or not isinstance(evaluation_index, int)
                or evaluation_index < 1
                or evaluation_index > declaration.budget
                or evaluation_index in seen_indices
            ):
                raise PaperContractError("primary evaluation index is invalid or duplicated")
            seen_indices.add(evaluation_index)
            coordinate = SeedCoordinate(
                "simulation",
                (
                    declaration.campaign_id,
                    target_id,
                    declaration.budget,
                    method,
                    evaluation_index,
                    "simulation",
                ),
            )
            simulator_context = {
                "family_identity": context["family_identity"],
                "purpose": "primary_search_simulation",
                "logical_index": evaluation_index - 1,
                "evaluation_index": evaluation_index,
                "evaluation_seed": deriver.seed_for(coordinate),
                "coordinate": {
                    "namespace": coordinate.namespace,
                    "coordinates": list(coordinate.coordinates),
                },
            }
            evaluated_thetas.append(tuple(float(value) for value in theta))
            simulator_contexts.append(simulator_context)
            return cap.invoke(
                "method_search",
                lambda: evaluate_search(method, theta, simulator_context),
            )

        outcome = run_method(method_declaration, context, design, evaluate)
        if not isinstance(outcome, PrimaryDecisionOutcome):
            raise PaperContractError("primary method callback returned the wrong type")
        if (
            outcome.method != method
            or outcome.budget != declaration.budget
            or outcome.configuration_identity
            != method_declaration.configuration_identity
            or outcome.search_policy_identifier
            != method_declaration.search_policy_identifier
            or outcome.terminal_policy_identifier
            != method_declaration.terminal_policy_identifier
        ):
            raise PaperContractError("primary method outcome differs from declaration")
        if outcome.attempted_search_calls != cap.attempted - before_attempted:
            raise PaperContractError("primary attempted-search ledger mismatch")
        if outcome.successful_search_calls != cap.successful - before_successful:
            raise PaperContractError("primary successful-search ledger mismatch")
        if tuple(item.theta for item in outcome.trajectory) != tuple(evaluated_thetas):
            raise PaperContractError("primary trajectory theta differs from evaluated theta")
        if tuple(dict(item.simulator_context) for item in outcome.trajectory) != tuple(
            dict(item) for item in simulator_contexts[: outcome.successful_search_calls]
        ):
            raise PaperContractError("primary trajectory simulator contexts differ from calls")
        expected_prefix = tuple(
            tuple(float(value) for value in row)
            for row in design.shared_prefix.detach().cpu().tolist()
        )
        observed_initial = tuple(item.theta for item in outcome.trajectory[:16])
        if observed_initial != expected_prefix[: len(observed_initial)]:
            raise PaperContractError("primary method did not use the shared initial geometry")
        if method == "sobol":
            expected_full = tuple(
                tuple(float(value) for value in row)
                for row in design.full.theta.detach().cpu().tolist()
            )
            if tuple(item.theta for item in outcome.trajectory) != expected_full:
                raise PaperContractError("Sobol trajectory differs from predetermined design")
        outcomes.append(outcome)

    freeze_schema = str(getattr(
        declaration,
        "decision_freeze_schema",
        "mutaco-primary-all-method-decision-freeze-v1",
    ))
    freeze_domain = str(getattr(
        declaration,
        "decision_freeze_identity_domain",
        "primary-all-method-decision-freeze-v1",
    ))
    freeze_body = {
        "schema_version": freeze_schema,
        "protocol_id": declaration.comparison.protocol_id,
        "campaign_id": declaration.campaign_id,
        "target_id": target_id,
        "budget": declaration.budget,
        "shared_prefix_identity": design.shared_prefix_identity,
        "decisions": [item.document() for item in outcomes],
        "assessment_started": False,
    }
    freeze = {
        **freeze_body,
        "freeze_identity": stable_hash(freeze_domain, freeze_body),
    }
    persist_freeze(freeze)

    assessment_context = primary_assessment_context(declaration, target_id)
    records: list[dict[str, Any]] = []
    for outcome in outcomes:
        record: dict[str, Any] = {
            "schema_version": declaration.comparison.record_schema,
            "protocol_id": declaration.comparison.protocol_id,
            "campaign_id": declaration.campaign_id,
            "study_identity": declaration.study_identity,
            "target_id": target_id,
            "budget": declaration.budget,
            "method": outcome.method,
            "shared_prefix_identity": design.shared_prefix_identity,
            "decision_freeze_identity": freeze["freeze_identity"],
            "decision": outcome.document(),
            "assessment_context": None,
            "heldout_response": None,
            "heldout_physical_mae": None,
            "status": "failed" if outcome.status == "failed" else "completed",
        }
        if outcome.status == "recommended":
            assert outcome.recommended_theta is not None
            heldout = cap.invoke(
                "paired_heldout_assessment",
                lambda: evaluate_assessment(
                    outcome.recommended_theta, assessment_context
                ),
            )
            values = tuple(float(value) for value in heldout)
            if len(values) != len(target_response):
                raise PaperContractError("primary held-out response dimension mismatch")
            mae = sum(
                abs(value - float(target))
                for value, target in zip(values, target_response)
            ) / len(values)
            record.update(
                {
                    "assessment_context": dict(assessment_context),
                    "heldout_response": list(values),
                    "heldout_physical_mae": mae,
                }
            )
        record["record_identity"] = stable_hash(
            str(getattr(
                declaration,
                "record_identity_domain",
                "primary-target-method-record-v1",
            )),
            record,
        )
        records.append(record)

    return PrimaryTargetExecution(
        target_id=target_id,
        budget=declaration.budget,
        shared_design=design.document(),
        decision_freeze=freeze,
        records=tuple(records),
        physical_calls=cap.document(),
        schema_version=str(getattr(
            declaration,
            "target_execution_schema",
            "mutaco-primary-target-execution-v1",
        )),
    )
