"""Append-only assessment rescue for an immutable primary decision freeze."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mutaco import stable_hash

from experiments.paper.protocol.validation import PaperContractError

from .execution import primary_assessment_context
from .protocol import PRIMARY_METHODS, PrimaryPopulationDeclaration
from .qualification_execution import append_hashed_json


RESCUE_SCHEMA = "mutaco-primary-assessment-rescue-declaration-v1"
RESCUE_RECORD_SCHEMA = "mutaco-primary-assessment-rescue-record-v1"


def load_primary_assessment_rescue_declaration(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(document)
    identity = value.pop("declaration_identity", None)
    if (
        value.get("schema_version") != RESCUE_SCHEMA
        or value.get("state") != "declared_unexecuted_submission_unauthorized"
        or value.get("classification")
        != "prequalification_engineering_rescue_not_q1_or_primary_performance_evidence"
        or value.get("maximum_assessment_calls") != 3
        or value.get("search_calls") != 0
        or value.get("automatic_retry") is not False
        or value.get("scientific_submission_authorized") is not False
        or value.get("primary_benchmark_evidence_eligible") is not False
    ):
        raise PaperContractError("primary assessment rescue declaration is invalid")
    methods = value.get("methods")
    if not isinstance(methods, list) or tuple(
        item.get("method") for item in methods if isinstance(item, Mapping)
    ) != PRIMARY_METHODS:
        raise PaperContractError("primary assessment rescue methods are invalid")
    if identity != stable_hash("primary-assessment-rescue-declaration-v1", value):
        raise PaperContractError("primary assessment rescue declaration identity mismatch")
    value["declaration_identity"] = identity
    return value


def validate_primary_assessment_rescue_freeze(
    *,
    rescue: Mapping[str, Any],
    primary: PrimaryPopulationDeclaration,
    freeze: Mapping[str, Any],
    freeze_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    primary_identity = primary.document()["declaration_identity"]
    if (
        rescue.get("primary_declaration_identity") != primary_identity
        or rescue.get("campaign_id") != primary.campaign_id
        or rescue.get("target_id") != primary.target_record_identities[0]
        or rescue.get("budget") != primary.budget
        or rescue.get("decision_freeze_sha256") != freeze_sha256
        or rescue.get("decision_freeze_identity") != freeze.get("freeze_identity")
    ):
        raise PaperContractError("primary assessment rescue source binding mismatch")
    if (
        freeze.get("schema_version") != "mutaco-primary-all-method-decision-freeze-v1"
        or freeze.get("assessment_started") is not False
        or freeze.get("campaign_id") != primary.campaign_id
        or freeze.get("target_id") != primary.target_record_identities[0]
        or freeze.get("budget") != primary.budget
    ):
        raise PaperContractError("primary assessment rescue freeze is ineligible")
    freeze_body = {key: value for key, value in freeze.items() if key != "freeze_identity"}
    if freeze.get("freeze_identity") != stable_hash(
        "primary-all-method-decision-freeze-v1", freeze_body
    ):
        raise PaperContractError("primary assessment rescue freeze identity mismatch")
    decisions = freeze.get("decisions")
    declared = rescue.get("methods")
    if not isinstance(decisions, list) or len(decisions) != len(PRIMARY_METHODS):
        raise PaperContractError("primary assessment rescue decision coverage is invalid")
    for decision, expected in zip(decisions, declared):
        if (
            not isinstance(decision, Mapping)
            or decision.get("method") != expected.get("method")
            or decision.get("status") != "recommended"
            or decision.get("budget") != primary.budget
            or decision.get("search_calls")
            != {"attempted": primary.budget, "successful": primary.budget}
            or decision.get("recommended_theta") != expected.get("recommended_theta")
        ):
            raise PaperContractError("primary assessment rescue decision differs from freeze")
    return tuple(decisions)


def primary_assessment_rescue_preview(
    *,
    rescue: Mapping[str, Any],
    primary: PrimaryPopulationDeclaration,
    freeze: Mapping[str, Any],
    freeze_sha256: str,
) -> dict[str, Any]:
    decisions = validate_primary_assessment_rescue_freeze(
        rescue=rescue,
        primary=primary,
        freeze=freeze,
        freeze_sha256=freeze_sha256,
    )
    context = primary_assessment_context(primary, primary.target_record_identities[0])
    return {
        "schema_version": "mutaco-primary-assessment-rescue-preview-v1",
        "declaration_identity": rescue["declaration_identity"],
        "primary_declaration_identity": primary.document()["declaration_identity"],
        "decision_freeze_identity": freeze["freeze_identity"],
        "target_id": primary.target_record_identities[0],
        "methods": [decision["method"] for decision in decisions],
        "search_calls": 0,
        "maximum_assessment_calls": 3,
        "assessment_context": context,
        "decision_freeze_validated": True,
        "scientific_calls_executed": 0,
        "job_submitted": False,
        "submission_authorized": False,
    }


def execute_primary_assessment_rescue(
    *,
    rescue: Mapping[str, Any],
    primary: PrimaryPopulationDeclaration,
    freeze: Mapping[str, Any],
    freeze_sha256: str,
    target_response: Sequence[float],
    evaluate_assessment: Callable[[Sequence[float], Mapping[str, Any]], Sequence[float]],
    output: Path,
    source_commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise PaperContractError("primary assessment rescue output must be append-only")
    decisions = validate_primary_assessment_rescue_freeze(
        rescue=rescue,
        primary=primary,
        freeze=freeze,
        freeze_sha256=freeze_sha256,
    )
    identity = str(rescue["declaration_identity"])
    context = primary_assessment_context(primary, primary.target_record_identities[0])
    append_hashed_json(output / "rescue_start.json", {
        "schema_version": "mutaco-primary-assessment-rescue-start-v1",
        "source_commit": source_commit,
        "declaration_identity": identity,
        "decision_freeze_identity": freeze["freeze_identity"],
        "maximum_assessment_calls": 3,
        "search_calls": 0,
    })
    attempted = successful = failed = 0
    records: list[dict[str, Any]] = []
    error: dict[str, str] | None = None
    try:
        for decision in decisions:
            if attempted >= 3:
                raise PaperContractError("primary assessment rescue call cap exhausted")
            attempted += 1
            try:
                heldout = evaluate_assessment(decision["recommended_theta"], context)
            except Exception:
                failed += 1
                raise
            successful += 1
            values = tuple(float(value) for value in heldout)
            if len(values) != len(target_response):
                raise PaperContractError("primary assessment rescue response dimension mismatch")
            body = {
                "schema_version": RESCUE_RECORD_SCHEMA,
                "declaration_identity": identity,
                "source_attempt_job_id": rescue["source_attempt_job_id"],
                "decision_freeze_identity": freeze["freeze_identity"],
                "primary_declaration_identity": primary.document()["declaration_identity"],
                "target_id": primary.target_record_identities[0],
                "budget": primary.budget,
                "method": decision["method"],
                "recommended_theta": list(decision["recommended_theta"]),
                "assessment_context": dict(context),
                "heldout_response": list(values),
                "heldout_physical_mae": sum(
                    abs(value - float(target))
                    for value, target in zip(values, target_response)
                ) / len(values),
            }
            record = {
                **body,
                "record_identity": stable_hash("primary-assessment-rescue-record-v1", body),
            }
            append_hashed_json(output / "records" / f"{decision['method']}.json", record)
            records.append(record)
    except Exception as exc:
        error = {"failure_type": type(exc).__name__, "explanation": str(exc)}
    calls = {
        "maximum_calls": 3,
        "attempted_calls": attempted,
        "successful_calls": successful,
        "failed_calls": failed,
    }
    status = "completed" if error is None and successful == 3 else "failed"
    result_body = {
        "schema_version": "mutaco-primary-assessment-rescue-result-v1",
        "source_commit": source_commit,
        "declaration_identity": identity,
        "decision_freeze_identity": freeze["freeze_identity"],
        "status": status,
        "records": records,
        "physical_calls": calls,
        "failure": error,
        "search_rerun": False,
    }
    result = {
        **result_body,
        "rescue_identity": stable_hash("primary-assessment-rescue-result-v1", result_body),
    }
    append_hashed_json(output / "rescue_result.json", result)
    append_hashed_json(output / "rescue_terminal.json", {
        "schema_version": "mutaco-primary-assessment-rescue-terminal-v1",
        "source_commit": source_commit,
        "declaration_identity": identity,
        "decision_freeze_identity": freeze["freeze_identity"],
        "status": status,
        "physical_calls": calls,
        "record_count": len(records),
        "failure": error,
    })
    if status != "completed":
        raise PaperContractError("primary assessment rescue failed; evidence was preserved")
    return result
