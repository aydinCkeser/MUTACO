"""Lean versioned schemas for the Paper Lab orchestration layer."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any, Mapping

from .identities import assert_document_identity


SCHEMAS = frozenset({
    "mutaco-paper-study-spec-v1",
    "mutaco-paper-execution-spec-v1",
    "mutaco-paper-ingestion-spec-v1",
    "mutaco-paper-analysis-spec-v1",
    "mutaco-paper-report-spec-v1",
    "mutaco-paper-figure-spec-v1",
    "mutaco-paper-exploration-spec-v1",
})


def _require_text(document: Mapping[str, Any], *fields: str) -> None:
    missing = [field for field in fields if not isinstance(document.get(field), str) or not str(document[field]).strip()]
    if missing:
        raise ValueError(f"paper workflow document lacks text fields: {missing}")


def _require_false(document: Mapping[str, Any], field: str) -> None:
    if document.get(field) is not False:
        raise ValueError(f"{field} must be explicitly false")


def _validate_study(document: Mapping[str, Any]) -> None:
    _require_text(document, "study_id", "revision", "classification", "domain")
    if not isinstance(document.get("scientific_inputs"), Mapping):
        raise ValueError("study scientific_inputs must be an object")


def _validate_execution(document: Mapping[str, Any]) -> None:
    _require_text(document, "execution_id", "study_identity", "source_commit")
    _require_false(document, "execution_authorized")
    for field in ("runtime", "scheduler"):
        if not isinstance(document.get(field), Mapping):
            raise ValueError(f"execution {field} must be an object")


def _validate_ingestion(document: Mapping[str, Any]) -> None:
    _require_text(document, "ingestion_id", "bundle_id", "cohort_identity")
    _require_false(document, "consolidation_authorized")
    methods = document.get("methods")
    fragments = document.get("source_fragments")
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods):
        raise ValueError("ingestion methods must be a nonempty unique list")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError("ingestion source_fragments must be nonempty")
    ids = []
    for fragment in fragments:
        if not isinstance(fragment, Mapping):
            raise ValueError("ingestion source fragment must be an object")
        _require_text(fragment, "source_id", "kind")
        if fragment["kind"] not in {"compact_bundle", "worker_population"}:
            raise ValueError("unsupported ingestion source kind")
        ids.append(fragment["source_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("ingestion source ids must be unique")
    if document.get("bundle_format") == "canonical_population_v1":
        _require_text(document, "cohort_manifest", "cohort_sha256", "output")
        for field in (
            "inference_budget_per_method_target", "bootstrap_seed",
            "bootstrap_resamples", "expected_target_count",
        ):
            value = document.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"canonical ingestion {field} must be positive")
        calls = document.get("expected_physical_calls")
        if not isinstance(calls, Mapping) or set(calls) != {
            "attempted_calls", "successful_calls", "failed_calls", "retry_calls"
        }:
            raise ValueError("canonical ingestion expected_physical_calls is invalid")
        for fragment in fragments:
            _require_text(
                fragment, "source_evidence_identity", "source_commit",
                "declaration_identity", "collection_sha256",
            )
            for field in ("target_count", "record_count", "successful_calls"):
                value = fragment.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"canonical source {field} must be positive")


def _validate_analysis(document: Mapping[str, Any]) -> None:
    _require_text(document, "analysis_id", "bundle_identity", "state")
    if not isinstance(document.get("metrics"), list) or not document["metrics"]:
        raise ValueError("analysis metrics must be nonempty")
    if not isinstance(document.get("procedures"), Mapping):
        raise ValueError("analysis procedures must be an object")


def _validate_report(document: Mapping[str, Any]) -> None:
    _require_text(document, "report_id", "study_id", "state", "output")
    budget = document.get("inference_budget_per_method_target")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("report inference budget must be positive")
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("report source must be an object")
    _require_text(source, "state", "path")
    if source["state"] not in {"accepted_tracked_evidence", "pending_canonical_consolidation"}:
        raise ValueError("unsupported report source state")
    if source["state"] == "accepted_tracked_evidence":
        _require_text(source, "sha256")
        if not isinstance(document.get("claims"), Mapping):
            raise ValueError("accepted report specifications require machine claims")
    elif document.get("claims") not in (None, {}):
        raise ValueError("pending reports cannot contain performance claims")
    if not isinstance(document.get("sections"), list) or not document["sections"]:
        raise ValueError("report sections must be nonempty")


def _validate_figure(document: Mapping[str, Any]) -> None:
    _require_text(document, "figure_set_id", "study_id", "state", "source_report", "output_root")
    _require_false(document, "rendering_authorized")
    if document["state"] != "discussion_pending":
        raise ValueError("foundation figure specifications must remain discussion_pending")
    if document.get("panels") != []:
        raise ValueError("foundation figure scaffolds cannot define panels")


def _validate_exploration(document: Mapping[str, Any]) -> None:
    _require_text(document, "workspace_id", "classification", "promotion_status")
    if document["classification"] != "exploratory_non_paper":
        raise ValueError("exploration must be explicitly non-paper")
    if document["promotion_status"] != "not_promoted":
        raise ValueError("new exploration cannot be pre-promoted")
    _require_false(document, "scientific_execution_authorized")


_VALIDATORS = {
    "mutaco-paper-study-spec-v1": _validate_study,
    "mutaco-paper-execution-spec-v1": _validate_execution,
    "mutaco-paper-ingestion-spec-v1": _validate_ingestion,
    "mutaco-paper-analysis-spec-v1": _validate_analysis,
    "mutaco-paper-report-spec-v1": _validate_report,
    "mutaco-paper-figure-spec-v1": _validate_figure,
    "mutaco-paper-exploration-spec-v1": _validate_exploration,
}


def validate_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structure and canonical identity without executing work."""

    value = dict(document)
    schema = str(value.get("schema_version", ""))
    if schema not in SCHEMAS:
        raise ValueError(f"unsupported paper workflow schema: {schema!r}")
    _VALIDATORS[schema](value)
    assert_document_identity(value)
    return value


def load_document(path: Path) -> dict[str, Any]:
    """Load a JSON or TOML workflow document."""

    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".toml":
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("paper workflow documents must be JSON or TOML")
    if not isinstance(value, Mapping):
        raise ValueError("paper workflow document must be an object")
    return dict(value)


def load_and_validate(path: Path) -> dict[str, Any]:
    return validate_document(load_document(path))
