"""Canonical, zero-call evidence adapters and deterministic paper statistics."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

from mutaco import stable_hash

from .cross_experiment_contract import (
    EVIDENCE_STATUSES,
    PAPER120_COHORT_SHA256,
    RECOMMENDATION_PROVENANCE,
    contract_document,
)
from .validation import PaperContractError


ROW_SCHEMA = "mutaco-cross-experiment-analysis-row-v1"
REPORT_SCHEMA = "mutaco-cross-experiment-analysis-report-v1"
SPEC_SCHEMA = "mutaco-cross-experiment-analysis-spec-v1"
ROW_IDENTITY_DOMAIN = "cross-experiment-analysis-row-v1"
REPORT_IDENTITY_DOMAIN = "cross-experiment-analysis-report-v1"
SPEC_IDENTITY_DOMAIN = "cross-experiment-analysis-spec-v1"

ERROR_KEYS = (
    "observed_search_error",
    "raw_terminal_surrogate_error",
    "optimized_terminal_surrogate_error",
    "heldout_assessment_error",
)
CALL_FIELDS = ("attempted_calls", "successful_calls", "failed_calls", "retry_calls")


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperContractError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PaperContractError(f"{path} must be finite")
    return result


def _metric(mae: Any, response_dimension: int, *, path: str) -> dict[str, float]:
    value = _finite(mae, path)
    if value < 0.0:
        raise PaperContractError(f"{path} must be nonnegative")
    return {"component_mean_mae": value, "response_l1_sum": value * response_dimension}


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "metric": None}


def _available(mae: Any, response_dimension: int, *, source: str, path: str) -> dict[str, Any]:
    return {
        "available": True,
        "source": source,
        "metric": _metric(mae, response_dimension, path=path),
    }


def _cohort_ids(cohort: Mapping[str, Any]) -> tuple[str, ...]:
    records = cohort.get("records")
    if not isinstance(records, Mapping):
        raise PaperContractError("cohort records are missing")
    try:
        positions = sorted(int(key) for key in records)
    except (TypeError, ValueError) as exc:
        raise PaperContractError("cohort positions are invalid") from exc
    if positions != list(range(120)):
        raise PaperContractError("canonical cohort must contain ordered positions 0..119")
    ids = tuple(str(records[str(position)].get("target_record_id", "")) for position in positions)
    if any(not value for value in ids) or len(set(ids)) != 120:
        raise PaperContractError("canonical cohort target identities are invalid")
    return ids


def _target_responses(cohort_directory: Path, cohort: Mapping[str, Any]) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    for position, target_id in enumerate(_cohort_ids(cohort)):
        path = cohort_directory / f"target_{position:03d}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("target_record_id") != target_id:
            raise PaperContractError(f"cohort target_{position:03d} identity mismatch")
        response = tuple(_finite(value, f"target[{position}].response") for value in document.get("target_response", ()))
        if not response:
            raise PaperContractError(f"cohort target_{position:03d} response is missing")
        result[target_id] = response
    return result


def _physical_mae(response: Sequence[Any], target: Sequence[float], *, path: str) -> float:
    values = tuple(_finite(value, path) for value in response)
    if len(values) != len(target):
        raise PaperContractError(f"{path} response dimension mismatch")
    return sum(abs(left - right) for left, right in zip(values, target, strict=True)) / len(target)


def _close(actual: float, expected: Any, path: str, *, tolerance: float = 2e-6) -> None:
    reference = _finite(expected, path)
    if not math.isclose(actual, reference, rel_tol=tolerance, abs_tol=tolerance):
        raise PaperContractError(f"{path} metric recomputation mismatch")


def _call_totals(ledger: Mapping[str, Any], *, budget: int, path: str) -> dict[str, int]:
    totals = {field: 0 for field in CALL_FIELDS}
    for category, entry in ledger.items():
        if not isinstance(entry, Mapping):
            raise PaperContractError(f"{path}.{category} call ledger is invalid")
        for field in CALL_FIELDS:
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PaperContractError(f"{path}.{category}.{field} is invalid")
            totals[field] += value
    inference = ledger.get("method_inference_or_training", {})
    assessment = ledger.get("paired_assessment_and_metric", {})
    if inference.get("successful_calls") != budget or assessment.get("successful_calls") != 1:
        raise PaperContractError(f"{path} does not contain exactly B search calls plus one assessment")
    if totals["failed_calls"] or totals["retry_calls"] or totals["successful_calls"] != budget + 1:
        raise PaperContractError(f"{path} call accounting is not a complete no-retry record")
    return totals


def _row(body: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(body))
    value["schema_version"] = ROW_SCHEMA
    value["row_identity"] = stable_hash(ROW_IDENTITY_DOMAIN, value)
    validate_analysis_rows((value,), require_complete_grid=False)
    return value


def _scalar_provenance(method: str) -> dict[str, Any]:
    return {
        "stage": "best_observed" if method == "sobol" else "corrected_refined_terminal",
        "policy_conformance": "current_declared_policy",
    }


def adapt_scalar_primary_export(
    export: Mapping[str, Any],
    *,
    source_identity: str,
    source_sha256: str,
    cohort: Mapping[str, Any],
    cohort_directory: Path,
    cohort_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Adapt full scalar-primary records without changing their native evidence."""

    if export.get("schema_version") != "mutaco-scalar-primary-compact-export-v1":
        raise PaperContractError("scalar primary compact export schema mismatch")
    if cohort_sha256 != PAPER120_COHORT_SHA256:
        raise PaperContractError("scalar primary adapter received the wrong cohort hash")
    cohort_ids = _cohort_ids(cohort)
    positions = {target_id: position for position, target_id in enumerate(cohort_ids)}
    targets = _target_responses(cohort_directory, cohort)
    records = export.get("records")
    if not isinstance(records, list):
        raise PaperContractError("scalar primary records are missing")
    rows = []
    for native in records:
        target_id = str(native.get("target_id", ""))
        if target_id not in positions:
            raise PaperContractError("scalar primary record is off cohort")
        method = str(native.get("method", ""))
        if method not in {"mutaco", "scalar_qlognei", "sobol"}:
            raise PaperContractError("scalar primary method is unsupported")
        if native.get("status") != "completed":
            raise PaperContractError("scalar primary analysis requires completed records")
        budget = int(native.get("budget", -1))
        decision = native.get("decision")
        if not isinstance(decision, Mapping) or decision.get("status") != "recommended":
            raise PaperContractError("scalar primary decision is incomplete")
        trajectory = decision.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != budget:
            raise PaperContractError("scalar primary trajectory length differs from budget")
        target = targets[target_id]
        observed = []
        for index, item in enumerate(trajectory, 1):
            if item.get("evaluation_index") != index:
                raise PaperContractError("scalar primary trajectory order is invalid")
            value = _physical_mae(item.get("response_physical", ()), target, path=f"trajectory[{index}]")
            _close(value, item.get("observed_physical_mae"), f"trajectory[{index}].observed_physical_mae")
            observed.append(_finite(item.get("observed_physical_mae"), f"trajectory[{index}].observed_physical_mae"))
        observed_best = min(observed)
        _close(observed_best, trajectory[-1].get("best_observed_mae_so_far"), "trajectory.best_observed_mae_so_far")
        response_dimension = len(target)
        recomputed_heldout_mae = _physical_mae(native.get("heldout_response", ()), target, path="heldout_response")
        _close(recomputed_heldout_mae, native.get("heldout_physical_mae"), "heldout_physical_mae")
        heldout_mae = _finite(native.get("heldout_physical_mae"), "heldout_physical_mae")
        metadata = decision.get("terminal_metadata", {})
        if not isinstance(metadata, Mapping):
            raise PaperContractError("scalar primary terminal metadata is missing")
        if method == "mutaco":
            histories = metadata.get("restart_histories")
            if not isinstance(histories, list) or not histories:
                raise PaperContractError("MUTACO raw terminal history is missing")
            raw_error = _available(histories[0].get("initial_score"), response_dimension, source="final_gp_raw_terminal_screen", path="terminal.raw_score")
            predicted = metadata.get("predicted_physical_response")
            recomputed_terminal_mae = _physical_mae(predicted, target, path="terminal.predicted_physical_response")
            _close(recomputed_terminal_mae, metadata.get("score"), "terminal.score")
            terminal_mae = _finite(metadata.get("score"), "terminal.score")
            terminal_error = _available(terminal_mae, response_dimension, source="final_gp_posterior_mean_at_returned_recommendation", path="terminal.score")
        elif method == "scalar_qlognei":
            raw_error = _unavailable("native terminal evidence does not persist raw-screen score")
            terminal_error = _available(metadata.get("score"), response_dimension, source="final_scalar_gp_posterior_mean_loss", path="terminal.score")
        else:
            raw_error = _unavailable("surrogate-free method")
            terminal_error = _unavailable("surrogate-free method")
            _close(observed_best, metadata.get("score"), "sobol.terminal.score")
        search_calls = decision.get("search_calls", {})
        if not isinstance(search_calls, Mapping) or search_calls.get("attempted") != budget or search_calls.get("successful") != budget:
            raise PaperContractError("scalar primary search-call accounting differs from budget")
        calls = {
            "attempted_calls": budget + 1,
            "successful_calls": budget + 1,
            "failed_calls": 0,
            "retry_calls": 0,
        }
        timing = decision.get("timing", {})
        if not isinstance(timing, Mapping):
            raise PaperContractError("scalar primary timing record is missing")
        rows.append(_row({
            "study_family": "scalar_primary",
            "study_identity": str(native.get("study_identity", "")),
            "campaign_id": str(native.get("campaign_id", "")),
            "source": {"adapter": "scalar_primary_compact_export_v1", "evidence_identity": source_identity, "sha256": source_sha256},
            "target_position": positions[target_id],
            "target_id": target_id,
            "budget": budget,
            "method": method,
            "native_record_identity": str(native.get("record_identity", "")),
            "native_search_policy": str(decision.get("search_policy_identifier", "")),
            "native_terminal_policy": str(decision.get("terminal_policy_identifier", "")),
            "recommendation_provenance": _scalar_provenance(method),
            "evidence_statuses": ["valid_current_policy_evidence"],
            "response_dimension": response_dimension,
            "errors": {
                "observed_search_error": _available(observed_best, response_dimension, source="recomputed_noisy_search_trajectory", path="observed_best"),
                "raw_terminal_surrogate_error": raw_error,
                "optimized_terminal_surrogate_error": terminal_error,
                "heldout_assessment_error": _available(heldout_mae, response_dimension, source="recomputed_fresh_heldout_response", path="heldout_mae"),
            },
            "assessment": {"pairing_status": "exact_context_available", "context": deepcopy(native.get("assessment_context"))},
            "calls": calls,
            "timing": {name: (_finite(value, f"timing.{name}") if value is not None else None) for name, value in sorted(timing.items())},
            "diagnostics": {
                "terminal_candidate_source": metadata.get("returned_candidate_source"),
                "terminal_precision_fallback_used": metadata.get("precision_fallback_used"),
                "terminal_posterior_calls": metadata.get("posterior_calls"),
            },
        }))
    result = tuple(sorted(rows, key=lambda item: (item["target_position"], item["method"])))
    validate_analysis_rows(result, expected_methods=("mutaco", "scalar_qlognei", "sobol"), expected_budget=int(result[0]["budget"]), expected_targets=cohort_ids)
    return result


def adapt_paper_v3_bundle(
    records: Iterable[Mapping[str, Any]],
    *,
    source_identity: str,
    source_sha256: str,
    cohort: Mapping[str, Any],
    budget: int,
) -> tuple[dict[str, Any], ...]:
    """Adapt historical MUTACO/NPE/SNPE endpoints with explicit legacy limits."""

    cohort_ids = _cohort_ids(cohort)
    result = []
    for paired in records:
        position = paired.get("target_position")
        if isinstance(position, bool) or not isinstance(position, int) or position not in range(120):
            raise PaperContractError("Paper-v3 target position is invalid")
        target_id = str(paired.get("target_record_identity", ""))
        if cohort_ids[position] != target_id:
            raise PaperContractError("Paper-v3 target order differs from canonical cohort")
        methods = paired.get("methods")
        if not isinstance(methods, Mapping) or set(methods) != {"mutaco", "sbi_npe", "sbi_snpe"}:
            raise PaperContractError("Paper-v3 paired method grid is incomplete")
        for method, native in methods.items():
            if native.get("status") != "completed":
                raise PaperContractError("Paper-v3 method record is incomplete")
            response_dimension = 20
            mae = _finite(native.get("response_error_mae"), "Paper-v3 heldout MAE")
            _close(mae * response_dimension, native.get("response_error_l1_sum"), "Paper-v3 L1", tolerance=1e-10)
            calls = _call_totals(native.get("call_ledger", {}), budget=budget, path="Paper-v3.call_ledger")
            stage = "historical_one_step_terminal" if method == "mutaco" else None
            statuses = ["valid_historical_policy_evidence"]
            if method == "mutaco":
                statuses.append("protocol_noncompliant_for_current_claim")
            timing_events = native.get("timing_events", [])
            method_wall = None
            assessment_wall = None
            if isinstance(timing_events, list):
                for event in timing_events:
                    if event.get("phase") == "method_inference_and_zero_call_decision_dispatch":
                        method_wall = event.get("wall_seconds")
                    elif event.get("phase") == "paired_held_out_assessment":
                        assessment_wall = event.get("wall_seconds")
            result.append(_row({
                "study_family": "historical_paper_v3_sbi_comparison",
                "study_identity": "graph_paper120_mutaco_npe_snpe_v1",
                "campaign_id": f"graph_paper120_b{budget}_mutaco_npe_snpe_v1",
                "source": {"adapter": "historical_paper_v3_paired_records_v1", "evidence_identity": source_identity, "sha256": source_sha256},
                "target_position": position,
                "target_id": target_id,
                "budget": budget,
                "method": method,
                "native_record_identity": str(native.get("record_content_hash", "")),
                "native_search_policy": None,
                "native_terminal_policy": str(native.get("decision_policy_identifier", "")),
                "recommendation_provenance": {
                    "stage": stage,
                    "policy_conformance": "historical_executed_policy",
                    "not_applicable_reason": None if stage else "SBI posterior-sample MAP is not a GP terminal stage",
                },
                "evidence_statuses": statuses,
                "response_dimension": response_dimension,
                "errors": {
                    "observed_search_error": _unavailable("not retained in consolidated Paper-v3 endpoint bundle"),
                    "raw_terminal_surrogate_error": _unavailable("not retained in consolidated Paper-v3 endpoint bundle"),
                    "optimized_terminal_surrogate_error": _unavailable("not retained in consolidated Paper-v3 endpoint bundle"),
                    "heldout_assessment_error": _available(mae, response_dimension, source="native_paired_heldout_physical_metric", path="heldout_mae"),
                },
                "assessment": {
                    "pairing_status": "partial_legacy_context" if method != "sbi_snpe" else "native_context_available_for_this_method_only",
                    "context": deepcopy(native.get("assessment_context")),
                },
                "calls": calls,
                "timing": {
                    "method_dispatch_total": None if method_wall is None else _finite(method_wall, "method_wall"),
                    "heldout_assessment": None if assessment_wall is None else _finite(assessment_wall, "assessment_wall"),
                },
                "diagnostics": {"historical_terminal_optimizer_audit_required": method == "mutaco"},
            }))
    rows = tuple(sorted(result, key=lambda item: (item["target_position"], item["method"])))
    validate_analysis_rows(rows, expected_methods=("mutaco", "sbi_npe", "sbi_snpe"), expected_budget=budget, expected_targets=cohort_ids, require_exact_assessment_pairing=False)
    return rows


def adapt_repaired_sbi_assessment_export(
    export: Mapping[str, Any],
    *,
    source_identity: str,
    source_sha256: str,
    cohort: Mapping[str, Any],
    cohort_directory: Path,
    cohort_sha256: str,
    budget: int,
) -> tuple[dict[str, Any], ...]:
    """Adapt the fresh repaired MUTACO--SBI paired assessment endpoint."""

    if export.get("schema_version") != "mutaco-repaired-sbi-assessment-compact-v1":
        raise PaperContractError("repaired SBI compact export schema mismatch")
    if export.get("collection_identity") != source_identity:
        raise PaperContractError("repaired SBI collection identity mismatch")
    if cohort_sha256 != PAPER120_COHORT_SHA256:
        raise PaperContractError("repaired SBI adapter received the wrong cohort hash")
    if budget not in {40, 100}:
        raise PaperContractError("repaired SBI analysis budget is unsupported")
    cohort_ids = _cohort_ids(cohort)
    targets = _target_responses(cohort_directory, cohort)
    records = export.get("records")
    if not isinstance(records, list):
        raise PaperContractError("repaired SBI assessment records are missing")
    selected = [record for record in records if record.get("budget") == budget]
    methods = ("mutaco", "sbi_npe", "sbi_snpe")
    rows = []
    for native in selected:
        position = native.get("target_position")
        if isinstance(position, bool) or not isinstance(position, int) or position not in range(120):
            raise PaperContractError("repaired SBI target position is invalid")
        target_id = str(native.get("target_record_id", ""))
        if cohort_ids[position] != target_id:
            raise PaperContractError("repaired SBI target order differs from canonical cohort")
        method = str(native.get("method", ""))
        if method not in methods:
            raise PaperContractError("repaired SBI method is unsupported")
        target = targets[target_id]
        response_dimension = len(target)
        heldout_mae = _physical_mae(
            native.get("assessment_response", ()), target,
            path=f"repaired_sbi[{budget},{position},{method}].assessment_response",
        )
        _close(heldout_mae, native.get("response_error_mae"), "repaired SBI heldout MAE")
        _close(heldout_mae * response_dimension, native.get("response_error_l1_sum"), "repaired SBI heldout L1")
        accounting = native.get("call_accounting")
        if accounting != {
            "assessment_attempted_calls": 1,
            "assessment_failed_calls": 0,
            "assessment_successful_calls": 1,
            "gp_fit_attempts": 0,
            "retry_calls": 0,
            "search_calls": 0,
            "terminal_optimizations": 0,
        }:
            raise PaperContractError("repaired SBI assessment call accounting is invalid")
        timing = native.get("timing")
        if not isinstance(timing, Mapping):
            raise PaperContractError("repaired SBI assessment timing is missing")
        stage = "corrected_refined_terminal" if method == "mutaco" else None
        rows.append(_row({
            "study_family": "repaired_mutaco_sbi_comparison",
            "study_identity": "paper120_repaired_mutaco_npe_snpe_assessment_v1",
            "campaign_id": "paper120_repaired_mutaco_npe_snpe_b40_b100_assessment_v1",
            "source": {
                "adapter": "repaired_sbi_assessment_compact_export_v1",
                "evidence_identity": source_identity,
                "sha256": source_sha256,
            },
            "target_position": position,
            "target_id": target_id,
            "budget": budget,
            "method": method,
            "native_record_identity": str(native.get("result_identity", "")),
            "native_search_policy": "historical_budget_complete_search_reused_v1",
            "native_terminal_policy": (
                "continuous_posterior_mean_physical_l1_mae"
                if method == "mutaco" else "posterior_sample_map_physical_v1"
            ),
            "recommendation_provenance": {
                "stage": stage,
                "policy_conformance": "current_declared_policy",
                "not_applicable_reason": None if stage else "SBI uses an immutable posterior-sample MAP recommendation",
            },
            "evidence_statuses": ["valid_current_policy_evidence"],
            "response_dimension": response_dimension,
            "errors": {
                "observed_search_error": _unavailable("assessment-only compact evidence; search evidence remains in its native archive"),
                "raw_terminal_surrogate_error": _unavailable("not part of the fresh heldout assessment endpoint"),
                "optimized_terminal_surrogate_error": _unavailable("not part of the fresh heldout assessment endpoint"),
                "heldout_assessment_error": _available(
                    heldout_mae, response_dimension,
                    source="recomputed_fresh_paired_heldout_response",
                    path="repaired SBI heldout MAE",
                ),
            },
            "assessment": {"pairing_status": "exact_context_available", "context": deepcopy(native.get("assessment_context"))},
            "calls": {"attempted_calls": 1, "successful_calls": 1, "failed_calls": 0, "retry_calls": 0},
            "timing": {
                "heldout_assessment_wall": _finite(timing.get("wall_seconds"), "repaired SBI wall timing"),
                "heldout_assessment_process_cpu": _finite(timing.get("process_cpu_seconds"), "repaired SBI CPU timing"),
            },
            "diagnostics": {
                "assessment_randomness_evidence_identity": stable_hash(
                    "repaired-sbi-assessment-randomness-evidence-v1",
                    native.get("assessment_randomness_evidence"),
                ),
                "recommendation_identity": str(native.get("recommendation_identity", "")),
                "reused_historical_search": True,
            },
        }))
    result = tuple(sorted(rows, key=lambda item: (item["target_position"], item["method"])))
    validate_analysis_rows(
        result, expected_methods=methods, expected_budget=budget,
        expected_targets=cohort_ids, require_exact_assessment_pairing=True,
    )
    return result


def adapt_mechanism_ablation_records(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], ...]:
    """Prospective adapter boundary; no mechanism evidence exists yet."""

    records = kwargs.get("records", args[0] if args else ())
    if tuple(records):
        raise PaperContractError("mechanism-ablation native schema must be frozen before adaptation")
    return ()


def validate_analysis_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str] | None = None,
    expected_budget: int | None = None,
    expected_targets: Sequence[str] | None = None,
    require_complete_grid: bool = True,
    require_exact_assessment_pairing: bool = True,
) -> dict[str, Any]:
    if not rows:
        raise PaperContractError("analysis rows are empty")
    identities = []
    grid: dict[tuple[int, str], Mapping[str, Any]] = {}
    for item in rows:
        row = dict(item)
        if row.get("schema_version") != ROW_SCHEMA:
            raise PaperContractError("analysis row schema mismatch")
        identity = row.pop("row_identity", None)
        if identity != stable_hash(ROW_IDENTITY_DOMAIN, row):
            raise PaperContractError("analysis row identity mismatch")
        identities.append(identity)
        if tuple(row.get("errors", {}).keys()) != ERROR_KEYS:
            raise PaperContractError("analysis row error vocabulary is incomplete or reordered")
        statuses = row.get("evidence_statuses")
        if not isinstance(statuses, list) or not statuses or any(value not in EVIDENCE_STATUSES for value in statuses):
            raise PaperContractError("analysis evidence status is invalid")
        stage = row.get("recommendation_provenance", {}).get("stage")
        if stage is not None and stage not in RECOMMENDATION_PROVENANCE:
            raise PaperContractError("analysis recommendation provenance is invalid")
        calls = row.get("calls")
        if not isinstance(calls, Mapping) or set(calls) != set(CALL_FIELDS):
            raise PaperContractError("analysis call accounting is incomplete")
        if any(isinstance(calls[field], bool) or not isinstance(calls[field], int) or calls[field] < 0 for field in CALL_FIELDS):
            raise PaperContractError("analysis call accounting is invalid")
        key = (int(row.get("target_position", -1)), str(row.get("method", "")))
        if key in grid:
            raise PaperContractError("analysis grid contains a duplicate target-method row")
        grid[key] = item
    if len(set(identities)) != len(identities):
        raise PaperContractError("analysis row identities are duplicated")
    if require_complete_grid:
        if expected_methods is None or expected_targets is None or expected_budget is None:
            raise PaperContractError("complete-grid validation requires methods, targets, and budget")
        expected = {(position, method) for position in range(len(expected_targets)) for method in expected_methods}
        if set(grid) != expected:
            raise PaperContractError("analysis target-method grid is incomplete")
        for (position, _), row in grid.items():
            if row.get("target_id") != expected_targets[position] or row.get("budget") != expected_budget:
                raise PaperContractError("analysis target order or budget mismatch")
        if require_exact_assessment_pairing:
            for position in range(len(expected_targets)):
                contexts = [grid[(position, method)]["assessment"] for method in expected_methods]
                if any(value.get("pairing_status") != "exact_context_available" for value in contexts):
                    raise PaperContractError("analysis assessment context is not exact")
                canonical = [json.dumps(value.get("context"), sort_keys=True, separators=(",", ":")) for value in contexts]
                if len(set(canonical)) != 1:
                    raise PaperContractError("paired assessment context differs across methods")
    return {"status": "valid", "rows": len(rows), "unique_rows": len(identities)}


def analysis_specification(*, analysis_id: str, bootstrap_seed: int, bootstrap_resamples: int) -> dict[str, Any]:
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise PaperContractError("bootstrap seed must be a nonnegative integer")
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples <= 0:
        raise PaperContractError("bootstrap resamples must be positive")
    body = {
        "schema_version": SPEC_SCHEMA,
        "analysis_id": analysis_id,
        "contract_identity": contract_document()["contract_identity"],
        "row_schema": ROW_SCHEMA,
        "error_quantities": list(ERROR_KEYS),
        "primary_metric": "heldout_assessment_error.component_mean_mae",
        "summaries": ["n", "mean", "standard_deviation", "median", "q1", "q3", "minimum", "maximum"],
        "paired_procedures": ["mean_difference", "median_difference", "wins_ties_losses", "mean_rank", "seeded_percentile_bootstrap_95"],
        "bootstrap": {"algorithm": "python_random_index_resampling_v1", "seed": bootstrap_seed, "resamples": bootstrap_resamples, "lower_percentile": 0.025, "upper_percentile": 0.975},
        "timing_policy": "descriptive_persisted_components_only_null_is_not_zero",
        "missing_metric_policy": "retain_explicit_unavailable_reason_never_impute",
    }
    return {**body, "analysis_spec_identity": stable_hash(SPEC_IDENTITY_DOMAIN, body)}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "standard_deviation": None, "median": None, "q1": None, "q3": None, "minimum": None, "maximum": None}
    return {
        "n": len(values), "mean": mean(values), "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values), "q1": _quantile(values, 0.25), "q3": _quantile(values, 0.75),
        "minimum": min(values), "maximum": max(values),
    }


def _bootstrap_interval(values: Sequence[float], *, seed: int, resamples: int) -> list[float]:
    generator = random.Random(seed)
    n = len(values)
    draws = [mean([values[generator.randrange(n)] for _ in range(n)]) for _ in range(resamples)]
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def deterministic_analysis(rows: Sequence[Mapping[str, Any]], specification: Mapping[str, Any]) -> dict[str, Any]:
    validate_analysis_rows(rows, require_complete_grid=False)
    if specification.get("schema_version") != SPEC_SCHEMA:
        raise PaperContractError("analysis specification schema mismatch")
    spec_body = dict(specification)
    identity = spec_body.pop("analysis_spec_identity", None)
    if identity != stable_hash(SPEC_IDENTITY_DOMAIN, spec_body):
        raise PaperContractError("analysis specification identity mismatch")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    targets: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[str(row["method"])].append(row)
        targets[int(row["target_position"])][str(row["method"])] = row
    methods = tuple(sorted(groups))
    if len({int(row["budget"]) for row in rows}) != 1:
        raise PaperContractError("one analysis report cannot mix budgets")
    if len({str(row["study_family"]) for row in rows}) != 1 or len({str(row["source"]["evidence_identity"]) for row in rows}) != 1:
        raise PaperContractError("one analysis report cannot mix study families or evidence sources")
    expected_positions = list(range(len(targets)))
    if sorted(targets) != expected_positions or any(set(targets[position]) != set(methods) for position in expected_positions):
        raise PaperContractError("analysis report requires one complete ordered target-method grid")
    for method in methods:
        policy_labels = {
            (
                row.get("native_search_policy"),
                row.get("native_terminal_policy"),
                json.dumps(row.get("recommendation_provenance"), sort_keys=True, separators=(",", ":")),
                tuple(row.get("evidence_statuses", ())),
            )
            for row in groups[method]
        }
        if len(policy_labels) != 1:
            raise PaperContractError("analysis report cannot mix policies or evidence labels within a method")
    error_summaries = {}
    timing_summaries = {}
    call_summaries = {}
    terminal_diagnostics = {}
    for method in methods:
        error_summaries[method] = {}
        for error_key in ERROR_KEYS:
            values = [float(row["errors"][error_key]["metric"]["component_mean_mae"]) for row in groups[method] if row["errors"][error_key]["available"]]
            error_summaries[method][error_key] = _summary(values)
        timing_names = sorted({name for row in groups[method] for name in row.get("timing", {})})
        timing_summaries[method] = {name: _summary([float(row["timing"][name]) for row in groups[method] if row["timing"].get(name) is not None]) for name in timing_names}
        call_summaries[method] = {field: sum(int(row["calls"][field]) for row in groups[method]) for field in CALL_FIELDS}
        diagnostic_names = sorted({name for row in groups[method] for name in row.get("diagnostics", {})})
        terminal_diagnostics[method] = {
            name: {
                "available_records": sum(row["diagnostics"].get(name) is not None for row in groups[method]),
                "value_counts": {
                    str(value): sum(row["diagnostics"].get(name) == value for row in groups[method])
                    for value in sorted({row["diagnostics"].get(name) for row in groups[method] if row["diagnostics"].get(name) is not None}, key=str)
                },
            }
            for name in diagnostic_names
        }
    paired = {}
    bootstrap = specification["bootstrap"]
    primary = "heldout_assessment_error"
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            differences = []
            for position in sorted(targets):
                pair = targets[position]
                if left not in pair or right not in pair:
                    raise PaperContractError("paired analysis grid is incomplete")
                left_error = pair[left]["errors"][primary]
                right_error = pair[right]["errors"][primary]
                if not left_error["available"] or not right_error["available"]:
                    raise PaperContractError("paired primary endpoint is unavailable")
                differences.append(float(left_error["metric"]["component_mean_mae"]) - float(right_error["metric"]["component_mean_mae"]))
            pair_seed = int(stable_hash("cross-experiment-bootstrap-pair-seed-v1", {"root_seed": bootstrap["seed"], "left": left, "right": right}).rsplit("-", 1)[-1][:16], 16)
            paired[f"{left}__{right}"] = {
                "n": len(differences), "mean_left_minus_right": mean(differences), "median_left_minus_right": median(differences),
                "left_wins": sum(value < 0 for value in differences), "ties": sum(value == 0 for value in differences), "left_losses": sum(value > 0 for value in differences),
                "paired_mean_bootstrap_95_percentile_interval": _bootstrap_interval(differences, seed=pair_seed, resamples=int(bootstrap["resamples"])),
                "derived_pair_seed": pair_seed,
            }
    wins = {method: 0 for method in methods}
    winner_ties = 0
    ranks = {method: [] for method in methods}
    for pair in targets.values():
        scores = {method: float(pair[method]["errors"][primary]["metric"]["component_mean_mae"]) for method in methods}
        best = min(scores.values())
        winners = tuple(method for method in methods if scores[method] == best)
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            winner_ties += 1
        ordered_values = sorted(scores.values())
        for method, score in scores.items():
            matching = [index + 1 for index, value in enumerate(ordered_values) if value == score]
            ranks[method].append(mean(matching))
    body = {
        "schema_version": REPORT_SCHEMA,
        "analysis_spec_identity": identity,
        "contract_identity": contract_document()["contract_identity"],
        "row_count": len(rows), "target_count": len(targets), "methods": list(methods),
        "budgets": sorted({int(row["budget"]) for row in rows}),
        "error_summaries_component_mean_mae": error_summaries,
        "paired_primary_endpoint": paired,
        "target_winners": wins,
        "target_winner_ties": winner_ties,
        "mean_rank": {method: mean(values) for method, values in ranks.items()},
        "call_accounting": call_summaries,
        "descriptive_timing_seconds": timing_summaries,
        "terminal_diagnostics": terminal_diagnostics,
        "scientific_calls_added_by_analysis": 0,
    }
    return {**body, "analysis_report_identity": stable_hash(REPORT_IDENTITY_DOMAIN, body)}


def canonical_json_text(document: Any) -> str:
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
