"""Validated, computation-free loading and paired paper summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from statistics import mean, median, stdev
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..common.method_append import load_method_record
from ..common.integrity import load_execution_receipts, validate_attempt_journals
from ..common.validation import PaperContractError, validate_method_ids
from ..common.protocol_v3 import RECORD_SCHEMA_V2, RECORD_SCHEMA_V3, V3_METRIC_FIELDS


@dataclass(frozen=True)
class PaperRecordSet:
    root: Path
    records: tuple[dict[str, Any], ...]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)


def _validate_cohort(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    schema = records[0].get("schema_version")
    if any(row.get("schema_version") != schema for row in records[1:]):
        raise PaperContractError("incompatible paper record cohort: schema_version")
    if schema == RECORD_SCHEMA_V2:
        invariants = (
            "campaign_identity", "simulator_identity", "simulator_configuration_hash",
            "target_bank_identity", "hidden_target_ledger_hash", "inference_budget",
            "budget_interpretation", "evaluation_randomness_policy", "assessment_policy",
            "parameter_dimension", "response_dimension", "dtype_policy",
            "resolved_device", "software_compatibility_hash",
        )
    elif schema == RECORD_SCHEMA_V3:
        invariants = (
            "paper_comparison_protocol",
            "paper_comparison_configuration_identity",
            "simulator_definition_identity",
            "simulator_configuration_identity",
            "assessment_fidelity",
            "assessment_policy",
            "metric_schema",
        )
    else:
        raise PaperContractError(f"unsupported paper record cohort schema: {schema!r}")
    first = records[0]
    mismatches = [field for field in invariants if any(row.get(field) != first.get(field) for row in records[1:])]
    if mismatches:
        raise PaperContractError("incompatible paper record cohort: " + ", ".join(mismatches))
    keys = [(row["target_record_identity"], row["method"]) for row in records]
    if len(keys) != len(set(keys)):
        raise PaperContractError("duplicate target-method records")


def load_paper_records(path: str | Path) -> PaperRecordSet:
    root = Path(path).resolve()
    receipts = load_execution_receipts(root)
    validate_attempt_journals(root, receipts)
    records = []
    for record_file in sorted((root / "records").glob("*/*.json")):
        record = load_method_record(record_file)
        assert record is not None
        records.append(record)
    _validate_cohort(records)
    return PaperRecordSet(root, tuple(records))


def select_methods(records: PaperRecordSet, methods: Sequence[str]) -> PaperRecordSet:
    selected = validate_method_ids(methods)
    available = {record["method"] for record in records.records}
    missing = set(selected) - available
    if missing:
        raise PaperContractError(f"selected methods have no records: {sorted(missing)}")
    return PaperRecordSet(
        records.root,
        tuple(record for record in records.records if record["method"] in selected),
    )


def summarize_paired(records: PaperRecordSet | Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(records.records if isinstance(records, PaperRecordSet) else records)
    _validate_cohort(rows)
    completed = [row for row in rows if row["status"] == "completed"]
    methods = sorted({row["method"] for row in rows})
    by_target = {
        target: {row["method"]: row for row in rows if row["target_record_identity"] == target}
        for target in sorted({row["target_record_identity"] for row in rows})
    }
    schema = rows[0].get("schema_version") if rows else RECORD_SCHEMA_V2
    metrics = (
        ("target_error_l1", "target_error_l2", "design_error_l2")
        if schema == RECORD_SCHEMA_V2
        else V3_METRIC_FIELDS
    )
    response_l2_field = (
        "target_error_l2" if schema == RECORD_SCHEMA_V2 else "response_error_l2"
    )
    method_summary = {}
    for method in methods:
        method_rows = [row for row in completed if row["method"] == method]
        method_summary[method] = {
            "completed": len(method_rows),
            "failed": sum(row["method"] == method and row["status"] == "failed" for row in rows),
            "means": {
                metric: mean(float(row[metric]) for row in method_rows if row.get(metric) is not None)
                if any(row.get(metric) is not None for row in method_rows) else None
                for metric in metrics
            },
        }
    paired = []
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            common = [items for items in by_target.values() if left in items and right in items and items[left]["status"] == items[right]["status"] == "completed"]
            paired.append({
                "left": left,
                "right": right,
                "target_count": len(common),
                "mean_left_minus_right_response_l2": (
                    mean(float(items[left][response_l2_field]) - float(items[right][response_l2_field]) for items in common)
                    if common else None
                ),
            })
    if schema == RECORD_SCHEMA_V2:
        for item in paired:
            item["mean_left_minus_right_target_l2"] = item.pop(
                "mean_left_minus_right_response_l2"
            )
    return {
        "record_schema": schema,
        "join_key": "target_record_identity",
        "metrics": metrics,
        "methods": method_summary,
        "paired": paired,
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {key: None for key in (
            "mean", "median", "sample_standard_deviation", "q1", "q3", "iqr",
            "p90", "p95", "minimum", "maximum", "lower_whisker", "upper_whisker",
        )} | {"count": 0, "outliers": []}
    q1, q3 = _percentile(ordered, 0.25), _percentile(ordered, 0.75)
    assert q1 is not None and q3 is not None
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = [value for value in ordered if low_fence <= value <= high_fence]
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "median": median(ordered),
        "sample_standard_deviation": stdev(ordered) if len(ordered) > 1 else None,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "lower_whisker": inside[0],
        "upper_whisker": inside[-1],
        "outliers": [value for value in ordered if value < low_fence or value > high_fence],
    }


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, resamples: int
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    lower = _percentile(samples, 0.025)
    upper = _percentile(samples, 0.975)
    assert lower is not None and upper is not None
    return [lower, upper]


def _sign_test_two_sided(wins: int, losses: int) -> float | None:
    count = wins + losses
    if count == 0:
        return None
    tail = sum(math.comb(count, k) for k in range(0, min(wins, losses) + 1)) / (2**count)
    return min(1.0, 2.0 * tail)


def _timing_summary(rows: Sequence[Mapping[str, Any]], phase: str) -> dict[str, float | None]:
    events = [
        event
        for row in rows
        for event in row.get("population_timing_events", ())
        if event.get("phase") == phase
    ]
    return {
        "wall_seconds": sum(float(event["wall_seconds"]) for event in events) if events else None,
        "process_cpu_seconds": (
            sum(float(event["process_cpu_seconds"]) for event in events)
            if events else None
        ),
    }


def summarize_population_v3(
    records: PaperRecordSet | Iterable[Mapping[str, Any]], *,
    declared_target_identities: Sequence[str], bootstrap_seed: int,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Full deterministic v3 summaries with explicit failure denominators."""

    rows = tuple(records.records if isinstance(records, PaperRecordSet) else records)
    _validate_cohort(rows)
    if rows and rows[0].get("schema_version") != RECORD_SCHEMA_V3:
        raise PaperContractError("population v3 summary requires v3 records")
    targets = tuple(str(value) for value in declared_target_identities)
    if len(set(targets)) != len(targets):
        raise PaperContractError("declared target identities must be unique")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise PaperContractError("bootstrap_seed must be a nonnegative integer")
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples <= 0:
        raise PaperContractError("bootstrap_resamples must be positive")
    target_set = set(targets)
    if any(str(row["target_record_identity"]) not in target_set for row in rows):
        raise PaperContractError("record lies outside declared target cohort")
    methods = sorted({str(row["method"]) for row in rows})
    slots = {(str(row["target_record_identity"]), str(row["method"])): row for row in rows}
    method_documents: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        present = [slots[(target, method)] for target in targets if (target, method) in slots]
        completed = [row for row in present if row["status"] == "completed"]
        failed = [row for row in present if row["status"] == "failed"]
        metric_documents = {}
        for metric_index, metric in enumerate(V3_METRIC_FIELDS):
            values = [float(row[metric]) for row in completed if row.get(metric) is not None]
            metric_documents[metric] = {
                **_descriptive(values),
                "mean_bootstrap_95_percentile_interval": _bootstrap_mean_interval(
                    values,
                    seed=bootstrap_seed + 1009 * method_index + 9176 * metric_index,
                    resamples=bootstrap_resamples,
                ),
            }
        method_documents[method] = {
            "declared_target_count": len(targets),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "missing_count": len(targets) - len(present),
            "failure_rate_full_denominator": len(failed) / len(targets) if targets else None,
            "missing_target_identities": [target for target in targets if (target, method) not in slots],
            "failure_reasons": [
                {"target_record_identity": row["target_record_identity"],
                 "failure_stage": row.get("failure_stage"),
                 "failure_type": row.get("failure_type")}
                for row in failed
            ],
            "metrics": metric_documents,
            "call_ledger": {
                category: {
                    field: sum(
                        int(row["call_ledger"][category][field]) for row in present
                    )
                    for field in (
                        "attempted_calls", "successful_calls", "failed_calls", "retry_calls"
                    )
                }
                for category in (
                    "target_construction_or_reuse",
                    "method_inference_or_training",
                    "terminal_recommendation",
                    "paired_assessment_and_metric",
                )
            },
            "timing": {
                phase: _timing_summary(present, phase)
                for phase in (
                    "method_inference_and_zero_call_decision_dispatch",
                    "paired_held_out_assessment",
                )
            },
        }
    pairs = []
    for left_index, left in enumerate(methods):
        for right_index, right in enumerate(methods[left_index + 1:], start=left_index + 1):
            complete_targets = [
                target for target in targets
                if (target, left) in slots and (target, right) in slots
                and slots[(target, left)]["status"] == slots[(target, right)]["status"] == "completed"
            ]
            excluded = [target for target in targets if target not in set(complete_targets)]
            excluded_details = []
            for target in excluded:
                detail: dict[str, Any] = {"target_record_identity": target, "reasons": []}
                for method in (left, right):
                    row = slots.get((target, method))
                    if row is None:
                        detail["reasons"].append({"method": method, "reason": "missing_record"})
                    elif row["status"] != "completed":
                        detail["reasons"].append({
                            "method": method,
                            "reason": "failed",
                            "failure_stage": row.get("failure_stage"),
                            "failure_type": row.get("failure_type"),
                        })
                excluded_details.append(detail)
            metric_documents = {}
            for metric_index, metric in enumerate(V3_METRIC_FIELDS):
                differences = [
                    float(slots[(target, left)][metric]) - float(slots[(target, right)][metric])
                    for target in complete_targets
                    if slots[(target, left)].get(metric) is not None and slots[(target, right)].get(metric) is not None
                ]
                wins = sum(value < 0 for value in differences)
                losses = sum(value > 0 for value in differences)
                ties = len(differences) - wins - losses
                metric_documents[metric] = {
                    **_descriptive(differences),
                    "paired_mean_bootstrap_95_percentile_interval": _bootstrap_mean_interval(
                        differences,
                        seed=bootstrap_seed + 7919 * left_index + 104729 * right_index + 9176 * metric_index,
                        resamples=bootstrap_resamples,
                    ),
                    "left_wins": wins,
                    "ties": ties,
                    "left_losses": losses,
                    "exact_two_sided_sign_test_p_value": _sign_test_two_sided(wins, losses),
                }
            pairs.append({
                "left": left,
                "right": right,
                "declared_target_count": len(targets),
                "complete_pair_count": len(complete_targets),
                "excluded_pair_count": len(excluded),
                "excluded_target_identities": excluded,
                "excluded_pairs": excluded_details,
                "metrics": metric_documents,
            })
    return {
        "schema_version": "mutaco-paper-v3-population-summary-v1",
        "record_schema": RECORD_SCHEMA_V3,
        "declared_target_count": len(targets),
        "bootstrap": {"seed": bootstrap_seed, "resamples": bootstrap_resamples, "interval": "percentile_95"},
        "multiplicity_policy": "not_selected_by_engineering",
        "methods": method_documents,
        "paired": pairs,
        "scientific_calls": 0,
    }

# Preserve historical pickle/global references while both imports share this module.
for _legacy_export in (
    PaperRecordSet,
    _validate_cohort,
    load_paper_records,
    select_methods,
    summarize_paired,
    _percentile,
    _descriptive,
    _bootstrap_mean_interval,
    _sign_test_two_sided,
    _timing_summary,
    summarize_population_v3,
):
    _legacy_export.__module__ = "experiments.paper.common.aggregate"
del _legacy_export
