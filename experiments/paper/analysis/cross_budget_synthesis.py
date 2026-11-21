"""Deterministic scalar-primary B40/B100 scientific synthesis."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from statistics import mean, median, stdev
from typing import Any, Mapping, Sequence

from mutaco import stable_hash

from ..common.evidence_analysis import ROW_SCHEMA, validate_analysis_rows
from ..common.validation import PaperContractError


SPEC_SCHEMA = "mutaco-scalar-primary-cross-budget-synthesis-spec-v1"
ROW_SCHEMA_CROSS_BUDGET = "mutaco-scalar-primary-cross-budget-row-v1"
REPORT_SCHEMA = "mutaco-scalar-primary-cross-budget-report-v1"
SPEC_IDENTITY_DOMAIN = "scalar-primary-cross-budget-synthesis-spec-v1"
ROW_IDENTITY_DOMAIN = "scalar-primary-cross-budget-row-v1"
REPORT_IDENTITY_DOMAIN = "scalar-primary-cross-budget-report-v1"
METHODS = ("mutaco", "scalar_qlognei", "sobol")
BUDGETS = (40, 100)
PRIMARY_ERROR = "heldout_assessment_error"


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
        raise PaperContractError("cross-budget summary cannot be empty")
    return {
        "n": len(values),
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
        "minimum": min(values),
        "maximum": max(values),
    }


def _bootstrap_interval(values: Sequence[float], *, seed: int, resamples: int) -> list[float]:
    generator = random.Random(seed)
    n = len(values)
    draws = [mean([values[generator.randrange(n)] for _ in range(n)]) for _ in range(resamples)]
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def validate_synthesis_specification(specification: Mapping[str, Any]) -> None:
    if specification.get("schema_version") != SPEC_SCHEMA:
        raise PaperContractError("cross-budget synthesis specification schema mismatch")
    body = dict(specification)
    identity = body.pop("synthesis_spec_identity", None)
    if identity != stable_hash(SPEC_IDENTITY_DOMAIN, body):
        raise PaperContractError("cross-budget synthesis specification identity mismatch")
    if tuple(specification.get("budgets", ())) != BUDGETS:
        raise PaperContractError("cross-budget synthesis requires exactly B40 and B100")
    if tuple(specification.get("methods", ())) != METHODS:
        raise PaperContractError("cross-budget synthesis method order mismatch")
    if specification.get("source_row_schema") != ROW_SCHEMA:
        raise PaperContractError("cross-budget synthesis source row schema mismatch")
    if specification.get("primary_metric") != f"{PRIMARY_ERROR}.component_mean_mae":
        raise PaperContractError("cross-budget synthesis primary metric mismatch")
    if specification.get("campaign_relation") != "independent_campaigns_same_ordered_target_cohort_not_nested":
        raise PaperContractError("cross-budget campaign relation is not frozen")


def _validate_source_rows(rows_by_budget: Mapping[int, Sequence[Mapping[str, Any]]]) -> None:
    if set(rows_by_budget) != set(BUDGETS):
        raise PaperContractError("cross-budget source rows require both frozen budgets")
    target_ids: dict[int, tuple[str, ...]] = {}
    for budget in BUDGETS:
        rows = rows_by_budget[budget]
        ids = tuple(
            str(row["target_id"])
            for row in rows
            if row["method"] == METHODS[0]
        )
        validate_analysis_rows(
            rows,
            expected_methods=METHODS,
            expected_budget=budget,
            expected_targets=ids,
        )
        if len(ids) != 120 or len(set(ids)) != 120:
            raise PaperContractError("cross-budget source cohort is not the ordered Paper-120 population")
        target_ids[budget] = ids
    if target_ids[40] != target_ids[100]:
        raise PaperContractError("B40 and B100 target identity/order differs")


def build_cross_budget_rows(
    rows_by_budget: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    """Build one hashed B40/B100 endpoint row per target and method."""

    _validate_source_rows(rows_by_budget)
    indexed: dict[int, dict[tuple[int, str], Mapping[str, Any]]] = {}
    for budget in BUDGETS:
        indexed[budget] = {
            (int(row["target_position"]), str(row["method"])): row
            for row in rows_by_budget[budget]
        }
    result = []
    for position in range(120):
        for method in METHODS:
            b40 = indexed[40][position, method]
            b100 = indexed[100][position, method]
            if b40["target_id"] != b100["target_id"]:
                raise PaperContractError("cross-budget target pairing mismatch")
            values = {}
            for budget, source in ((40, b40), (100, b100)):
                error = source["errors"][PRIMARY_ERROR]
                if not error["available"]:
                    raise PaperContractError("cross-budget primary endpoint is unavailable")
                values[str(budget)] = {
                    "component_mean_mae": float(error["metric"]["component_mean_mae"]),
                    "response_l1_sum": float(error["metric"]["response_l1_sum"]),
                    "source_row_identity": source["row_identity"],
                    "assessment_context": source["assessment"]["context"],
                }
            delta = values["100"]["component_mean_mae"] - values["40"]["component_mean_mae"]
            body = {
                "schema_version": ROW_SCHEMA_CROSS_BUDGET,
                "target_position": position,
                "target_id": b40["target_id"],
                "method": method,
                "budget_endpoints": values,
                "b100_minus_b40_component_mean_mae": delta,
                "direction": "lower_at_b100" if delta < 0 else "equal" if delta == 0 else "higher_at_b100",
                "target_blocking": "same_target_identity",
                "campaign_relation": "independent_campaigns_not_nested",
            }
            result.append({**body, "row_identity": stable_hash(ROW_IDENTITY_DOMAIN, body)})
    return tuple(result)


def deterministic_cross_budget_synthesis(
    rows_by_budget: Mapping[int, Sequence[Mapping[str, Any]]],
    reports_by_budget: Mapping[int, Mapping[str, Any]],
    source_manifests: Mapping[int, Mapping[str, Any]],
    specification: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Synthesize frozen budget reports without changing their native evidence."""

    validate_synthesis_specification(specification)
    rows = build_cross_budget_rows(rows_by_budget)
    if set(reports_by_budget) != set(BUDGETS) or set(source_manifests) != set(BUDGETS):
        raise PaperContractError("cross-budget synthesis input package set is incomplete")
    for budget in BUDGETS:
        report = reports_by_budget[budget]
        manifest = source_manifests[budget]
        if report.get("budgets") != [budget] or int(manifest.get("budget")) != budget:
            raise PaperContractError("cross-budget report or manifest budget mismatch")
        if manifest.get("evidence_identity") != specification["source_evidence_identities"][str(budget)]:
            raise PaperContractError("cross-budget source evidence identity mismatch")
        if report.get("analysis_report_identity") != specification["source_analysis_report_identities"][str(budget)]:
            raise PaperContractError("cross-budget source analysis report identity mismatch")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    bootstrap = specification["bootstrap"]
    changes = {}
    for method in METHODS:
        deltas = [float(row["b100_minus_b40_component_mean_mae"]) for row in grouped[method]]
        derived_seed = int(
            stable_hash(
                "scalar-primary-cross-budget-bootstrap-seed-v1",
                {"root_seed": bootstrap["seed"], "method": method},
            ).rsplit("-", 1)[-1][:16],
            16,
        )
        changes[method] = {
            "summary_b100_minus_b40": _summary(deltas),
            "target_counts": {
                "lower_at_b100": sum(value < 0 for value in deltas),
                "equal": sum(value == 0 for value in deltas),
                "higher_at_b100": sum(value > 0 for value in deltas),
            },
            "target_blocked_mean_bootstrap_95_percentile_interval": _bootstrap_interval(
                deltas,
                seed=derived_seed,
                resamples=int(bootstrap["resamples"]),
            ),
            "derived_seed": derived_seed,
            "interpretation": "descriptive target-blocked change across independent campaigns; not a nested trajectory",
        }

    endpoint_by_budget = {
        str(budget): {
            "heldout_component_mean_mae": {
                method: reports_by_budget[budget]["error_summaries_component_mean_mae"][method][PRIMARY_ERROR]
                for method in METHODS
            },
            "within_budget_paired_primary_endpoint": reports_by_budget[budget]["paired_primary_endpoint"],
            "target_winners": reports_by_budget[budget]["target_winners"],
            "target_winner_ties": reports_by_budget[budget]["target_winner_ties"],
            "mean_rank": reports_by_budget[budget]["mean_rank"],
        }
        for budget in BUDGETS
    }
    terminal_by_budget = {
        str(budget): {
            method: {
                key: reports_by_budget[budget]["error_summaries_component_mean_mae"][method][key]
                for key in ("raw_terminal_surrogate_error", "optimized_terminal_surrogate_error")
            }
            for method in METHODS
        }
        for budget in BUDGETS
    }
    terminal_refinement_effect = {}
    for budget in BUDGETS:
        terminal_refinement_effect[str(budget)] = {}
        for method in METHODS:
            method_rows = [row for row in rows_by_budget[budget] if row["method"] == method]
            paired_values = []
            for row in method_rows:
                raw = row["errors"]["raw_terminal_surrogate_error"]
                optimized = row["errors"]["optimized_terminal_surrogate_error"]
                if raw["available"] and optimized["available"]:
                    paired_values.append(
                        float(optimized["metric"]["component_mean_mae"])
                        - float(raw["metric"]["component_mean_mae"])
                    )
            terminal_refinement_effect[str(budget)][method] = (
                {
                    "available": True,
                    "metric": "optimized_minus_raw_terminal_surrogate_component_mean_mae",
                    "summary": _summary(paired_values),
                    "improved": sum(value < 0 for value in paired_values),
                    "equal": sum(value == 0 for value in paired_values),
                    "worsened": sum(value > 0 for value in paired_values),
                    "interpretation": "secondary surrogate-space terminal-refinement diagnostic",
                }
                if len(paired_values) == len(method_rows)
                else {
                    "available": False,
                    "reason": "native evidence does not provide paired raw and optimized terminal-surrogate errors for every target",
                }
            )
    body = {
        "schema_version": REPORT_SCHEMA,
        "synthesis_spec_identity": specification["synthesis_spec_identity"],
        "source_evidence_identities": specification["source_evidence_identities"],
        "source_analysis_report_identities": specification["source_analysis_report_identities"],
        "budgets": list(BUDGETS),
        "methods": list(METHODS),
        "target_count": 120,
        "row_count": len(rows),
        "primary_endpoint_by_budget": endpoint_by_budget,
        "descriptive_target_blocked_cross_budget_change": changes,
        "terminal_surrogate_summaries_by_budget": terminal_by_budget,
        "terminal_refinement_effect_by_budget": terminal_refinement_effect,
        "descriptive_symmetric_timing_by_budget": {
            str(budget): reports_by_budget[budget]["descriptive_timing_seconds"]
            for budget in BUDGETS
        },
        "scalar_qlognei_numerical_diagnostics_by_budget": {
            str(budget): source_manifests[budget]["scalar_qlognei_diagnostics"]
            for budget in BUDGETS
        },
        "claim_ledger": {
            "confirmatory": [
                "Within each frozen budget, compare methods using paired held-out physical-response component-mean MAE.",
                "Within-budget paired bootstrap intervals, wins, and ranks are supported by the complete Paper-120 grid.",
            ],
            "descriptive": [
                "B100-minus-B40 target-blocked changes summarize independent campaigns on the same targets.",
                "Persisted symmetric decision-time components and numerical fallbacks describe these executions only.",
                "Search and terminal-surrogate errors are secondary mechanism diagnostics, not held-out endpoints.",
            ],
            "prohibited": [
                "Do not interpret B100 as a continuation of B40 or claim per-run monotonic improvement.",
                "Do not pool retired vector-NEI evidence or repaired MUTACO-SBI evidence into this comparison.",
                "Do not infer unavailable metrics, production compute-efficiency, or mechanism causality.",
            ],
        },
        "campaign_relation": "independent_campaigns_same_ordered_target_cohort_not_nested",
        "historical_evidence_modified": False,
        "scientific_calls_added_by_synthesis": 0,
        "truba_jobs_submitted": 0,
    }
    report = {**body, "synthesis_report_identity": stable_hash(REPORT_IDENTITY_DOMAIN, body)}
    return rows, report

# Preserve historical pickle/global references while both imports share this module.
for _legacy_export in (
    _quantile,
    _summary,
    _bootstrap_interval,
    validate_synthesis_specification,
    _validate_source_rows,
    build_cross_budget_rows,
    deterministic_cross_budget_synthesis,
):
    _legacy_export.__module__ = "experiments.paper.common.cross_budget_synthesis"
del _legacy_export
