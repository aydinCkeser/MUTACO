"""Simulator-agnostic summaries over persisted method-result records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch


METRIC_FIELDS = (
    "target_error_l1",
    "target_error_l2",
    "design_error_l2",
    "latent_error_l1",
    "latent_error_l2",
)


def vector_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Describe component residuals and unnormalized L1, L2 and L-infinity norms.

    Inputs must already share coordinate conventions and order. L1 is the sum of
    absolute errors, not mean absolute error; L2 is the Euclidean norm, not its
    square. These stored benchmark fields must not be confused with the core's
    continuous physical-MAE recommendation objective. No scale conversion occurs.
    """
    residual = actual - expected
    return {
        "absolute_per_component": residual.abs().detach().cpu().tolist(),
        "l1": float(torch.linalg.vector_norm(residual, ord=1).detach().cpu()),
        "l2": float(torch.linalg.vector_norm(residual, ord=2).detach().cpu()),
        "linf": float(residual.abs().max().detach().cpu()),
    }


def summarize(values: Iterable[float]) -> dict[str, Any]:
    """Summarize persisted scalar errors in CPU float64 without rerunning a model.

    Quartiles/percentiles use linear interpolation. Whiskers are the outer observed
    values within inclusive 1.5-IQR fences, not the fences themselves. sample_std
    uses N-1 for N>1, with the existing singleton convention of zero; an empty
    population yields only count=0. Outliers retain their incoming order.
    """
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        tensor, torch.tensor([0.25, 0.5, 0.75, 0.9, 0.95], dtype=tensor.dtype)
    )
    q1, median, q3, p90, p95 = (float(item) for item in quantiles)
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = tensor[(tensor >= low_fence) & (tensor <= high_fence)]
    return {
        "count": int(tensor.numel()),
        "minimum": float(tensor.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "maximum": float(tensor.max()),
        "iqr": iqr,
        "mean": float(tensor.mean()),
        "sample_std": float(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0,
        "p90": p90,
        "p95": p95,
        "whisker_low": float(inside.min()),
        "whisker_high": float(inside.max()),
        "outliers": tensor[(tensor < low_fence) | (tensor > high_fence)].tolist(),
        "quantile_interpolation": "linear",
        "whisker_rule": "1.5_iqr",
    }


def aggregate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = METRIC_FIELDS,
    group_by: Sequence[str] = ("method",),
) -> dict[str, Any]:
    """Group persisted target records, separating completion from scientific claims.

    Metric summaries use nonmissing values from completed rows only; failures
    remain explicit in target counts and failure rates, not numerical penalties.
    Two or more initial-GP-fit failures flag invalid aggregate MUTACO claims, both
    within a group and across the campaign. Group order follows first occurrence;
    metric definitions and stored responses are not recomputed or standardized.
    """
    materialized = tuple(dict(row) for row in rows)
    unknown = set(metrics) - set(METRIC_FIELDS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        grouped[tuple(row.get(field) for field in group_by)].append(row)
    groups = []

    def initial_gp_failure(row: Mapping[str, Any]) -> bool:
        runner = row.get("runner_provenance")
        runner_status = runner.get("status") if isinstance(runner, Mapping) else None
        return bool(
            row.get("status") == "initial_gp_fit_failure"
            or row.get("failure_kind") == "initial_gp_fit_failure"
            or runner_status == "initial_gp_fit_failure"
            or "initial_gp_fit_failure" in str(row.get("failure_explanation", ""))
        )

    for key, group in grouped.items():
        completed = [row for row in group if row.get("status") == "completed"]
        initial_failures = sum(initial_gp_failure(row) for row in group)
        recurrent_initial_failure = initial_failures >= 2
        groups.append(
            {
                "group": dict(zip(group_by, key)),
                "target_count": len(group),
                "completed_count": len(completed),
                "failed_count": len(group) - len(completed),
                "failure_rate": (len(group) - len(completed)) / len(group),
                "initial_gp_fit_failure_count": initial_failures,
                "aggregate_claims_valid": not recurrent_initial_failure,
                "campaign_quality_warning": (
                    "recurrent_initial_gp_fit_failure_invalidates_aggregate_mutaco_claims"
                    if recurrent_initial_failure
                    else None
                ),
                "metrics": {
                    metric: summarize(
                        float(row[metric])
                        for row in completed
                        if row.get(metric) is not None
                    )
                    for metric in metrics
                },
            }
        )
    total_initial_failures = sum(initial_gp_failure(row) for row in materialized)
    return {
        "schema_version": "mutaco-benchmark-summary-v1",
        "group_by": tuple(group_by),
        "metrics": tuple(metrics),
        "groups": groups,
        "campaign_quality": {
            "initial_gp_fit_failure_count": total_initial_failures,
            "recurrent_initial_gp_fit_failure": total_initial_failures >= 2,
            "aggregate_mutaco_claims_valid": total_initial_failures < 2,
            "warning": (
                "recurrent_initial_gp_fit_failure_invalidates_aggregate_mutaco_claims"
                if total_initial_failures >= 2
                else None
            ),
        },
    }
