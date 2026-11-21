"""Exact cohort metric and box-statistic definitions."""

from __future__ import annotations

from typing import Any, Iterable

import torch


def vector_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    residual = actual - expected
    return {
        "absolute_per_component": residual.abs().detach().cpu().tolist(),
        "l1": float(torch.linalg.vector_norm(residual, ord=1).detach().cpu()),
        "l2": float(torch.linalg.vector_norm(residual, ord=2).detach().cpu()),
        "linf": float(residual.abs().max().detach().cpu()),
    }


def summarize(values: Iterable[float]) -> dict[str, Any]:
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
        "sample_std": (
            float(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0
        ),
        "p90": p90,
        "p95": p95,
        "whisker_low": float(inside.min()),
        "whisker_high": float(inside.max()),
        "outliers": tensor[(tensor < low_fence) | (tensor > high_fence)].tolist(),
        "quantile_interpolation": "linear",
        "whisker_rule": "1.5_iqr",
    }


PRIMARY_METRICS = ("target_error_l1", "target_error_l2", "design_error_l2")
LATENT_METRICS = ("latent_error_l1", "latent_error_l2")


def aggregate(rows: list[dict[str, Any]], target_count: int) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    return {
        "target_count": target_count,
        "completed_count": len(completed),
        "failed_count": target_count - len(completed),
        "failure_rate": (target_count - len(completed)) / target_count,
        "metrics": {
            name: summarize(row[name] for row in completed)
            for name in PRIMARY_METRICS + LATENT_METRICS
        },
    }
