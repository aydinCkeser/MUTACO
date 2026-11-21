"""Predeclared Prompt 7 parity cohorts and fail-closed metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


FIXED_INTERMEDIATE_RTOL = 2e-5
FIXED_INTERMEDIATE_ATOL = 2e-6
FINAL_RESPONSE_RTOL = 5e-5
FINAL_RESPONSE_ATOL = 5e-6
FINAL_RESPONSE_RELATIVE_L2 = 5e-5
STATISTICAL_COVARIANCE_RELATIVE_FROBENIUS = 0.15

# Center, faces/corners, then deterministic Sobol points mapped into P=4 bounds.
SCIENTIFIC_FIXED_THETA_P4: tuple[tuple[float, float, float, float], ...] = (
    (1.25, 1.25, 1.25, 0.50),
    (0.50, 1.25, 1.25, 0.50),
    (2.00, 1.25, 1.25, 0.50),
    (1.25, 0.50, 1.25, 0.50),
    (1.25, 2.00, 1.25, 0.50),
    (1.25, 1.25, 0.50, 0.50),
    (1.25, 1.25, 2.00, 0.50),
    (1.25, 1.25, 1.25, 0.00),
    (1.25, 1.25, 1.25, 1.00),
    (0.50, 0.50, 0.50, 0.00),
    (2.00, 2.00, 2.00, 1.00),
    (1.625, 0.875, 1.625, 0.25),
    (0.875, 1.625, 0.875, 0.75),
    (1.0625, 1.0625, 1.8125, 0.625),
    (1.8125, 1.8125, 1.0625, 0.125),
    (0.6875, 1.4375, 1.4375, 0.875),
)
RETAINED_FIXED_THETA_P2: tuple[tuple[float, float], ...] = tuple(
    (row[1], row[3]) for row in SCIENTIFIC_FIXED_THETA_P4
)
STATISTICAL_THETA_P4 = SCIENTIFIC_FIXED_THETA_P4[:8]
FIXED_SEEDS = tuple(73007 + 104729 * index for index in range(16))
STATISTICAL_SEEDS = tuple(91009 + 130363 * index for index in range(16))


@dataclass(frozen=True)
class DifferenceMetrics:
    max_absolute: float
    quantile_50: float
    quantile_95: float
    quantile_99: float
    relative_l2: float
    allclose: bool


def difference_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> DifferenceMetrics:
    if reference.shape != candidate.shape:
        raise ValueError("parity tensors have different shapes")
    reference_cpu = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate_cpu = candidate.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(reference_cpu).all() and torch.isfinite(candidate_cpu).all()):
        raise ValueError("parity tensors must be finite")
    delta = torch.abs(candidate_cpu - reference_cpu).reshape(-1)
    denominator = max(float(torch.linalg.vector_norm(reference_cpu)), 1e-30)
    return DifferenceMetrics(
        max_absolute=float(delta.max()) if delta.numel() else 0.0,
        quantile_50=float(torch.quantile(delta, 0.50)) if delta.numel() else 0.0,
        quantile_95=float(torch.quantile(delta, 0.95)) if delta.numel() else 0.0,
        quantile_99=float(torch.quantile(delta, 0.99)) if delta.numel() else 0.0,
        relative_l2=float(torch.linalg.vector_norm(candidate_cpu - reference_cpu)) / denominator,
        allclose=bool(torch.allclose(candidate_cpu, reference_cpu, rtol=rtol, atol=atol)),
    )


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 7007,
) -> tuple[np.ndarray, np.ndarray]:
    if differences.ndim != 2 or differences.shape[0] < 2:
        raise ValueError("paired bootstrap differences must be [seed, component]")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.shape[0], size=(bootstrap_samples, differences.shape[0]))
    means = differences[indices].mean(axis=1)
    return np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def statistical_gate(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Mapping[str, object]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("statistical parity arrays must share [seed,component] shape")
    low, high = paired_bootstrap_interval(candidate - reference)
    pooled_sd = np.sqrt(
        0.5 * (reference.var(axis=0, ddof=1) + candidate.var(axis=0, ddof=1))
    )
    margin = np.maximum(0.01, 0.10 * pooled_sd)
    reference_cov = np.cov(reference, rowvar=False)
    candidate_cov = np.cov(candidate, rowvar=False)
    denominator = max(float(np.linalg.norm(reference_cov, ord="fro")), 1e-30)
    covariance_relative = float(
        np.linalg.norm(candidate_cov - reference_cov, ord="fro") / denominator
    )
    component_pass = (low >= -margin) & (high <= margin)
    return {
        "bootstrap_low": low.tolist(),
        "bootstrap_high": high.tolist(),
        "margin": margin.tolist(),
        "component_pass": component_pass.tolist(),
        "covariance_relative_frobenius": covariance_relative,
        "covariance_pass": covariance_relative
        <= STATISTICAL_COVARIANCE_RELATIVE_FROBENIUS,
        "passed": bool(
            component_pass.all()
            and covariance_relative <= STATISTICAL_COVARIANCE_RELATIVE_FROBENIUS
        ),
    }
