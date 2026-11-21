"""Benchmark method adapters, resolved lazily for method-family isolation."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AMORTIZED_METHODS",
    "POSTERIOR_SAMPLE_MAP_POLICY",
    "PosteriorSampleMapResult",
    "posterior_sample_map_decision",
    "run_amortized_bank",
    "run_mutaco_target",
    "run_sbi_target_process",
]


def __getattr__(name: str) -> Any:
    if name == "run_mutaco_target":
        from .mutaco_reference import run_mutaco_target

        return run_mutaco_target
    if name == "run_sbi_target_process":
        from .process import run_sbi_target_process

        return run_sbi_target_process
    if name in {"AMORTIZED_METHODS", "run_amortized_bank"}:
        from . import sbi_amortized

        return getattr(sbi_amortized, name)
    if name in {
        "POSTERIOR_SAMPLE_MAP_POLICY",
        "PosteriorSampleMapResult",
        "posterior_sample_map_decision",
    }:
        from . import direct_posterior_decision

        return getattr(direct_posterior_decision, name)
    raise AttributeError(name)
