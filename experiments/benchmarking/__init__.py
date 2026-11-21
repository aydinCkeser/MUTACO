"""Public simulator-agnostic benchmark interfaces, loaded on first use."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BenchmarkResultSet",
    "export_tables",
    "load_campaign",
    "load_results",
    "plot_results",
    "run_campaign",
    "summarize_results",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name in {"load_campaign", "run_campaign"}:
        from . import api as owner
    else:
        from . import results as owner
    return getattr(owner, name)
