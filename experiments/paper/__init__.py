"""Paper campaign interfaces, resolved only when requested."""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any


_EXPORTS = {
    'load_paper_records': ('analysis.aggregate', 'load_paper_records'),
    'select_methods': ('analysis.aggregate', 'select_methods'),
    'summarize_paired': ('analysis.aggregate', 'summarize_paired'),
    'summarize_population_v3': ('analysis.aggregate', 'summarize_population_v3'),
    'ExecutionSelection': ('protocol.campaign', 'ExecutionSelection'),
    'PaperCampaignDeclaration': ('protocol.campaign', 'PaperCampaignDeclaration'),
    'execute_paper_campaign': ('protocol.campaign', 'execute_paper_campaign'),
    'load_execution_selection': ('protocol.campaign', 'load_execution_selection'),
    'load_paper_campaign': ('protocol.campaign', 'load_paper_campaign'),
    'summarize_persisted_timings': ('analysis.timing_report', 'summarize_persisted_timings'),
}
_SUBMODULES = ('common',)

__all__ = [
    "ExecutionSelection",
    "PaperCampaignDeclaration",
    "execute_paper_campaign",
    "load_execution_selection",
    "load_paper_campaign",
    "load_paper_records",
    "select_methods",
    "summarize_paired",
    "summarize_population_v3",
    "summarize_persisted_timings",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module, symbol = _EXPORTS[name]
        value = getattr(_import_module("." + module, __name__), symbol)
    elif name in _SUBMODULES:
        value = _import_module("." + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Cache the actual object, preserving ordinary package monkeypatch behavior.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
