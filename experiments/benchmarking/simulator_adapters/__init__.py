"""Simulator adapter registry."""

from __future__ import annotations

from importlib import import_module
import sys
from typing import Any

from . import gp_landscape, handwritten, linear
from .gp_landscape import FrozenGPLandscapeAdapter
from .handwritten import HandwrittenAnalyticAdapter
from .linear import LinearDiagnosticAdapter


# This historical file is fingerprinted in the terminal-domain declarations.
graph_scientific_d4 = import_module("experiments.benchmarking.simulators.graph_scientific_d4")
GraphScientificD4Adapter = graph_scientific_d4.GraphScientificD4Adapter


def get_simulator_adapter(identifier: str) -> Any:
    registry = {
        "graph_scientific_d4": GraphScientificD4Adapter,
        "handwritten": HandwrittenAnalyticAdapter,
        "handwritten_analytic": HandwrittenAnalyticAdapter,
        "linear_diagnostic": LinearDiagnosticAdapter,
        "analytic_gp_landscape": FrozenGPLandscapeAdapter,
    }
    try:
        return registry[identifier]()
    except KeyError as exc:
        raise ValueError(f"unknown simulator adapter: {identifier!r}") from exc


__all__ = [
    "GraphScientificD4Adapter",
    "FrozenGPLandscapeAdapter",
    "HandwrittenAnalyticAdapter",
    "LinearDiagnosticAdapter",
    "get_simulator_adapter",
]

# The v1/v2 held-out declarations hash the original graph adapter file. Both
# registry addresses must share the same adapter classes and module globals.
sys.modules[__name__ + ".graph_scientific_d4"] = graph_scientific_d4
for _module in (gp_landscape, handwritten, linear):
    sys.modules["experiments.benchmarking.simulators." + _module.__name__.rsplit(".", 1)[1]] = _module
del _module

get_simulator_adapter.__module__ = "experiments.benchmarking.simulators"
