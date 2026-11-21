"""Typed initial-design policies."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import initial_design
from .initial_design import (
    INITIAL_DESIGN_POLICIES,
    InitialDesignPolicy,
    InitialDesignResult,
    generate_initial_design,
    initial_design_support_markdown,
    validate_caller_design,
)

initial = _bind_legacy_module(initial_design, __name__ + ".initial")

__all__ = [
    "INITIAL_DESIGN_POLICIES",
    "InitialDesignPolicy",
    "InitialDesignResult",
    "generate_initial_design",
    "initial_design_support_markdown",
    "validate_caller_design",
]
