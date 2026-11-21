"""Torch-native reference simulators for the clean MUTACO project."""

if __name__ == "experiments.analytic_target_matching.forward_models":
    from importlib import import_module
    import sys

    package = import_module("mutaco_analytic")
    for suffix in ("", ".gp_landscape", ".handwritten"):
        sys.modules[__name__ + suffix] = sys.modules[package.__name__ + suffix]
else:
    from .gp_landscape import (
        FrozenGPLandscapeConfig,
        FrozenGPLandscapeSimulator,
        FrozenGPLandscapeState,
    )
    from .handwritten import (
        DEFAULT_NOISE_STD,
        THETA_STAR,
        HandwrittenAnalyticSimulator,
        analytic_response,
    )

    __all__ = [
        "DEFAULT_NOISE_STD",
        "THETA_STAR",
        "FrozenGPLandscapeConfig",
        "FrozenGPLandscapeSimulator",
        "FrozenGPLandscapeState",
        "HandwrittenAnalyticSimulator",
        "analytic_response",
    ]
