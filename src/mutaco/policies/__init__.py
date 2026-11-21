"""Typed D mathematical policy registry."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import policy_surface

# The unchanged policy registry imports the historical surface module.
surface = _bind_legacy_module(policy_surface, __name__ + ".surface")

from . import scientific_policy_registry
from .scientific_policy_registry import (
    BETA_POLICIES,
    EXPLORATION_POLICIES,
    RECOMMENDATION_POLICIES,
    BetaPolicy,
    ExplorationPolicy,
    MathematicsPolicyConfig,
    OptimizationDirection,
    PolicyClassification,
    PolicyDescriptor,
    RecommendationPolicy,
    descriptor_for,
    render_policy_support_markdown,
)
from .policy_surface import (
    POLICY_SURFACE,
    PolicySurface,
    PolicySurfaceRecord,
    policy_surface_document,
    render_policy_surface_markdown,
    surface_record,
)

registry = _bind_legacy_module(scientific_policy_registry, __name__ + ".registry")

__all__ = [
    "BETA_POLICIES",
    "BetaPolicy",
    "EXPLORATION_POLICIES",
    "RECOMMENDATION_POLICIES",
    "ExplorationPolicy",
    "MathematicsPolicyConfig",
    "OptimizationDirection",
    "PolicyClassification",
    "PolicyDescriptor",
    "POLICY_SURFACE",
    "PolicySurface",
    "PolicySurfaceRecord",
    "RecommendationPolicy",
    "descriptor_for",
    "render_policy_support_markdown",
    "policy_surface_document",
    "render_policy_surface_markdown",
    "surface_record",
]
