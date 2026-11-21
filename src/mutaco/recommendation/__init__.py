"""Terminal recommendations on the continuous domain or evaluated designs."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import evaluated_design_recommendation
from .evaluated_design_recommendation import (
    EvaluatedDesigns,
    RecommendationResult,
    recommend_evaluated,
)
from .comparators import physical_squared_l2_recommendation
from .comparators.physical_squared_l2_recommendation import (
    CONTINUOUS_PHYSICAL_L2,
    CONTINUOUS_PHYSICAL_L2_SETTINGS,
    EVALUATED_PHYSICAL_L2,
    MeanOnlyPosterior,
    PhysicalL2RecommendationError,
    PhysicalL2RecommendationResult,
    physical_l2_scores,
    recommend_continuous_physical_l2,
    recommend_evaluated_physical_l2,
)
from . import _continuous_terminal_optimizer

policies = _bind_legacy_module(evaluated_design_recommendation, __name__ + ".policies")
physical_l2 = _bind_legacy_module(physical_squared_l2_recommendation, __name__ + ".physical_l2")
# Exact terminal overlays substitute the old optimizer name before reloading MAE.
_continuous = _bind_legacy_module(_continuous_terminal_optimizer, __name__ + "._continuous")

from . import physical_mae_terminal_recommendation
from .physical_mae_terminal_recommendation import (
    CONTINUOUS_PHYSICAL_MAE,
    CONTINUOUS_PHYSICAL_MAE_SETTINGS,
    PhysicalMAERecommendationError,
    PhysicalMAERecommendationResult,
    physical_mae_scores,
    recommend_continuous_physical_mae,
)
from mutaco.comparators import observed_physical_mae_recommendation as _observed
from mutaco.comparators.observed_physical_mae_recommendation import (
    EVALUATED_BEST_OBSERVED_PHYSICAL_MAE,
    ObservedPhysicalMAERecommendationResult,
    recommend_best_observed_physical_mae,
)

physical_mae = _bind_legacy_module(physical_mae_terminal_recommendation, __name__ + ".physical_mae")
observed_physical_mae = _bind_legacy_module(_observed, __name__ + ".observed_physical_mae")

__all__ = [
    "CONTINUOUS_PHYSICAL_MAE", "CONTINUOUS_PHYSICAL_MAE_SETTINGS",
    "CONTINUOUS_PHYSICAL_L2", "CONTINUOUS_PHYSICAL_L2_SETTINGS",
    "EVALUATED_PHYSICAL_L2", "EvaluatedDesigns", "MeanOnlyPosterior",
    "EVALUATED_BEST_OBSERVED_PHYSICAL_MAE",
    "ObservedPhysicalMAERecommendationResult",
    "PhysicalL2RecommendationError", "PhysicalL2RecommendationResult",
    "PhysicalMAERecommendationError", "PhysicalMAERecommendationResult",
    "RecommendationResult", "physical_l2_scores", "physical_mae_scores",
    "recommend_continuous_physical_l2", "recommend_continuous_physical_mae",
    "recommend_evaluated", "recommend_evaluated_physical_l2",
    "recommend_best_observed_physical_mae",
]
