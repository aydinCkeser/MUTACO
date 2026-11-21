"""Exact multi-output GP surrogate boundary."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import exact_multitask_gp
from .exact_multitask_gp import (
    GPFitConfig,
    GPFitDiagnostic,
    GPFitFailureKind,
    GPFitResult,
    GPFitRecoveryPolicy,
    GP_FIT_RECOVERY_POLICIES,
    GRAPH_M20_RESTART_COUNT,
    GRAPH_M20_TASK_VARIANCE_EPSILON_MULTIPLIER,
    GRAPH_M20_TASK_VARIANCE_FLOOR,
    GRAPH_M20_TASK_VARIANCE_REFERENCE_SCALE,
    GPFitStatus,
    GPWarmStartState,
    LIKELIHOOD_POLICIES,
    ExactMultiTaskGPSurrogate,
    LikelihoodPolicy,
    ObservationPosteriorSummary,
    POSTERIOR_NUMERICS_POLICIES,
    PosteriorNumericsPolicy,
    SurrogateTrainingData,
    TaskVarianceFloorPolicy,
    fit_exact_multitask_gp,
    fit_exact_multitask_gp_with_recovery,
    gp_fit_recovery_support_markdown,
    likelihood_support_markdown,
    posterior_numerics_support_markdown,
)

exact_multitask = _bind_legacy_module(exact_multitask_gp, __name__ + ".exact_multitask")

__all__ = [
    "GPFitConfig",
    "GPFitDiagnostic",
    "GPFitFailureKind",
    "GPFitResult",
    "GPFitRecoveryPolicy",
    "GP_FIT_RECOVERY_POLICIES",
    "GRAPH_M20_RESTART_COUNT",
    "GRAPH_M20_TASK_VARIANCE_EPSILON_MULTIPLIER",
    "GRAPH_M20_TASK_VARIANCE_FLOOR",
    "GRAPH_M20_TASK_VARIANCE_REFERENCE_SCALE",
    "GPFitStatus",
    "GPWarmStartState",
    "LIKELIHOOD_POLICIES",
    "ExactMultiTaskGPSurrogate",
    "LikelihoodPolicy",
    "ObservationPosteriorSummary",
    "POSTERIOR_NUMERICS_POLICIES",
    "PosteriorNumericsPolicy",
    "SurrogateTrainingData",
    "TaskVarianceFloorPolicy",
    "fit_exact_multitask_gp",
    "fit_exact_multitask_gp_with_recovery",
    "gp_fit_recovery_support_markdown",
    "likelihood_support_markdown",
    "posterior_numerics_support_markdown",
]
