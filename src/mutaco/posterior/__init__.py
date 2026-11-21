"""Surrogate-independent latent posterior summaries."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import posterior_summary
from .posterior_summary import (
    CovarianceValidationPolicy,
    InvalidPosteriorError,
    JointLatentPosteriorSummary,
    LatentPosteriorMeanSummary,
    LatentPosteriorSummary,
    structurally_symmetric_materialize,
)

summary = _bind_legacy_module(posterior_summary, __name__ + ".summary")

__all__ = [
    "CovarianceValidationPolicy",
    "InvalidPosteriorError",
    "JointLatentPosteriorSummary",
    "LatentPosteriorMeanSummary",
    "LatentPosteriorSummary",
    "structurally_symmetric_materialize",
]
