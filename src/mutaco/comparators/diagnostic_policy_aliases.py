"""Retained diagnostic selectors, separated from ordinary configuration imports."""

from mutaco.acquisition.candidate_proposal import RawCandidatePolicy
from mutaco.design.initial import InitialDesignPolicy
from mutaco.identity.sequential import (
    FIXED_FULL_STREAM_CRN_POLICY,
    PAIRED_EVALUATION_SEED_POLICY,
)
from mutaco.policies import ExplorationPolicy
from mutaco.surrogate import PosteriorNumericsPolicy


IID_UNIFORM_INITIAL_DESIGN = InitialDesignPolicy.IID_UNIFORM
IID_UNIFORM_CANDIDATES = RawCandidatePolicy.IID_UNIFORM
TRACE_MARGINAL_VARIANCE_DIAGNOSTIC = ExplorationPolicy.TRACE_MARGINAL_VARIANCE
NATIVE_FLOAT32_DIAGNOSTIC = PosteriorNumericsPolicy.NATIVE_FLOAT32
FULL_FLOAT64_DIAGNOSTIC = PosteriorNumericsPolicy.FULL_FLOAT64_DIAGNOSTIC
TRACE_EFFECTIVE_DIMENSION_RADIUS_DIAGNOSTIC = (
    "trace_effective_dimension_radius_diagnostic_v1"
)
MINIMUM_NORMALIZED_TARGET_MISS_DIAGNOSTIC = (
    "minimum_normalized_target_miss_diagnostic_v1"
)
PAIRED_COORDINATE_DIAGNOSTIC = PAIRED_EVALUATION_SEED_POLICY
FIXED_FULL_STREAM_CRN_DIAGNOSTIC = FIXED_FULL_STREAM_CRN_POLICY
