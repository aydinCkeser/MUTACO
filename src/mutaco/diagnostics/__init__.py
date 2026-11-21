"""Explicit imports for retained diagnostic MUTACO policies."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module
from mutaco.comparators import diagnostic_policy_aliases as _policy_aliases
from mutaco.comparators.diagnostic_policy_aliases import (
    FIXED_FULL_STREAM_CRN_DIAGNOSTIC,
    FULL_FLOAT64_DIAGNOSTIC,
    IID_UNIFORM_CANDIDATES,
    IID_UNIFORM_INITIAL_DESIGN,
    MINIMUM_NORMALIZED_TARGET_MISS_DIAGNOSTIC,
    NATIVE_FLOAT32_DIAGNOSTIC,
    PAIRED_COORDINATE_DIAGNOSTIC,
    TRACE_EFFECTIVE_DIMENSION_RADIUS_DIAGNOSTIC,
    TRACE_MARGINAL_VARIANCE_DIAGNOSTIC,
)

policies = _bind_legacy_module(_policy_aliases, __name__ + ".policies")

__all__ = [name for name in globals() if name.isupper()]
