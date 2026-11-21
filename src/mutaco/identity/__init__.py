"""Canonical hashing and run identities."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import canonical_identity_hashing, strict_metadata_validation
from .canonical_identity_hashing import (
    ExecutionIdentity,
    ScientificIdentity,
    canonical_json_bytes,
    stable_hash,
)

from .strict_metadata_validation import (
    StrictJsonValue,
    StrictMetadataError,
    freeze_strict_metadata,
    strict_metadata_document,
)

# Sequential identity imports these two legacy addresses and the seed policy.
# Publish the names the seed module needs before loading sequential identity.
canonical = _bind_legacy_module(canonical_identity_hashing, __name__ + ".canonical")
metadata = _bind_legacy_module(strict_metadata_validation, __name__ + ".metadata")

from . import sequential_scientific_identity
from .sequential_scientific_identity import (
    BEST_RANKED_UNEVALUATED_RAW_POLICY,
    EXACT_DUPLICATE_POLICY,
    SEQUENTIAL_INTENT_SCHEMA,
    SEQUENTIAL_OBSERVATION_SCHEMA,
    SequentialScientificIntent,
    sequential_observation_identity,
)

sequential = _bind_legacy_module(sequential_scientific_identity, __name__ + ".sequential")

__all__ = [
    "BEST_RANKED_UNEVALUATED_RAW_POLICY",
    "EXACT_DUPLICATE_POLICY",
    "ExecutionIdentity",
    "SEQUENTIAL_INTENT_SCHEMA",
    "SEQUENTIAL_OBSERVATION_SCHEMA",
    "ScientificIdentity",
    "SequentialScientificIntent",
    "StrictJsonValue",
    "StrictMetadataError",
    "canonical_json_bytes",
    "freeze_strict_metadata",
    "sequential_observation_identity",
    "stable_hash",
    "strict_metadata_document",
]
