"""Validation for externally supplied frozen target responses used by Paper-v3."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .protocol_v3 import PaperComparisonConfigV3
from .validation import PaperContractError


def _finite_vector(value: object, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise PaperContractError(f"supplied target {field} must be a nonempty list")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise PaperContractError(f"supplied target {field} must be finite")
        result.append(float(item))
    return result


def load_supplied_target_set(
    document: Mapping[str, Any], *, comparison: PaperComparisonConfigV3
) -> tuple[dict[str, Any], ...]:
    """Load already-frozen y* records; this performs no target construction."""

    if document.get("schema_version") != "mutaco-paper-supplied-target-set-v1":
        raise PaperContractError("supplied targets require the v1 target-set schema")
    if document.get("state") != "frozen":
        raise PaperContractError("supplied target set must be frozen before inference")
    if document.get("simulator_definition_identity") != comparison.simulator_definition_identity:
        raise PaperContractError("supplied target simulator definition is incompatible")
    if document.get("simulator_configuration_identity") != comparison.simulator_configuration_identity:
        raise PaperContractError("supplied target simulator configuration is incompatible")
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise PaperContractError("supplied target set requires records")
    records = []
    identities = set()
    indices = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise PaperContractError("supplied target records must be objects")
        identity = str(raw.get("target_record_id", ""))
        family = str(raw.get("target_family_id", ""))
        index = raw.get("target_index")
        seed = raw.get("target_seed")
        if not identity or not family or identity in identities:
            raise PaperContractError("supplied target identities must be nonempty and unique")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in indices:
            raise PaperContractError("supplied target indices must be unique nonnegative integers")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise PaperContractError("supplied target seed must be nonnegative")
        provenance = raw.get("target_randomness_provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise PaperContractError("supplied y* requires target randomness provenance")
        if raw.get("target_disposition") != "externally_supplied_frozen":
            raise PaperContractError("supplied target disposition is invalid")
        theta = raw.get("theta_star")
        records.append({
            **dict(raw),
            "target_record_id": identity,
            "target_family_id": family,
            "target_index": index,
            "target_seed": seed,
            "target_response": _finite_vector(raw.get("target_response"), "response"),
            "theta_star": None if theta is None else _finite_vector(theta, "theta_star"),
            "target_randomness_provenance": dict(provenance),
        })
        identities.add(identity)
        indices.add(index)
    return tuple(records)

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.supplied_targets")
