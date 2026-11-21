"""Strict JSON-compatible metadata admitted at simulator boundaries."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias


StrictJsonScalar: TypeAlias = None | bool | int | float | str
StrictJsonValue: TypeAlias = StrictJsonScalar | tuple["StrictJsonValue", ...] | Mapping[
    str, "StrictJsonValue"
]


class StrictMetadataError(ValueError):
    """Metadata is not a strict, finite JSON-compatible document."""


def _freeze(value: Any, *, path: str) -> StrictJsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictMetadataError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, list):
        return tuple(_freeze(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise StrictMetadataError(f"{path} requires string object keys")
        return MappingProxyType(
            {
                key: _freeze(value[key], path=f"{path}.{key}")
                for key in sorted(value)
            }
        )
    raise StrictMetadataError(
        f"{path} contains unsupported strict JSON type {type(value).__name__}"
    )


def freeze_strict_metadata(
    metadata: Mapping[str, Any],
    *,
    label: str = "metadata",
) -> Mapping[str, StrictJsonValue]:
    """Validate a JSON object and return a recursively immutable snapshot."""

    if isinstance(metadata, MappingProxyType):
        metadata = strict_metadata_document(metadata)
    if not isinstance(metadata, dict):
        raise StrictMetadataError(f"{label} must be a JSON object")
    frozen = _freeze(metadata, path=label)
    assert isinstance(frozen, Mapping)
    return frozen


def strict_metadata_document(
    metadata: Mapping[str, StrictJsonValue],
) -> dict[str, Any]:
    """Return a detached mutable JSON-shaped copy for provenance/serialization."""

    def thaw(value: StrictJsonValue) -> Any:
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, list):
            return [thaw(item) for item in value]
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    return {key: thaw(value) for key, value in metadata.items()}
