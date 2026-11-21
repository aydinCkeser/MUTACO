"""Canonical, path-aware representation for persisted experiment evidence."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


NONFINITE_TAG = "__mutaco_evidence_nonfinite_float_v1__"
POSITIVE_INFINITY = "positive_infinity"
NEGATIVE_INFINITY = "negative_infinity"


class _EncodedNonfinite(dict[str, str]):
    """Distinguish encoder-created tags from colliding scientific mappings."""


class EvidenceCanonicalizationError(ValueError):
    """A value cannot enter canonical evidence; includes its exact document path."""

    def __init__(self, message: str, *, path: str, value: Any) -> None:
        self.path = path
        self.value_type = type(value).__name__
        self.value_repr = safe_evidence_repr(value)
        super().__init__(f"{message} at {path} ({self.value_type}: {self.value_repr})")


def safe_evidence_repr(value: Any, *, limit: int = 240) -> str:
    """Return a bounded diagnostic representation that cannot raise."""

    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive boundary
        rendered = f"<repr failed: {type(exc).__name__}>"
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _child_path(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def _nonfinite_tag(value: float) -> _EncodedNonfinite:
    return _EncodedNonfinite(
        {NONFINITE_TAG: POSITIVE_INFINITY if value > 0.0 else NEGATIVE_INFINITY}
    )


def canonicalize_evidence(value: Any, *, path: str = "$") -> Any:
    """Recursively create the sole JSON-safe semantic evidence representation.

    Finite values retain the representation accepted by MUTACO's canonical JSON.
    Signed infinities receive reserved tagged objects. NaN and tag collisions fail
    closed with an exact JSON-style path.
    """

    if isinstance(value, _EncodedNonfinite):
        if set(value) != {NONFINITE_TAG} or value[NONFINITE_TAG] not in {
            POSITIVE_INFINITY,
            NEGATIVE_INFINITY,
        }:
            raise EvidenceCanonicalizationError(
                "invalid encoder-created nonfinite tag", path=path, value=value
            )
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize_evidence(asdict(value), path=path)
    if isinstance(value, Enum):
        return canonicalize_evidence(value.value, path=path)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise EvidenceCanonicalizationError(
                "evidence mappings require string keys", path=path, value=value
            )
        if NONFINITE_TAG in value:
            raise EvidenceCanonicalizationError(
                "scientific evidence collides with reserved nonfinite tag",
                path=_child_path(path, NONFINITE_TAG),
                value=value[NONFINITE_TAG],
            )
        return {
            key: canonicalize_evidence(value[key], path=_child_path(path, key))
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [
            canonicalize_evidence(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return canonicalize_evidence(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return canonicalize_evidence(value.item(), path=path)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise EvidenceCanonicalizationError(
                "canonical evidence forbids NaN", path=path, value=value
            )
        if math.isinf(value):
            return _nonfinite_tag(value)
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise EvidenceCanonicalizationError(
        "unsupported canonical evidence value", path=path, value=value
    )


def encoded_nonfinite_kind(value: Any) -> str | None:
    """Return the signed-infinity kind from canonical or reloaded evidence."""

    if isinstance(value, Mapping) and set(value) == {NONFINITE_TAG}:
        kind = value.get(NONFINITE_TAG)
        if kind in {POSITIVE_INFINITY, NEGATIVE_INFINITY}:
            return kind
    return None
