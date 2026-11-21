"""Strict canonical JSON and separate scientific/execution identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON forbids non-finite floats")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the identity payload deterministically as compact UTF-8 JSON.

    Mapping keys are sorted, but list/tuple order is preserved: observation and
    task ordering remain scientifically significant. Nonfinite floats and tensors
    are rejected; callers must explicitly project tensor values into the declared
    identity schema rather than relying on device-specific object serialization.
    """

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_hash(prefix: str, value: Any) -> str:
    if not prefix or any(character.isspace() for character in prefix):
        raise ValueError("hash prefix must be nonempty and contain no whitespace")
    return f"{prefix}-{sha256(canonical_json_bytes(value)).hexdigest()}"


@dataclass(frozen=True)
class ScientificIdentity:
    """Identity of mathematical and stochastic experiment intent."""

    value: str

    @classmethod
    def create(cls, definition: Mapping[str, Any]) -> "ScientificIdentity":
        return cls(stable_hash("scientific-v1", definition))


@dataclass(frozen=True)
class ExecutionIdentity:
    """Identity of one placement/environment execution of a scientific run."""

    value: str
    scientific_identity: str

    @classmethod
    def create(
        cls,
        scientific_identity: ScientificIdentity,
        execution_provenance: Mapping[str, Any],
    ) -> "ExecutionIdentity":
        payload = {
            "scientific_identity": scientific_identity.value,
            "execution": execution_provenance,
        }
        return cls(stable_hash("execution-v1", payload), scientific_identity.value)
