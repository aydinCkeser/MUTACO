"""Validated simulator-independent parameter and response metadata."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class ParameterAxis:
    """One ordered physical parameter coordinate with closed bounds.

    unit, transformation and role describe the declared model coordinate; this
    record performs no unit conversion or logarithmic/logit mapping. The current
    sequential intent accepts identity transformations only.
    """
    name: str
    lower: float
    upper: float
    unit: str = "dimensionless"
    transformation: Literal["identity", "log", "logit"] = "identity"
    role: Literal["inferred", "fixed"] = "inferred"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must be nonempty")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError(f"parameter {self.name!r} bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError(f"parameter {self.name!r} lower bound must be below upper bound")
        if self.transformation not in {"identity", "log", "logit"}:
            raise ValueError(f"unsupported parameter transformation: {self.transformation!r}")
        if self.role not in {"inferred", "fixed"}:
            raise ValueError(f"unsupported parameter role: {self.role!r}")
        if self.transformation == "log" and self.lower <= 0:
            raise ValueError(f"log-transformed parameter {self.name!r} requires a positive lower bound")


@dataclass(frozen=True)
class ResponseAxis:
    name: str
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("response name must be nonempty")


@dataclass(frozen=True)
class ParameterSpec:
    """Ordered axes defining columns of every physical parameter tensor [...,P].

    Names are unique and lower/upper tuples retain this order. Array position,
    not alphabetical sorting, determines which model parameter a value represents.
    """
    axes: tuple[ParameterAxis, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("at least one parameter axis is required")
        names = tuple(axis.name for axis in self.axes)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique and ordered")

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def lower(self) -> tuple[float, ...]:
        return tuple(axis.lower for axis in self.axes)

    @property
    def upper(self) -> tuple[float, ...]:
        return tuple(axis.upper for axis in self.axes)


@dataclass(frozen=True)
class ResponseSpec:
    """Ordered physical response channels defining the task axis [...,M].

    Names are unique and units are descriptive. Frozen transforms, GP tasks and
    target vectors must preserve this same order; the specification itself neither
    normalizes responses nor aggregates channels with different physical scales.
    """
    axes: tuple[ResponseAxis, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("at least one response axis is required")
        names = tuple(axis.name for axis in self.axes)
        if len(set(names)) != len(names):
            raise ValueError("response names must be unique and ordered")

    @property
    def dimension(self) -> int:
        return len(self.axes)


# Trusted historical pickles refer to this original Python module name.
for _legacy_export in (
    ParameterAxis,
    ResponseAxis,
    ParameterSpec,
    ResponseSpec,
):
    _legacy_export.__module__ = "mutaco.types.specifications"
del _legacy_export
