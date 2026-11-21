"""Named internal axes without selecting unresolved graph physics."""

from __future__ import annotations

from dataclasses import dataclass


GRAPH_AXIS_MEANINGS = {
    "H": "parameter-vector or candidate batch",
    "P": "inferred-parameter coordinate",
    "N": "oscillator or node",
    "B": "independent graph-disorder ensemble batch",
    "R": "stochastic repetition conditional on a graph-disorder realization",
    "V": "ordered force/gauge bias condition",
    "M": "aggregated response component",
}


@dataclass(frozen=True)
class GraphAxisSizes:
    """Concrete sizes for the graph tensor vocabulary."""

    H: int
    P: int
    N: int
    B: int
    R: int
    V: int
    M: int

    def __post_init__(self) -> None:
        for name in GRAPH_AXIS_MEANINGS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"graph axis {name} must be a positive integer")

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in GRAPH_AXIS_MEANINGS}
