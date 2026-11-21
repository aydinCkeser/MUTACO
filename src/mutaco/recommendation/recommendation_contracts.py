"""Inputs and failure contracts shared by evaluated and continuous terminal policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from mutaco._legacy_module import preserve_legacy_definition


@dataclass(frozen=True)
class EvaluatedDesigns:
    """Ordered previously evaluated physical parameters [N,P] and unique IDs.

    Rows define deterministic earliest-tie selection. This validated container
    retains the supplied tensor; it neither copies observations nor authorizes
    new simulator evaluations.
    """
    theta: torch.Tensor
    observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.theta, torch.Tensor) or self.theta.ndim != 2:
            raise ValueError("evaluated theta must have shape [N,P]")
        if self.theta.shape[0] < 1 or self.theta.shape[1] < 1:
            raise ValueError("evaluated theta requires positive N and P")
        if self.theta.dtype not in {torch.float32, torch.float64}:
            raise ValueError("evaluated theta requires float32 or float64")
        if not self.theta.is_contiguous() or not bool(torch.isfinite(self.theta).all()):
            raise ValueError("evaluated theta must be contiguous and finite")
        if len(self.observation_ids) != self.theta.shape[0]:
            raise ValueError("observation_ids must match evaluated theta rows")
        if len(set(self.observation_ids)) != len(self.observation_ids) or any(
            not item for item in self.observation_ids
        ):
            raise ValueError("observation_ids must be unique and nonempty")

    @property
    def parameter_dim(self) -> int:
        return self.theta.shape[1]


class MeanOnlyPosterior(Protocol):
    model_id: str

    def posterior_mean(self, theta: torch.Tensor) -> Any: ...


class PhysicalL2RecommendationError(RuntimeError):
    """Typed terminal failure that authorizes no refit, resample, or simulator call."""

# Legacy result pickles and error handling address these public definitions by name.
preserve_legacy_definition(EvaluatedDesigns, "mutaco.recommendation.policies")
preserve_legacy_definition(MeanOnlyPosterior, "mutaco.recommendation.physical_l2")
preserve_legacy_definition(PhysicalL2RecommendationError, "mutaco.recommendation.physical_l2")
