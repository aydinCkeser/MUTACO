"""Manuscript Step 3 objectives in frozen standardized response coordinates.

Minimize J=||r||_2-alpha*log(Delta) in the strict compatibility interior;
otherwise the paper's exploration branch maximizes latent lambda_max.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from mutaco.policies import ExplorationPolicy, descriptor_for
from mutaco.posterior import LatentPosteriorSummary
from mutaco.response import StandardizedTarget

from .confidence_geometry import FixedBeta, evaluate_confidence_geometry


def standardized_l2_exploitation(
    posterior: LatentPosteriorSummary,
    target: StandardizedTarget,
) -> torch.Tensor:
    """Return ||mu-target||_2 per candidate, a scalar or [C] standardized score."""
    if posterior.transform_id != target.transform_id:
        raise ValueError("posterior and target transform identities do not match")
    if target.values.shape != (posterior.response_dim,):
        raise ValueError("standardized target has the wrong response dimension")
    if target.values.dtype != posterior.mean.dtype or target.values.device != posterior.mean.device:
        raise ValueError("standardized target must match posterior dtype and device")
    return torch.linalg.vector_norm(posterior.mean - target.values, ord=2, dim=-1)


@dataclass(frozen=True)
class BarrierObjectiveResult:
    values: torch.Tensor
    valid_domain: torch.Tensor
    exploitation: torch.Tensor
    margin: torch.Tensor
    feasibility_score: torch.Tensor
    policy_id: str = "standardized_l2_matched_log_barrier"


@dataclass(frozen=True)
class MatchedBarrierObjective:
    """Barrier with tau_interior equal to the manuscript's epsilon_Delta."""

    alpha: float
    tau_interior: float
    beta: FixedBeta = FixedBeta()
    policy_id: str = "standardized_l2_matched_log_barrier"
    radius_policy: str = "operator_norm_radius_v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(self.tau_interior) or self.tau_interior < 0:
            raise ValueError("tau_interior must be finite and nonnegative")

    def evaluate(
        self,
        posterior: LatentPosteriorSummary,
        target: StandardizedTarget,
    ) -> BarrierObjectiveResult:
        """Evaluate J for Delta>tau_interior; return +inf outside that domain.

        margin is Delta, exploitation is the standardized L2 norm, and values
        are minimized. valid_domain tests strict slack; screening additionally
        requires finite objective values. Inputs retain posterior dtype/device.
        """
        geometry = evaluate_confidence_geometry(
            posterior, target, beta=self.beta, radius_policy=self.radius_policy
        )
        exploitation = standardized_l2_exploitation(posterior, target)
        valid = geometry.margin > self.tau_interior
        # Mask before log as well as afterwards: invalid slack must not generate
        # undefined intermediate values in differentiable refinement.
        safe_margin = torch.where(valid, geometry.margin, torch.ones_like(geometry.margin))
        interior_value = exploitation - self.alpha * torch.log(safe_margin)
        values = torch.where(valid, interior_value, torch.full_like(interior_value, torch.inf))
        return BarrierObjectiveResult(
            values=values,
            valid_domain=valid,
            exploitation=exploitation,
            margin=geometry.margin,
            feasibility_score=geometry.feasibility_score,
            policy_id=self.policy_id,
        )


@dataclass(frozen=True)
class ExplorationResult:
    values: torch.Tensor
    policy_id: str
    classification: str
    direction: str
    simulator_calls: int = 0


def evaluate_exploration(
    posterior: LatentPosteriorSummary,
    policy: ExplorationPolicy | str = ExplorationPolicy.OPERATOR_NORM_UNCERTAINTY,
) -> ExplorationResult:
    """Return uncertainty to maximize, scalar or [C], without simulator work.

    The paper policy uses the largest latent task-covariance eigenvalue; trace
    is a registered alternative. Neither score is a physical response error.
    """
    try:
        selected = ExplorationPolicy(policy)
    except ValueError as exc:
        raise ValueError(f"unknown exploration policy: {policy!r}") from exc
    values = (
        posterior.largest_eigenvalue
        if selected is ExplorationPolicy.OPERATOR_NORM_UNCERTAINTY
        else posterior.trace
    )
    descriptor = descriptor_for(selected.value)
    return ExplorationResult(
        values=values,
        policy_id=selected.value,
        classification=descriptor.classification.value,
        direction=descriptor.direction.value,
        simulator_calls=descriptor.simulator_calls,
    )
