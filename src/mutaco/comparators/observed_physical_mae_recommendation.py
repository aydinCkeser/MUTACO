"""Surrogate-free evaluated-best physical-MAE terminal recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mutaco.policies import RecommendationPolicy, descriptor_for
from mutaco.recommendation.recommendation_contracts import EvaluatedDesigns


EVALUATED_BEST_OBSERVED_PHYSICAL_MAE = (
    "evaluated_best_observed_physical_mae_v1"
)


@dataclass(frozen=True)
class ObservedPhysicalMAERecommendationResult:
    policy_id: str
    classification: str
    direction: str
    selected_index: int
    selected_observation_id: str
    selected_theta: torch.Tensor
    score: torch.Tensor
    scores: torch.Tensor
    score_semantics: str = "observed_physical_response_mae"
    tie_rule: str = "earliest_evaluation"
    simulator_calls: int = 0
    gp_fits: int = 0

    def document(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "classification": self.classification,
            "direction": self.direction,
            "selected_index": self.selected_index,
            "selected_observation_id": self.selected_observation_id,
            "selected_theta": self.selected_theta.detach().cpu().tolist(),
            "score": float(self.score.detach().cpu()),
            "scores": self.scores.detach().cpu().tolist(),
            "score_semantics": self.score_semantics,
            "tie_rule": self.tie_rule,
            "simulator_calls": self.simulator_calls,
            "gp_fits": self.gp_fits,
        }


def recommend_best_observed_physical_mae(
    designs: EvaluatedDesigns,
    responses_physical: torch.Tensor,
    target_physical: torch.Tensor,
) -> ObservedPhysicalMAERecommendationResult:
    """Choose the earliest evaluated row minimizing mean_m |y_m - target_m|.

    Physical responses [N,M], target [M] and theta [N,P] share dtype/device and
    observation order. Scores use realized responses, which may be noisy; they
    are neither final-GP posterior-mean scores nor fresh held-out assessment.
    Selected values and score arrays are detached copies. No fit or simulator
    call is made, and exact ties retain the earliest observation.
    """
    if (
        not isinstance(responses_physical, torch.Tensor)
        or responses_physical.ndim != 2
        or responses_physical.shape[0] != designs.theta.shape[0]
        or responses_physical.shape[1] < 1
    ):
        raise ValueError("observed responses must have shape [N,M]")
    if target_physical.shape != (responses_physical.shape[1],):
        raise ValueError("physical target must have shape [M]")
    if (
        responses_physical.dtype != designs.theta.dtype
        or responses_physical.device != designs.theta.device
        or target_physical.dtype != designs.theta.dtype
        or target_physical.device != designs.theta.device
    ):
        raise ValueError("designs, responses, and target must share dtype/device")
    if (
        not responses_physical.is_contiguous()
        or not target_physical.is_contiguous()
        or not bool(torch.isfinite(responses_physical).all())
        or not bool(torch.isfinite(target_physical).all())
    ):
        raise ValueError("responses and target must be finite and contiguous")
    scores = torch.abs(responses_physical - target_physical).mean(dim=-1)
    index = int(torch.argmin(scores).item())
    policy = RecommendationPolicy.EVALUATED_BEST_OBSERVED_PHYSICAL_MAE
    descriptor = descriptor_for(policy.value)
    return ObservedPhysicalMAERecommendationResult(
        policy_id=policy.value,
        classification=descriptor.classification.value,
        direction=descriptor.direction.value,
        selected_index=index,
        selected_observation_id=designs.observation_ids[index],
        selected_theta=designs.theta[index].detach().clone().contiguous(),
        score=scores[index].detach().clone(),
        scores=scores.detach().clone().contiguous(),
    )
