"""Authoritative evaluated-point terminal scoring without simulator calls."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mutaco.policies import RecommendationPolicy, descriptor_for
from mutaco.posterior import LatentPosteriorMeanSummary, LatentPosteriorSummary
from mutaco.response import StandardizedTarget
from .recommendation_contracts import EvaluatedDesigns


@dataclass(frozen=True)
class RecommendationResult:
    policy_id: str
    classification: str
    direction: str
    selected_index: int
    selected_observation_id: str
    selected_theta: torch.Tensor
    score: torch.Tensor
    scores: torch.Tensor
    posterior_model_id: str
    transform_id: str
    simulator_calls: int = 0


def recommend_evaluated(
    designs: EvaluatedDesigns,
    posterior: LatentPosteriorSummary | LatentPosteriorMeanSummary,
    target: StandardizedTarget,
    *,
    policy: RecommendationPolicy | str = RecommendationPolicy.FINAL_POSTERIOR_STANDARDIZED_L1,
) -> RecommendationResult:
    """Minimize the selected standardized posterior loss over evaluated designs.

    Match target [M], transform identity and posterior task order. The L1 policy
    sums absolute mean residuals; expected squared L2 adds covariance trace to the
    squared mean residual norm. Neither is the paper's continuous physical MAE.
    Exact score ties choose the first evaluated row. Return owned selected theta
    and score copies without fitting a GP or evaluating a simulator.
    """
    if not posterior.is_batched or posterior.mean.shape[0] != designs.theta.shape[0]:
        raise ValueError("recommendation posterior must contain one row per evaluated theta")
    if designs.theta.device != posterior.mean.device or designs.theta.dtype != posterior.mean.dtype:
        raise ValueError("evaluated theta and posterior must share dtype and device")
    if target.transform_id != posterior.transform_id:
        raise ValueError("posterior and target transform identities do not match")
    if target.values.shape != (posterior.response_dim,):
        raise ValueError("standardized target has the wrong response dimension")
    if target.values.device != posterior.mean.device or target.values.dtype != posterior.mean.dtype:
        raise ValueError("standardized target must match posterior dtype and device")
    try:
        selected_policy = RecommendationPolicy(policy)
    except ValueError as exc:
        raise ValueError(f"unknown recommendation policy: {policy!r}") from exc

    residual = posterior.mean - target.values
    if selected_policy is RecommendationPolicy.FINAL_POSTERIOR_STANDARDIZED_L1:
        scores = torch.abs(residual).sum(dim=-1)
    elif selected_policy is RecommendationPolicy.EVALUATED_POSTERIOR_EXPECTED_L2:
        if not isinstance(posterior, LatentPosteriorSummary):
            raise ValueError(
                "evaluated_posterior_expected_l2 requires posterior covariance"
            )
        scores = residual.square().sum(dim=-1) + posterior.trace
    else:
        raise ValueError(
            f"{selected_policy.value} requires the physical-L2 recommendation interface"
        )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("recommendation scores must be finite")
    index = int(torch.argmin(scores).item())
    descriptor = descriptor_for(selected_policy.value)
    return RecommendationResult(
        policy_id=selected_policy.value,
        classification=descriptor.classification.value,
        direction=descriptor.direction.value,
        selected_index=index,
        selected_observation_id=designs.observation_ids[index],
        selected_theta=designs.theta[index].clone(),
        score=scores[index].clone(),
        scores=scores,
        posterior_model_id=posterior.model_id,
        transform_id=posterior.transform_id,
        simulator_calls=descriptor.simulator_calls,
    )
