"""Manuscript Step 4: continuous physical posterior-mean MAE recommendation.

Use the final fitted GP, undo its frozen response transform and minimize mean
absolute target mismatch. The shared bounded optimizer receives MAE scoring
callbacks; sharing its numerical settings with L2 does not change this objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from mutaco.policies import RecommendationPolicy, descriptor_for
from mutaco.randomness import SeedDeriver
from mutaco.response import FrozenResponseTransform
from mutaco.types import ParameterSpec

from ._continuous import optimize_continuous_physical
from .physical_l2 import (
    CONTINUOUS_PHYSICAL_L2_SETTINGS,
    MeanOnlyPosterior,
    PhysicalL2RecommendationError,
)


CONTINUOUS_PHYSICAL_MAE = (
    RecommendationPolicy.CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L1_MAE.value
)
CONTINUOUS_PHYSICAL_MAE_SETTINGS = dict(CONTINUOUS_PHYSICAL_L2_SETTINGS)


class PhysicalMAERecommendationError(PhysicalL2RecommendationError):
    """Typed terminal failure authorizing no refit or simulator call."""


@dataclass(frozen=True)
class PhysicalMAERecommendationResult:
    policy_id: str
    classification: str
    candidate_domain: str
    selected_theta: torch.Tensor
    predicted_physical_response: torch.Tensor
    score: torch.Tensor
    model_id: str
    transform_id: str
    target_id: str
    selected_evaluation_index: int | None
    selected_observation_id: str | None
    returned_candidate_source: str
    posterior_calls: int
    simulator_calls: int
    gp_fits: int
    provenance: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "classification": self.classification,
            "candidate_domain": self.candidate_domain,
            "score_formula": "mean_m |mu_phys,m(theta)-target_phys,m|",
            "selected_theta": self.selected_theta.detach().cpu().tolist(),
            "predicted_physical_response": (
                self.predicted_physical_response.detach().cpu().tolist()
            ),
            "score": float(self.score.detach().cpu()),
            "model_id": self.model_id,
            "transform_id": self.transform_id,
            "target_id": self.target_id,
            "selected_evaluation_index": self.selected_evaluation_index,
            "selected_observation_id": self.selected_observation_id,
            "returned_candidate_source": self.returned_candidate_source,
            "posterior_calls": self.posterior_calls,
            "simulator_calls": self.simulator_calls,
            "gp_fits": self.gp_fits,
            **dict(self.provenance),
        }


def physical_mae_scores(
    posterior_standardized_mean: torch.Tensor,
    *,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return physical means and mean_p |mu_phys[p]-target[p]| per candidate.

    A standardized mean [M] returns ([M], scalar); [C,M] returns ([C,M], [C]).
    The physical target [M], response order, dtype/device and contiguity must
    match the frozen transform. This is absolute error of the posterior mean,
    not posterior expectation of absolute error or standardized-space L1.
    """

    scalar = posterior_standardized_mean.ndim == 1
    values = (
        posterior_standardized_mean.unsqueeze(0)
        if scalar
        else posterior_standardized_mean
    )
    if values.ndim != 2 or target_physical.shape != (values.shape[1],):
        raise ValueError("physical-MAE scoring requires mean [C,M] and target [M]")
    if values.dtype != transform.dtype or values.device != transform.device:
        raise ValueError("posterior mean must match transform dtype and device")
    if target_physical.dtype != transform.dtype or target_physical.device != transform.device:
        raise ValueError("physical target must match transform dtype and device")
    if not values.is_contiguous() or not target_physical.is_contiguous():
        raise ValueError("physical-MAE inputs must be contiguous")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(target_physical).all()):
        raise ValueError("physical-MAE inputs must be finite")
    physical = transform.to_physical(
        values, response_names=transform.metadata.response_names
    )
    scores = (physical - target_physical).abs().mean(dim=-1)
    if not bool(torch.isfinite(physical).all()) or not bool(torch.isfinite(scores).all()):
        raise ValueError("physical-MAE outputs must be finite")
    return (physical[0], scores[0]) if scalar else (physical, scores)


def _score(
    surrogate: MeanOnlyPosterior,
    theta: torch.Tensor,
    transform: FrozenResponseTransform,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return physical_mae_scores(
        surrogate.posterior_mean(theta.contiguous()).mean,
        transform=transform,
        target_physical=target,
    )


def _raw_score(
    surrogate: MeanOnlyPosterior,
    theta: torch.Tensor,
    transform: FrozenResponseTransform,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = surrogate.posterior_mean(theta.contiguous()).mean
    if mean.ndim != 2 or mean.shape[0] != theta.shape[0]:
        raise ValueError("raw posterior mean must have shape [C,M]")
    valid = torch.isfinite(mean).all(dim=-1)
    physical = torch.full_like(mean, torch.nan)
    scores = torch.full(
        (theta.shape[0],), torch.inf, dtype=mean.dtype, device=mean.device
    )
    if bool(valid.any()):
        physical[valid], scores[valid] = physical_mae_scores(
            mean[valid].contiguous(),
            transform=transform,
            target_physical=target,
        )
    return physical, scores


def recommend_continuous_physical_mae(
    *,
    surrogate: MeanOnlyPosterior,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
    target_id: str,
    parameter_spec: ParameterSpec,
    seed_deriver: SeedDeriver,
    target_index: int,
) -> PhysicalMAERecommendationResult:
    """Recommend a bounded continuous [P] point from the final GP's physical MAE.

    Delegate to _continuous.optimize_continuous_physical with MAE score callbacks:
    1024 named-seed Sobol points and up to ten projected-Adam restarts (64 steps,
    learning rate 0.05 in the shared settings). L2 supplies the optimizer recipe,
    not the scoring formula. Queries match target/transform dtype and device.

    The existing optimizer selects among accepted per-start non-worsening
    refinements; only when none is accepted does it return the raw best. It does
    not compare an accepted refinement with the global raw best. Preserve this
    selection policy pending separate scientific review. No compatibility
    constraint, new GP fit or simulator response is added at this terminal stage.
    """

    optimized = optimize_continuous_physical(
        surrogate=surrogate,
        transform=transform,
        target_physical=target_physical,
        parameter_spec=parameter_spec,
        seed_deriver=seed_deriver,
        target_index=target_index,
        policy_id=CONTINUOUS_PHYSICAL_MAE,
        settings=CONTINUOUS_PHYSICAL_MAE_SETTINGS,
        score=_score,
        raw_score=_raw_score,
        error_type=PhysicalMAERecommendationError,
    )
    descriptor = descriptor_for(CONTINUOUS_PHYSICAL_MAE)
    return PhysicalMAERecommendationResult(
        policy_id=CONTINUOUS_PHYSICAL_MAE,
        classification=descriptor.classification.value,
        candidate_domain="continuous_bounded_theta",
        selected_theta=optimized.selected_theta,
        predicted_physical_response=optimized.selected_physical,
        score=torch.tensor(
            optimized.selected_score,
            device=target_physical.device,
            dtype=target_physical.dtype,
        ),
        model_id=surrogate.model_id,
        transform_id=transform.metadata.transform_id,
        target_id=target_id,
        selected_evaluation_index=None,
        selected_observation_id=None,
        returned_candidate_source=optimized.returned_candidate_source,
        posterior_calls=optimized.posterior_calls,
        simulator_calls=descriptor.simulator_calls,
        gp_fits=0,
        provenance=optimized.provenance,
    )
