"""Simulator-independent posterior-mean physical squared-L2 recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence

import torch

from mutaco.policies import RecommendationPolicy, descriptor_for
from mutaco.randomness import SeedDeriver
from mutaco.response import FrozenResponseTransform
from mutaco.types import ParameterSpec

from .._continuous_terminal_optimizer import optimize_continuous_physical
from ..recommendation_contracts import MeanOnlyPosterior, PhysicalL2RecommendationError


EVALUATED_PHYSICAL_L2 = RecommendationPolicy.EVALUATED_POSTERIOR_MEAN_PHYSICAL_L2.value
CONTINUOUS_PHYSICAL_L2 = RecommendationPolicy.CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L2.value
CONTINUOUS_PHYSICAL_L2_SETTINGS: dict[str, Any] = {
    "raw_samples": 1024,
    "restarts": 10,
    "max_iterations": 64,
    "learning_rate": 0.05,
    "optimizer": "projected_adam",
    "convergence_tolerance": 1.0e-7,
}


@dataclass(frozen=True)
class PhysicalL2RecommendationResult:
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
            "score_formula": "||mu_phys(theta)-target_phys||_2^2",
            "selected_theta": self.selected_theta.detach().cpu().tolist(),
            "predicted_physical_response": self.predicted_physical_response.detach().cpu().tolist(),
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


def physical_l2_scores(
    posterior_standardized_mean: torch.Tensor,
    *,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inverse-transformed means and physical squared-L2 residual scores.

    Standardized mean [M] or [C,M] and physical target [M] must match the frozen
    transform's dtype/device and task order. Outputs preserve that batching: means
    [M]/[C,M] and scores scalar/[C]. Scores sum squared physical residuals; they
    are neither Euclidean norms nor posterior expected errors including variance.
    """

    scalar = posterior_standardized_mean.ndim == 1
    values = posterior_standardized_mean.unsqueeze(0) if scalar else posterior_standardized_mean
    if values.ndim != 2 or target_physical.shape != (values.shape[1],):
        raise ValueError("physical-L2 scoring requires mean [C,M] and target [M]")
    if values.dtype != transform.dtype or values.device != transform.device:
        raise ValueError("posterior mean must match transform dtype and device")
    if target_physical.dtype != transform.dtype or target_physical.device != transform.device:
        raise ValueError("physical target must match transform dtype and device")
    if not values.is_contiguous() or not target_physical.is_contiguous():
        raise ValueError("physical-L2 inputs must be contiguous")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(target_physical).all()):
        raise ValueError("physical-L2 inputs must be finite")
    physical = transform.to_physical(values, response_names=transform.metadata.response_names)
    scores = (physical - target_physical).square().sum(dim=-1)
    if not bool(torch.isfinite(physical).all()) or not bool(torch.isfinite(scores).all()):
        raise ValueError("physical-L2 outputs must be finite")
    return (physical[0], scores[0]) if scalar else (physical, scores)


def _mean_and_score(
    surrogate: MeanOnlyPosterior,
    theta: torch.Tensor,
    transform: FrozenResponseTransform,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return physical_l2_scores(
        surrogate.posterior_mean(theta.contiguous()).mean,
        transform=transform,
        target_physical=target,
    )


def _raw_mean_and_score(
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
    scores = torch.full((theta.shape[0],), torch.inf, dtype=mean.dtype, device=mean.device)
    if bool(valid.any()):
        physical[valid], scores[valid] = physical_l2_scores(
            mean[valid].contiguous(), transform=transform, target_physical=target
        )
    return physical, scores


def recommend_evaluated_physical_l2(
    *,
    surrogate: MeanOnlyPosterior,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
    target_id: str,
    theta: torch.Tensor,
    observation_ids: Sequence[str],
) -> PhysicalL2RecommendationResult:
    """Minimize final posterior-mean physical squared L2 over theta [N,P].

    The supplied observation IDs retain evaluation order; exact ties select the
    first row. Query means once, undo frozen standardization and return detached
    selected values. No GP fit, new observation or continuous refinement occurs.
    """
    started = perf_counter()
    physical, scores = _mean_and_score(surrogate, theta, transform, target_physical)
    if scores.ndim != 1 or len(observation_ids) != theta.shape[0]:
        raise ValueError("evaluated policy requires aligned evaluated designs")
    if not bool(torch.isfinite(scores).all()):
        raise PhysicalL2RecommendationError("nonfinite_evaluated_score")
    index = int(torch.argmin(scores).item())
    descriptor = descriptor_for(EVALUATED_PHYSICAL_L2)
    return PhysicalL2RecommendationResult(
        policy_id=EVALUATED_PHYSICAL_L2,
        classification=descriptor.classification.value,
        candidate_domain="evaluated_theta_only",
        selected_theta=theta[index].detach().clone(),
        predicted_physical_response=physical[index].detach().clone(),
        score=scores[index].detach().clone(), model_id=surrogate.model_id,
        transform_id=transform.metadata.transform_id, target_id=target_id,
        selected_evaluation_index=index, selected_observation_id=str(observation_ids[index]),
        returned_candidate_source="evaluated", posterior_calls=1,
        simulator_calls=descriptor.simulator_calls, gp_fits=0,
        provenance={"elapsed_seconds": perf_counter() - started, "dtype": str(theta.dtype), "device": str(theta.device)},
    )


def recommend_continuous_physical_l2(
    *,
    surrogate: MeanOnlyPosterior,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
    target_id: str,
    parameter_spec: ParameterSpec,
    seed_deriver: SeedDeriver,
    target_index: int,
) -> PhysicalL2RecommendationResult:
    """Minimize physical squared L2 using the shared bounded terminal optimizer.

    The score is sum_m (mu_phys_m - target_m)^2, distinct from physical MAE even
    though the optimizer settings are shared. Use the existing final GP and frozen
    transform; no fit or physical evaluation is made. The delegated multistart
    search is local and carries no global-optimality guarantee.
    """
    optimized = optimize_continuous_physical(
        surrogate=surrogate,
        transform=transform,
        target_physical=target_physical,
        parameter_spec=parameter_spec,
        seed_deriver=seed_deriver,
        target_index=target_index,
        policy_id=CONTINUOUS_PHYSICAL_L2,
        settings=CONTINUOUS_PHYSICAL_L2_SETTINGS,
        score=_mean_and_score,
        raw_score=_raw_mean_and_score,
        error_type=PhysicalL2RecommendationError,
    )
    descriptor = descriptor_for(CONTINUOUS_PHYSICAL_L2)
    return PhysicalL2RecommendationResult(
        policy_id=CONTINUOUS_PHYSICAL_L2, classification=descriptor.classification.value,
        candidate_domain="continuous_bounded_theta", selected_theta=optimized.selected_theta,
        predicted_physical_response=optimized.selected_physical,
        score=torch.tensor(optimized.selected_score, device=target_physical.device, dtype=target_physical.dtype),
        model_id=surrogate.model_id, transform_id=transform.metadata.transform_id, target_id=target_id,
        selected_evaluation_index=None, selected_observation_id=None,
        returned_candidate_source=optimized.returned_candidate_source,
        posterior_calls=optimized.posterior_calls,
        simulator_calls=descriptor.simulator_calls, gp_fits=0,
        provenance=optimized.provenance,
    )
