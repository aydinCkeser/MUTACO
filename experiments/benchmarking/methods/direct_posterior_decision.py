"""Prospective zero-call final decisions for learned direct posteriors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

from experiments.benchmarking.timing import TimingLedger
from experiments.contracts import PhysicalCallCounts
from experiments.paper.common.protocol_v3 import DecisionOutcome
from .sbi_zoo_contract import SBI_ZOO_PHYSICAL_SCORE_API, SBI_ZOO_SCORE_IDENTITIES
from .sbi_zoo_failures import SbiZooScoringContractError


POSTERIOR_SAMPLE_MAP_POLICY = "posterior_sample_map_physical_v1"
POSTERIOR_SAMPLE_MAP_LABEL = "sample_map"
POSTERIOR_SAMPLE_MAP_TASK = "posterior_sample_map_scoring_selection"
POSTERIOR_SAMPLE_MAP_METHODS = frozenset(
    {
        "sbi_npe",
        "sbi_snpe",
        "sbi_nle_single_round",
        "sbi_nre_single_round",
    }
)
HISTORICAL_DIRECT_POSTERIOR_METHODS = frozenset({"sbi_npe", "sbi_snpe"})
HISTORICAL_DIRECT_POSTERIOR_SCORE_API = "sbi_direct_posterior_log_prob_physical_v1"


class PosteriorSampleMapDecisionError(RuntimeError):
    """Typed failure raised internally when sample-MAP cannot be selected."""


@dataclass(frozen=True)
class PosteriorSampleMapResult:
    """A scientific v3 decision plus separate observational timing evidence."""

    decision: DecisionOutcome
    timing_event: dict[str, Any]


def _score_identity(posterior: Any) -> str:
    cls = type(posterior)
    return f"{cls.__module__}.{cls.__qualname__}.log_prob"


def _base_metadata(
    *,
    method: str,
    samples: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    posterior: Any,
    score_identity: str,
    score_api: str,
) -> dict[str, Any]:
    return {
        "policy_identifier": POSTERIOR_SAMPLE_MAP_POLICY,
        "method_family": method,
        "decision_label": POSTERIOR_SAMPLE_MAP_LABEL,
        "candidate_count": int(samples.shape[0]),
        "selected_candidate_index": None,
        "selected_theta": None,
        "selected_score": None,
        "score_identity": score_identity,
        "score_api": score_api,
        "coordinate_system": "physical_theta",
        "density_measure": "physical_theta_lebesgue",
        "physical_bounds": {
            "lower": lower.detach().cpu().tolist(),
            "upper": upper.detach().cpu().tolist(),
        },
        "support_mask_policy": "finite_and_closed_physical_box_v1",
        "tie_rule": "earliest_exact_score_tie",
        "transform_jacobian_policy": "none",
        "refinement": "none",
        "recommendation_simulator_calls": 0,
    }


def posterior_sample_map_decision(
    *,
    method: str,
    posterior: Any,
    samples: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    method_configuration_identity: str,
    inference_budget: int,
    inference_calls: PhysicalCallCounts,
    target_index: int | None = None,
    physical_score: Callable[[torch.Tensor], torch.Tensor] | None = None,
    physical_score_identity: str | None = None,
    physical_score_api: str = HISTORICAL_DIRECT_POSTERIOR_SCORE_API,
) -> PosteriorSampleMapResult:
    """Select the earliest highest-scoring finite in-support physical sample.

    ``samples`` is the already configured ordered posterior sample population.
    This function draws no samples, consumes no simulator calls, and performs
    no refinement or fallback.
    """

    if method not in POSTERIOR_SAMPLE_MAP_METHODS:
        raise ValueError(
            "posterior sample-MAP is approved only for NPE and SNPE or the "
            "frozen single-round NLE/NRE extension"
        )
    if method in HISTORICAL_DIRECT_POSTERIOR_METHODS:
        if physical_score is not None or physical_score_identity is not None:
            raise SbiZooScoringContractError(
                "historical NPE/SNPE forbid custom physical scoring"
            )
        if physical_score_api != HISTORICAL_DIRECT_POSTERIOR_SCORE_API:
            raise SbiZooScoringContractError(
                "historical NPE/SNPE require the canonical direct-posterior score API"
            )
    else:
        expected_identity = SBI_ZOO_SCORE_IDENTITIES[method]
        if physical_score is None:
            raise SbiZooScoringContractError(
                "single-round NLE/NRE require the physical posterior-potential scorer"
            )
        if physical_score_api != SBI_ZOO_PHYSICAL_SCORE_API:
            raise SbiZooScoringContractError(
                "single-round NLE/NRE require the no-Jacobian physical score API"
            )
        if physical_score_identity != expected_identity:
            raise SbiZooScoringContractError(
                f"{method} requires score identity {expected_identity}"
            )
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError(
            "posterior samples must have shape (K, parameter_dimension)"
        )
    if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
        raise ValueError("physical bounds must be matching one-dimensional tensors")
    if samples.shape[1] != lower.shape[0]:
        raise ValueError("posterior sample and physical-bound dimensions differ")
    if samples.device != lower.device or samples.device != upper.device:
        raise ValueError("posterior samples and physical bounds must share a device")
    if samples.dtype != lower.dtype or samples.dtype != upper.dtype:
        raise ValueError("posterior samples and physical bounds must share a dtype")
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise ValueError("physical bounds must be finite")
    if not bool(torch.all(lower < upper)):
        raise ValueError("every physical lower bound must be below its upper bound")

    metadata = _base_metadata(
        method=method,
        samples=samples,
        lower=lower,
        upper=upper,
        posterior=posterior,
        score_identity=(
            _score_identity(posterior)
            if physical_score_identity is None
            else physical_score_identity
        ),
        score_api=physical_score_api,
    )
    ledger = TimingLedger()
    try:
        with ledger.event(
            POSTERIOR_SAMPLE_MAP_TASK,
            method=method,
            target_index=target_index,
            work_count=int(samples.shape[0]),
            simulator_calls=0,
            budget_classification="zero_call_terminal_recommendation",
            device=str(samples.device),
        ):
            finite = torch.isfinite(samples).all(dim=1)
            in_support = ((samples >= lower) & (samples <= upper)).all(dim=1)
            candidate_indices = torch.nonzero(
                finite & in_support, as_tuple=False
            ).flatten()
            if candidate_indices.numel() == 0:
                raise PosteriorSampleMapDecisionError(
                    "no finite in-support posterior sample candidate"
                )
            physical_candidates = samples.index_select(0, candidate_indices)
            with torch.no_grad():
                raw_scores = (
                    posterior.log_prob(
                        physical_candidates,
                        norm_posterior=False,
                    )
                    if physical_score is None
                    else physical_score(physical_candidates)
                )
            if not isinstance(raw_scores, torch.Tensor):
                raise PosteriorSampleMapDecisionError(
                    "direct-posterior log_prob did not return a tensor"
                )
            scores = raw_scores.reshape(-1)
            if scores.numel() != candidate_indices.numel():
                raise PosteriorSampleMapDecisionError(
                    "direct-posterior log_prob returned the wrong score count"
                )
            finite_scores = torch.isfinite(scores)
            if not bool(finite_scores.any()):
                raise PosteriorSampleMapDecisionError(
                    "no finite in-support posterior sample score"
                )
            eligible_indices = candidate_indices[finite_scores]
            eligible_scores = scores[finite_scores]
            # torch.argmax returns the first maximum, preserving the approved
            # earliest-candidate rule because eligible_indices retains order.
            local_index = int(torch.argmax(eligible_scores).item())
            selected_index = int(eligible_indices[local_index].item())
            selected_theta = samples[selected_index].detach().clone().contiguous()
            selected_score = float(eligible_scores[local_index].detach().cpu())
            if not math.isfinite(selected_score):
                raise PosteriorSampleMapDecisionError(
                    "selected posterior sample score is nonfinite"
                )
            metadata.update(
                selected_candidate_index=selected_index,
                selected_theta=selected_theta.detach().cpu().tolist(),
                selected_score=selected_score,
            )
    except Exception as exc:
        decision = DecisionOutcome(
            method=method,
            status="failed",
            decision_policy_identifier=POSTERIOR_SAMPLE_MAP_POLICY,
            method_configuration_identity=method_configuration_identity,
            inference_budget=inference_budget,
            inference_calls=inference_calls,
            failure_type=type(exc).__name__,
            failure_explanation=str(exc),
            decision_metadata=metadata,
        )
    else:
        decision = DecisionOutcome(
            method=method,
            status="recommended",
            decision_policy_identifier=POSTERIOR_SAMPLE_MAP_POLICY,
            method_configuration_identity=method_configuration_identity,
            inference_budget=inference_budget,
            inference_calls=inference_calls,
            recommended_theta=tuple(float(item) for item in selected_theta.cpu()),
            decision_metadata=metadata,
        )
    assert len(ledger.events) == 1
    return PosteriorSampleMapResult(decision=decision, timing_event=ledger.events[0])
