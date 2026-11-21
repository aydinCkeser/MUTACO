"""Internal bounded optimizer shared by continuous physical terminal policies."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import torch

from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.response import FrozenResponseTransform
from mutaco.types import ParameterSpec


ScoreFunction = Callable[
    [Any, torch.Tensor, FrozenResponseTransform, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True)
class ContinuousPhysicalOptimization:
    selected_theta: torch.Tensor
    selected_physical: torch.Tensor
    selected_score: float
    returned_candidate_source: str
    posterior_calls: int
    provenance: Mapping[str, Any]


def optimize_continuous_physical(
    *,
    surrogate: Any,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
    parameter_spec: ParameterSpec,
    seed_deriver: SeedDeriver,
    target_index: int,
    policy_id: str,
    settings: Mapping[str, Any],
    score: ScoreFunction,
    raw_score: ScoreFunction,
    error_type: type[RuntimeError],
) -> ContinuousPhysicalOptimization:
    """Minimize supplied physical-response scores by Sobol starts and projected Adam.

    Physical candidate rows [H,P] lie in ParameterSpec's closed box. score and
    raw_score supply posterior physical means [H,M] and scalar-per-row losses;
    they define MAE versus squared L2, not these shared optimizer settings. A
    named seed fixes the raw design; the existing final surrogate is queried only.

    Each restart retains finite iterates no worse than its own initial score.
    Choose the best accepted restart, breaking ties by restart order; use the
    best raw row only when no restart is accepted. Thus there is no unconditional
    comparison of accepted refinements with the global raw best, and no guarantee
    of a global optimum. Failures and stopping details remain in restart history;
    this function neither fits a GP nor evaluates a physical simulator.
    """

    started = perf_counter()
    resolved_settings = dict(settings)
    coordinate = SeedCoordinate(
        "terminal_recommendation", (target_index, policy_id)
    )
    seed = seed_deriver.seed_for(coordinate)
    lower = torch.tensor(
        parameter_spec.lower,
        device=target_physical.device,
        dtype=target_physical.dtype,
    )
    upper = torch.tensor(
        parameter_spec.upper,
        device=target_physical.device,
        dtype=target_physical.dtype,
    )
    engine = torch.quasirandom.SobolEngine(
        parameter_spec.dimension, scramble=True, seed=seed
    )
    unit = engine.draw(
        resolved_settings["raw_samples"], dtype=target_physical.dtype
    ).to(target_physical.device)
    raw = (lower + (upper - lower) * unit).contiguous()
    if not bool(torch.isfinite(raw).all()):
        raise error_type("no_valid_finite_raw_candidate")
    try:
        raw_physical, raw_scores = raw_score(
            surrogate, raw, transform, target_physical
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise error_type(
            f"no_valid_finite_raw_candidate: {type(exc).__name__}: {exc}"
        ) from exc
    finite = torch.isfinite(raw_scores)
    if not bool(finite.any()):
        raise error_type("no_valid_finite_raw_candidate")
    raw_completed = perf_counter()
    ranked = torch.argsort(
        torch.where(finite, raw_scores, torch.full_like(raw_scores, torch.inf)),
        stable=True,
    )
    ranked = ranked[finite[ranked]]
    histories: list[dict[str, Any]] = []
    accepted: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
    posterior_calls = 1
    for restart_index, raw_index_tensor in enumerate(
        ranked[: resolved_settings["restarts"]]
    ):
        restart_started = perf_counter()
        raw_index = int(raw_index_tensor.item())
        candidate = raw[raw_index].detach().clone().unsqueeze(0).requires_grad_(True)
        optimizer = torch.optim.Adam(
            [candidate], lr=resolved_settings["learning_rate"]
        )
        initial_score = float(raw_scores[raw_index].detach().cpu())
        previous = initial_score
        best_score = initial_score
        best_iteration: int | None = None
        best_theta: torch.Tensor | None = None
        best_physical: torch.Tensor | None = None
        status, iterations, failure = "max_iterations", 0, None
        try:
            for iteration in range(resolved_settings["max_iterations"]):
                optimizer.zero_grad(set_to_none=True)
                _, candidate_score = score(
                    surrogate, candidate, transform, target_physical
                )
                posterior_calls += 1
                if not bool(torch.isfinite(candidate_score)):
                    raise FloatingPointError("nonfinite_score")
                candidate_score.backward()
                if candidate.grad is None or not bool(
                    torch.isfinite(candidate.grad).all()
                ):
                    raise FloatingPointError("nonfinite_gradient")
                optimizer.step()
                with torch.no_grad():
                    candidate.clamp_(lower, upper)
                    current_physical, current_score_tensor = score(
                        surrogate,
                        candidate.detach(),
                        transform,
                        target_physical,
                    )
                    posterior_calls += 1
                if not bool(torch.isfinite(candidate).all()) or not bool(
                    torch.isfinite(current_score_tensor)
                ):
                    raise FloatingPointError("nonfinite_post_update_state")
                current = float(current_score_tensor.detach().cpu())
                iterations = iteration + 1
                if current <= best_score:
                    best_score = current
                    best_iteration = iterations
                    best_theta = candidate.detach()[0].clone()
                    best_physical = current_physical[0].detach().clone()
                if (
                    abs(previous - current)
                    <= resolved_settings["convergence_tolerance"]
                ):
                    status = "converged"
                    break
                previous = current
            if best_theta is None:
                status = "rejected_worsening_or_nonfinite"
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            status, failure = "failed", f"{type(exc).__name__}: {exc}"
        if best_theta is not None and best_physical is not None:
            accepted.append(
                (
                    best_score,
                    restart_index,
                    best_theta,
                    best_physical,
                )
            )
        histories.append(
            {
                "restart_index": restart_index,
                "raw_rank": restart_index,
                "raw_index": raw_index,
                "initial_score": initial_score,
                "iterations": iterations,
                "best_iteration": best_iteration,
                "best_score": best_score,
                "termination": status,
                "failure": failure,
                "elapsed_seconds": perf_counter() - restart_started,
            }
        )
    refinement_completed = perf_counter()
    # Accepted restarts compete with each other; the raw pool is only a fallback.
    if accepted:
        accepted.sort(key=lambda item: (item[0], item[1]))
        selected_score, selected_restart, selected_theta, selected_physical = (
            accepted[0]
        )
        source = "refined"
    else:
        best = int(ranked[0].item())
        selected_score = float(raw_scores[best].detach().cpu())
        selected_restart = None
        selected_theta = raw[best].detach().clone()
        selected_physical = raw_physical[best].detach().clone()
        source = "raw_fallback"
    return ContinuousPhysicalOptimization(
        selected_theta=selected_theta,
        selected_physical=selected_physical,
        selected_score=selected_score,
        returned_candidate_source=source,
        posterior_calls=posterior_calls,
        provenance={
            "physical_bounds": [
                list(parameter_spec.lower),
                list(parameter_spec.upper),
            ],
            "seed_namespace": coordinate.namespace,
            "seed_coordinates": list(coordinate.coordinates),
            "derived_seed": seed,
            "settings": resolved_settings,
            "raw_rankings": [int(item.item()) for item in ranked],
            "restart_histories": histories,
            "selected_restart_index": selected_restart,
            "elapsed_seconds": perf_counter() - started,
            "component_timings": {
                "raw_generation_and_scoring_seconds": raw_completed - started,
                "projected_refinement_seconds": (
                    refinement_completed - raw_completed
                ),
                "selection_and_serialization_seconds": (
                    perf_counter() - refinement_completed
                ),
            },
            "dtype": str(target_physical.dtype),
            "device": str(target_physical.device),
        },
    )
