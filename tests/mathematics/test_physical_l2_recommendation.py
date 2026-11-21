"""Core contracts for the promoted mean-only physical-L2 policies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mutaco import (
    FrozenResponseTransform,
    ParameterAxis,
    ParameterSpec,
    ResponseAxis,
    ResponseSpec,
    ScientificIdentity,
    SeedDeriver,
    descriptor_for,
    physical_l2_scores,
    recommend_continuous_physical_l2,
    recommend_evaluated_physical_l2,
)
from mutaco.recommendation._continuous import optimize_continuous_physical


ROOT = Path(__file__).resolve().parents[2]


def test_continuous_physical_policies_share_one_internal_optimizer() -> None:
    recommendation = ROOT / "src/mutaco/recommendation"
    l2_source = (recommendation / "comparators/physical_squared_l2_recommendation.py").read_text(encoding="utf-8")
    mae_source = (recommendation / "physical_mae_terminal_recommendation.py").read_text(encoding="utf-8")
    shared_source = (recommendation / "_continuous_terminal_optimizer.py").read_text(encoding="utf-8")
    assert "optimize_continuous_physical(" in l2_source
    assert "optimize_continuous_physical(" in mae_source
    assert "torch.optim.Adam" not in l2_source
    assert "torch.optim.Adam" not in mae_source
    assert "torch.optim.Adam" in shared_source


class MeanOnlySurface:
    model_id = "core-mean-only-test-v1"

    def __init__(self, *, detach: bool = False, nonfinite: bool = False) -> None:
        self.calls = 0
        self.detach = detach
        self.nonfinite = nonfinite

    def posterior_mean(self, theta: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        mean = torch.stack((theta[..., 0] + theta[..., 1], theta[..., 0] - theta[..., 1]), dim=-1)
        if self.detach:
            mean = mean.detach()
        if self.nonfinite:
            mean = torch.full_like(mean, torch.nan)
        return SimpleNamespace(mean=mean)

    def posterior(self, theta: torch.Tensor) -> None:  # pragma: no cover
        raise AssertionError("physical-L2 policies must not materialize covariance")


@pytest.fixture
def transform() -> FrozenResponseTransform:
    return FrozenResponseTransform.fit(
        torch.tensor([[0.0, 0.0], [2.0, 4.0]], dtype=torch.float32),
        response_spec=ResponseSpec((ResponseAxis("y0"), ResponseAxis("y1"))),
        observation_ids=("a", "b"),
    )


def spec() -> ParameterSpec:
    return ParameterSpec((ParameterAxis("x", 0.0, 1.0), ParameterAxis("z", 0.0, 1.0)))


def deriver() -> SeedDeriver:
    return SeedDeriver(ScientificIdentity.create({"test": "physical-l2"}), 8214)


def test_core_registry_and_shared_scalar_batched_reordered_score(transform: FrozenResponseTransform) -> None:
    for identifier in (
        "evaluated_posterior_mean_physical_l2",
        "continuous_posterior_mean_physical_l2",
    ):
        assert descriptor_for(identifier).classification.value == "approved optional"
    mean = torch.tensor([[0.2, -0.1], [0.8, 0.3], [-0.4, 0.9]], dtype=torch.float32)
    target = torch.tensor([1.4, 1.1], dtype=torch.float32)
    _, batch = physical_l2_scores(mean, transform=transform, target_physical=target)
    scalar = torch.stack([physical_l2_scores(row, transform=transform, target_physical=target)[1] for row in mean])
    order = torch.tensor([2, 0, 1])
    _, reordered = physical_l2_scores(mean[order].contiguous(), transform=transform, target_physical=target)
    assert torch.equal(batch, scalar)
    assert torch.equal(reordered, batch[order])


def test_evaluated_and_continuous_are_mean_only_zero_call_and_rng_neutral(transform: FrozenResponseTransform) -> None:
    theta = torch.tensor([[0.2, 0.4], [0.2, 0.4], [0.9, 0.1]], dtype=torch.float32)
    evaluated = recommend_evaluated_physical_l2(
        surrogate=MeanOnlySurface(), transform=transform,
        target_physical=torch.tensor([1.0, 1.0]), target_id="target",
        theta=theta, observation_ids=("first", "second", "third"),
    )
    assert evaluated.selected_evaluation_index == 0
    assert evaluated.simulator_calls == evaluated.gp_fits == 0
    arguments = dict(
        transform=transform, target_physical=torch.tensor([1.3, 1.1]),
        target_id="target", parameter_spec=spec(), seed_deriver=deriver(), target_index=7,
    )
    torch.manual_seed(1)
    first = recommend_continuous_physical_l2(surrogate=MeanOnlySurface(), **arguments)
    torch.manual_seed(999)
    second = recommend_continuous_physical_l2(surrogate=MeanOnlySurface(), **arguments)
    assert torch.equal(first.selected_theta, second.selected_theta)
    assert first.simulator_calls == first.gp_fits == 0
    assert bool(((first.selected_theta >= 0.0) & (first.selected_theta <= 1.0)).all())


def test_continuous_raw_fallback_and_typed_no_finite_failure(transform: FrozenResponseTransform) -> None:
    arguments = dict(
        transform=transform, target_physical=torch.tensor([1.0, 1.0]), target_id="target",
        parameter_spec=spec(), seed_deriver=deriver(), target_index=3,
    )
    assert recommend_continuous_physical_l2(
        surrogate=MeanOnlySurface(detach=True), **arguments
    ).returned_candidate_source == "raw_fallback"
    with pytest.raises(RuntimeError, match="no_valid_finite_raw_candidate"):
        recommend_continuous_physical_l2(surrogate=MeanOnlySurface(nonfinite=True), **arguments)


def _quadratic_score(
    surrogate: MeanOnlySurface,
    theta: torch.Tensor,
    transform: FrozenResponseTransform,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del surrogate, transform
    physical = theta.clone()
    return physical, (theta[:, 0] - target[0]).square()


def _terminal_optimization(
    transform: FrozenResponseTransform,
    *,
    target: float,
    learning_rate: float,
    max_iterations: int,
    convergence_tolerance: float = 0.0,
):
    return optimize_continuous_physical(
        surrogate=MeanOnlySurface(),
        transform=transform,
        target_physical=torch.tensor([target], dtype=torch.float32),
        parameter_spec=ParameterSpec((ParameterAxis("x", 0.0, 1.0),)),
        seed_deriver=deriver(),
        target_index=11,
        policy_id="terminal-regression-v1",
        settings={
            "raw_samples": 8,
            "restarts": 1,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "optimizer": "projected_adam",
            "convergence_tolerance": convergence_tolerance,
        },
        score=_quadratic_score,
        raw_score=_quadratic_score,
        error_type=RuntimeError,
    )


def test_continuous_convergence_compares_distinct_post_update_states(
    transform: FrozenResponseTransform,
) -> None:
    result = _terminal_optimization(
        transform, target=0.8, learning_rate=0.05, max_iterations=8
    )
    history = result.provenance["restart_histories"][0]
    assert history["iterations"] > 1
    assert history["best_iteration"] is not None
    assert result.selected_score <= history["initial_score"]


def test_continuous_already_optimal_state_can_converge_after_one_update(
    transform: FrozenResponseTransform,
) -> None:
    def flat_score(
        surrogate: MeanOnlySurface,
        theta: torch.Tensor,
        transform: FrozenResponseTransform,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del surrogate, transform, target
        return theta.clone(), (theta[:, 0] * 0.0)

    result = optimize_continuous_physical(
        surrogate=MeanOnlySurface(),
        transform=transform,
        target_physical=torch.tensor([0.0], dtype=torch.float32),
        parameter_spec=ParameterSpec((ParameterAxis("x", 0.0, 1.0),)),
        seed_deriver=deriver(),
        target_index=12,
        policy_id="terminal-flat-regression-v1",
        settings={
            "raw_samples": 8,
            "restarts": 1,
            "max_iterations": 8,
            "learning_rate": 0.05,
            "optimizer": "projected_adam",
            "convergence_tolerance": 0.0,
        },
        score=flat_score,
        raw_score=flat_score,
        error_type=RuntimeError,
    )
    history = result.provenance["restart_histories"][0]
    assert history["iterations"] == 1
    assert history["termination"] == "converged"


def test_continuous_worsening_step_preserves_raw_best_state(
    transform: FrozenResponseTransform,
) -> None:
    result = _terminal_optimization(
        transform, target=0.5, learning_rate=10.0, max_iterations=1
    )
    history = result.provenance["restart_histories"][0]
    assert history["best_iteration"] is None
    assert result.returned_candidate_source == "raw_fallback"
    assert result.selected_score == history["initial_score"]


def test_continuous_later_worsening_preserves_best_refined_state(
    transform: FrozenResponseTransform,
) -> None:
    calls = 0

    def raw_score(
        surrogate: MeanOnlySurface,
        theta: torch.Tensor,
        transform: FrozenResponseTransform,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del surrogate, transform, target
        return theta.clone(), torch.ones(theta.shape[0], dtype=theta.dtype)

    def scripted_score(
        surrogate: MeanOnlySurface,
        theta: torch.Tensor,
        transform: FrozenResponseTransform,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        del surrogate, transform, target
        calls += 1
        if calls % 2:
            return theta.clone(), theta[:, 0]
        value = 0.1 if calls == 2 else 0.5
        return theta.clone(), torch.full(
            (theta.shape[0],), value, dtype=theta.dtype, device=theta.device
        )

    result = optimize_continuous_physical(
        surrogate=MeanOnlySurface(),
        transform=transform,
        target_physical=torch.tensor([0.0], dtype=torch.float32),
        parameter_spec=ParameterSpec((ParameterAxis("x", 0.0, 1.0),)),
        seed_deriver=deriver(),
        target_index=13,
        policy_id="terminal-best-state-regression-v1",
        settings={
            "raw_samples": 8,
            "restarts": 1,
            "max_iterations": 2,
            "learning_rate": 0.05,
            "optimizer": "projected_adam",
            "convergence_tolerance": 0.0,
        },
        score=scripted_score,
        raw_score=raw_score,
        error_type=RuntimeError,
    )
    history = result.provenance["restart_histories"][0]
    assert history["iterations"] == 2
    assert history["best_iteration"] == 1
    assert result.returned_candidate_source == "refined"
    assert result.selected_score == pytest.approx(0.1)
