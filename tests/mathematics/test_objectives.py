from __future__ import annotations

import pytest
import torch

from mutaco import (
    LatentPosteriorSummary,
    MatchedBarrierObjective,
    StandardizedTarget,
    evaluate_exploration,
)


DTYPE = torch.float64
TARGET = StandardizedTarget(torch.zeros(2, dtype=DTYPE), ("a", "b"), "tx")


def summary(mean: torch.Tensor, covariance: torch.Tensor) -> LatentPosteriorSummary:
    return LatentPosteriorSummary(mean, covariance, "tx", "model")


def test_matched_barrier_uses_unsquared_standardized_l2_and_strict_domain() -> None:
    item = summary(
        torch.tensor([0.3, 0.4], dtype=DTYPE),
        torch.diag(torch.tensor([1.0, 0.25], dtype=DTYPE)),
    )
    result = MatchedBarrierObjective(alpha=2.0, tau_interior=0.0).evaluate(item, TARGET)
    margin = (4.0 - 0.25) / 4.0
    expected = 0.5 - 2.0 * torch.log(torch.tensor(margin, dtype=DTYPE))
    assert torch.allclose(result.values, expected)
    assert result.exploitation.item() == 0.5
    assert result.valid_domain.item()

    boundary = summary(torch.tensor([2.0, 0.0], dtype=DTYPE), torch.eye(2, dtype=DTYPE))
    boundary_result = MatchedBarrierObjective(alpha=1.0, tau_interior=0.0).evaluate(
        boundary, TARGET
    )
    assert not boundary_result.valid_domain.item()
    assert boundary_result.values.item() == torch.inf


def test_tau_is_a_strict_interior_threshold_without_boundary_clamp() -> None:
    item = summary(torch.tensor([1.0, 0.0], dtype=DTYPE), torch.eye(2, dtype=DTYPE))
    assert MatchedBarrierObjective(alpha=1.0, tau_interior=0.74).evaluate(
        item, TARGET
    ).valid_domain.item()
    assert not MatchedBarrierObjective(alpha=1.0, tau_interior=0.75).evaluate(
        item, TARGET
    ).valid_domain.item()


def test_exploration_policies_are_discriminating_and_unknown_is_rejected() -> None:
    item = summary(
        torch.zeros(2, dtype=DTYPE),
        torch.diag(torch.tensor([3.0, 2.0], dtype=DTYPE)),
    )
    operator = evaluate_exploration(item, "operator_norm_uncertainty")
    trace = evaluate_exploration(item, "trace_marginal_variance")
    assert operator.values.item() == 3.0
    assert trace.values.item() == 5.0
    assert operator.policy_id == "operator_norm_uncertainty"
    assert operator.classification == "D default"
    assert operator.direction == "maximize"
    assert operator.simulator_calls == 0
    with pytest.raises(ValueError, match="unknown"):
        evaluate_exploration(item, "silent-fallback")


def test_barrier_gradient_matches_finite_difference() -> None:
    mean = torch.tensor([0.2, 0.1], dtype=DTYPE, requires_grad=True)
    covariance = torch.diag(torch.tensor([1.0, 0.5], dtype=DTYPE))
    objective = MatchedBarrierObjective(alpha=0.7, tau_interior=0.0)
    value = objective.evaluate(summary(mean, covariance), TARGET).values
    (gradient,) = torch.autograd.grad(value, mean)

    epsilon = 1e-6
    plus = mean.detach().clone()
    minus = mean.detach().clone()
    plus[0] += epsilon
    minus[0] -= epsilon
    finite_difference = (
        objective.evaluate(summary(plus, covariance), TARGET).values
        - objective.evaluate(summary(minus, covariance), TARGET).values
    ) / (2 * epsilon)
    assert torch.allclose(gradient[0], finite_difference, rtol=1e-6, atol=1e-7)
