from __future__ import annotations

import torch

from experiments.paper.common.terminal_domain_ablation import (
    NEAR_EQUALITY_TOLERANCE,
    evaluated_posterior_mean_physical_mae,
    normalized_distances,
)
from mutaco import FrozenResponseTransform, ResponseAxis, ResponseSpec


class _Posterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean


class _Surrogate:
    def posterior_mean(self, theta: torch.Tensor) -> _Posterior:
        return _Posterior(theta[:, :1].contiguous())


def _transform() -> FrozenResponseTransform:
    response = torch.tensor([[0.0], [2.0]], dtype=torch.float32)
    return FrozenResponseTransform.fit(
        response,
        response_spec=ResponseSpec((ResponseAxis("response"),)),
        observation_ids=("a", "b"),
        scale_floor=1e-12,
    )


def test_evaluated_policy_uses_physical_mae_and_earliest_tie() -> None:
    theta = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    result = evaluated_posterior_mean_physical_mae(
        surrogate=_Surrogate(),
        evaluated_theta=theta,
        transform=_transform(),
        target_physical=torch.tensor([1.0], dtype=torch.float32),
    )
    assert result["selected_evaluation_index"] == 0
    assert result["selected_theta"] == [0.0]
    assert result["simulator_calls"] == 0


def test_normalized_distances_separate_exact_and_near_equality() -> None:
    lower = torch.tensor([0.0, 0.0])
    upper = torch.tensor([2.0, 4.0])
    b = torch.tensor([1.0, 2.0])
    c = b + torch.tensor([NEAR_EQUALITY_TOLERANCE, 0.0])
    result = normalized_distances(
        b,
        c,
        torch.stack((b, torch.tensor([0.0, 0.0]))),
        lower=lower,
        upper=upper,
    )
    assert not result["exact_theta_equal"]
    assert result["near_theta_equal"]
    assert result["c_to_nearest_evaluated_l2"] > 0.0
