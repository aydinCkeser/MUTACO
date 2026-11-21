from __future__ import annotations

import torch

from mutaco import (
    EvaluatedDesigns,
    recommend_best_observed_physical_mae,
)


def test_best_observed_physical_mae_is_zero_call_and_earliest_tie() -> None:
    designs = EvaluatedDesigns(
        torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64).contiguous(),
        ("first", "second", "third"),
    )
    responses = torch.tensor(
        [[0.0, 1.0], [0.4, 0.6], [0.6, 0.4]], dtype=torch.float64
    ).contiguous()
    target = torch.tensor([0.5, 0.5], dtype=torch.float64).contiguous()
    result = recommend_best_observed_physical_mae(designs, responses, target)
    assert result.policy_id == "evaluated_best_observed_physical_mae_v1"
    assert result.selected_index == 1
    assert result.selected_observation_id == "second"
    assert result.score_semantics == "observed_physical_response_mae"
    assert result.tie_rule == "earliest_evaluation"
    assert result.simulator_calls == result.gp_fits == 0
