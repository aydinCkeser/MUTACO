from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mutaco import (
    EvaluatedDesigns,
    LatentPosteriorMeanSummary,
    LatentPosteriorSummary,
    MathematicsPolicyConfig,
    StandardizedTarget,
    descriptor_for,
    recommend_evaluated,
    render_policy_support_markdown,
)


DTYPE = torch.float64
TARGET = StandardizedTarget(torch.zeros(2, dtype=DTYPE), ("a", "b"), "tx")


def cases(parameter_dim: int = 2) -> EvaluatedDesigns:
    return EvaluatedDesigns(
        torch.arange(3 * parameter_dim, dtype=DTYPE).reshape(3, parameter_dim),
        ("eval-0", "eval-1", "eval-2"),
    )


def summary(means: list[list[float]], variances: list[list[float]]) -> LatentPosteriorSummary:
    return LatentPosteriorSummary(
        torch.tensor(means, dtype=DTYPE),
        torch.diag_embed(torch.tensor(variances, dtype=DTYPE)),
        "tx",
        "final-model",
    )


def test_default_l1_and_optional_expected_l2_are_distinct_and_call_no_simulator() -> None:
    posterior = summary(
        [[0.9, 0.0], [0.5, 0.5], [2.0, 2.0]],
        [[10.0, 10.0], [0.0, 0.0], [0.0, 0.0]],
    )
    default = recommend_evaluated(cases(), posterior, TARGET)
    optional = recommend_evaluated(
        cases(), posterior, TARGET, policy="evaluated_posterior_expected_l2"
    )
    assert default.selected_index == 0
    assert optional.selected_index == 1
    assert default.simulator_calls == optional.simulator_calls == 0
    assert optional.score.item() == 0.5


def test_default_l1_accepts_mean_only_but_expected_l2_requires_covariance() -> None:
    mean_only = LatentPosteriorMeanSummary(
        torch.tensor([[0.9, 0.0], [0.5, 0.5], [2.0, 2.0]], dtype=DTYPE),
        "tx",
        "final-model",
    )
    assert recommend_evaluated(cases(), mean_only, TARGET).selected_index == 0
    with pytest.raises(ValueError, match="requires posterior covariance"):
        recommend_evaluated(
            cases(),
            mean_only,
            TARGET,
            policy="evaluated_posterior_expected_l2",
        )


def test_exact_ties_choose_earliest_but_near_ties_remain_distinct() -> None:
    equal = summary([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], [[0.0, 0.0]] * 3)
    assert recommend_evaluated(cases(), equal, TARGET).selected_index == 0

    near = summary(
        [[1.0 + 1e-13, 0.0], [1.0, 0.0], [2.0, 2.0]],
        [[0.0, 0.0]] * 3,
    )
    assert recommend_evaluated(cases(), near, TARGET).selected_index == 1


@pytest.mark.parametrize("parameter_dim", [2, 6])
def test_recommendation_preserves_parameter_dimension(parameter_dim: int) -> None:
    result = recommend_evaluated(
        cases(parameter_dim),
        summary([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [[0.0, 0.0]] * 3),
        TARGET,
    )
    assert result.selected_theta.shape == (parameter_dim,)


def test_registry_defaults_reject_unknown_ids_and_generate_documentation() -> None:
    config = MathematicsPolicyConfig()
    assert config.provenance() == {
        "exploration": "operator_norm_uncertainty",
        "recommendation": "final_posterior_standardized_l1",
    }
    assert descriptor_for("trace_marginal_variance").classification.value == "diagnostic-only"
    with pytest.raises(ValueError, match="unknown"):
        MathematicsPolicyConfig(exploration="not-a-policy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown"):
        descriptor_for("not-a-policy")

    documented = (
        Path(__file__).parents[2] / "scientific_contracts" / "mutaco_d_contract" / "POLICY_SUPPORT.md"
    ).read_text(encoding="utf-8")
    assert documented == render_policy_support_markdown()
