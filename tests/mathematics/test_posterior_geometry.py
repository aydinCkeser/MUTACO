from __future__ import annotations

import pytest
import torch

from mutaco import (
    CovarianceValidationPolicy,
    FixedBeta,
    InvalidPosteriorError,
    JointChiSquareResponseDimensionBeta,
    LatentPosteriorSummary,
    StandardizedTarget,
    evaluate_confidence_geometry,
    structurally_symmetric_materialize,
)


DTYPE = torch.float64


def test_accepted_joint_beta_document_preserves_pinned_scalar_identity() -> None:
    document = {
        "id": "joint_chi_square_response_dimension_v1",
        "coverage": 0.95,
        "response_dimension": 20,
        "chi_square_quantile": 31.410432844230918,
        "resolved_beta": 5.604501123581913,
        "quantile_provenance": {
            "distribution": "chi_square",
            "degrees_of_freedom": 20,
            "probability": 0.95,
            "algorithm": "torch_gammainc_bisection_float64_v1",
            "iterations": 128,
            "computation_dtype": "float64",
        },
    }

    beta = JointChiSquareResponseDimensionBeta.from_document(
        document, expected_response_dimension=20
    )

    assert beta.document() == document


def target(values: list[float]) -> StandardizedTarget:
    return StandardizedTarget(
        torch.tensor(values, dtype=DTYPE), tuple(f"y{i}" for i in range(len(values))), "tx"
    )


def posterior(mean: torch.Tensor, covariance: torch.Tensor) -> LatentPosteriorSummary:
    return LatentPosteriorSummary(mean, covariance, "tx", "model")


def test_geometry_matches_hand_calculation_for_scalar_and_batch() -> None:
    scalar = posterior(
        torch.tensor([1.0, 2.0], dtype=DTYPE),
        torch.diag(torch.tensor([0.25, 1.0], dtype=DTYPE)),
    )
    result = evaluate_confidence_geometry(scalar, target([0.0, 0.0]))
    assert result.residual_squared.item() == 5.0
    assert result.radius_squared.item() == 4.0
    assert result.feasibility_score.item() == 1.0
    assert result.margin.item() == -0.25
    assert not result.feasible.item()

    batch = posterior(
        scalar.mean.unsqueeze(0).repeat(2, 1),
        scalar.covariance.unsqueeze(0).repeat(2, 1, 1),
    )
    batched = evaluate_confidence_geometry(batch, target([0.0, 0.0]))
    assert torch.equal(batched.residual_squared, result.residual_squared.repeat(2))


def test_zero_radius_boundary_is_exact() -> None:
    zero_covariance = torch.zeros((2, 2), dtype=DTYPE)
    on_target = evaluate_confidence_geometry(
        posterior(torch.zeros(2, dtype=DTYPE), zero_covariance), target([0.0, 0.0])
    )
    assert on_target.feasible.item()
    assert on_target.margin.item() == 0.0

    off_target = evaluate_confidence_geometry(
        posterior(torch.tensor([1.0, 0.0], dtype=DTYPE), zero_covariance),
        target([0.0, 0.0]),
    )
    assert not off_target.feasible.item()
    assert off_target.margin.item() == -torch.inf


def test_covariance_tolerances_accept_roundoff_but_reject_material_errors() -> None:
    nearly_symmetric = torch.tensor([[1.0, 2e-11], [0.0, -5e-11]], dtype=DTYPE)
    accepted = posterior(torch.zeros(2, dtype=DTYPE), nearly_symmetric)
    assert accepted.largest_eigenvalue.item() > 0.0
    assert torch.allclose(accepted.covariance, accepted.covariance.T)

    with pytest.raises(InvalidPosteriorError, match="symmetric"):
        posterior(
            torch.zeros(2, dtype=DTYPE),
            torch.tensor([[1.0, 1e-3], [0.0, 1.0]], dtype=DTYPE),
        )
    with pytest.raises(InvalidPosteriorError, match="positive semidefinite"):
        posterior(
            torch.zeros(2, dtype=DTYPE),
            torch.diag(torch.tensor([1.0, -1e-3], dtype=DTYPE)),
        )


def test_explicit_float32_covariance_tolerance_accepts_dtype_roundoff_only() -> None:
    mean = torch.zeros(2, dtype=torch.float32)
    roundoff = torch.tensor(
        [[0.03, 2.98e-8], [0.0, 0.01]], dtype=torch.float32
    )
    with pytest.raises(InvalidPosteriorError, match="max_asymmetry"):
        LatentPosteriorSummary(mean, roundoff, "transform", "model")
    accepted = LatentPosteriorSummary(
        mean, roundoff, "transform", "model",
        symmetry_tolerance=1e-6, psd_tolerance=1e-6,
    )
    assert torch.equal(accepted.covariance, accepted.covariance.T)
    with pytest.raises(InvalidPosteriorError, match="symmetric"):
        LatentPosteriorSummary(
            mean,
            torch.tensor([[0.03, 1e-3], [0.0, 0.01]], dtype=torch.float32),
            "transform", "model",
            symmetry_tolerance=1e-6, psd_tolerance=1e-6,
        )
    with pytest.raises(InvalidPosteriorError, match="positive semidefinite"):
        LatentPosteriorSummary(
            mean,
            torch.diag(torch.tensor([0.03, -1e-3], dtype=torch.float32)),
            "transform", "model",
            symmetry_tolerance=1e-6, psd_tolerance=1e-6,
        )


def test_graph_m20_replacement_tolerance_accepts_saved_roundoff_only() -> None:
    mean = torch.zeros(2, dtype=torch.float32)
    saved_maximum = 1.43051147e-6
    covariance = torch.tensor(
        [[1.0, saved_maximum], [0.0, 1.0]], dtype=torch.float32
    )
    with pytest.raises(InvalidPosteriorError, match="max_asymmetry"):
        LatentPosteriorSummary(
            mean, covariance, "transform", "model",
            symmetry_tolerance=1e-6, psd_tolerance=1e-6,
        )
    accepted = LatentPosteriorSummary(
        mean, covariance, "transform", "model",
        symmetry_tolerance=2e-6, psd_tolerance=1e-6,
    )
    assert torch.equal(accepted.covariance, accepted.covariance.T)
    with pytest.raises(InvalidPosteriorError, match="max_asymmetry"):
        LatentPosteriorSummary(
            mean,
            torch.tensor([[1.0, 2.1e-6], [0.0, 1.0]], dtype=torch.float32),
            "transform",
            "model",
            symmetry_tolerance=2e-6,
            psd_tolerance=1e-6,
        )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("response_dim", (2, 20))
@pytest.mark.parametrize("scale", (1e-4, 1.0, 10.0))
def test_scale_aware_covariance_policy_is_dtype_dimension_and_scale_aware(
    dtype: torch.dtype,
    response_dim: int,
    scale: float,
) -> None:
    eps = torch.finfo(dtype).eps
    limit = 8 * eps + 4 * response_dim * eps * scale
    covariance = torch.eye(response_dim, dtype=dtype) * scale
    covariance[0, 1] = 0.5 * limit
    accepted = LatentPosteriorSummary(
        torch.zeros(response_dim, dtype=dtype),
        covariance.contiguous(),
        "transform",
        "model",
        covariance_validation_policy=(
            CovarianceValidationPolicy.DTYPE_DIMENSION_SCALE_AWARE_V1
        ),
    )
    assert torch.equal(accepted.covariance, accepted.covariance.T)
    diagnostics = accepted.validation_diagnostics
    assert diagnostics["policy_id"] == "dtype_dimension_scale_aware_v1"
    assert diagnostics["response_dim"] == response_dim
    assert diagnostics["symmetry_absolute_epsilon_multiplier"] == 8
    assert diagnostics["symmetry_relative_dimension_epsilon_multiplier"] == 4
    assert diagnostics["raw_asymmetry_max"] <= diagnostics["symmetry_limit_max"]
    assert diagnostics["symmetry_exceedance_ratio_max"] <= 1.0


@pytest.mark.parametrize("scale", (1e-4, 1.0, 10.0))
def test_scale_aware_covariance_rejects_just_over_limit_and_material_errors(
    scale: float,
) -> None:
    response_dim = 20
    eps = torch.finfo(torch.float32).eps
    limit = 8 * eps + 4 * response_dim * eps * scale
    covariance = torch.eye(response_dim, dtype=torch.float32) * scale
    covariance[0, 1] = 1.25 * limit
    with pytest.raises(InvalidPosteriorError, match="effective_limit"):
        LatentPosteriorSummary(
            torch.zeros(response_dim, dtype=torch.float32),
            covariance.contiguous(),
            "transform",
            "model",
            covariance_validation_policy="dtype_dimension_scale_aware_v1",
        )
    material = torch.eye(response_dim, dtype=torch.float32) * scale
    material[0, 1] = max(1e-3, 1e-3 * scale)
    with pytest.raises(InvalidPosteriorError, match="not symmetric"):
        LatentPosteriorSummary(
            torch.zeros(response_dim, dtype=torch.float32),
            material.contiguous(),
            "transform",
            "model",
            covariance_validation_policy="dtype_dimension_scale_aware_v1",
        )


def test_scale_aware_policy_accepts_recorded_graph_roundoff_cases() -> None:
    cases = (
        (1.76218796, 1.07288361e-6),
        (1.00346017, 1.43051147e-6),
        (0.618507862, 1.01327896e-6),
        (7.65660334, 2.86102295e-6),
        (0.725953579, 2.62260437e-6),
        (2.07143021, 3.09944153e-6),
    )
    for scale, asymmetry in cases:
        covariance = torch.eye(20, dtype=torch.float32) * scale
        covariance[0, 1] = asymmetry
        summary = LatentPosteriorSummary(
            torch.zeros(20, dtype=torch.float32),
            covariance.contiguous(),
            "transform",
            "model",
            covariance_validation_policy="dtype_dimension_scale_aware_v1",
        )
        assert summary.validation_diagnostics["raw_asymmetry_max"] <= (
            summary.validation_diagnostics["symmetry_limit_max"]
        )


def test_scale_aware_psd_gate_accepts_roundoff_without_clipping_and_rejects_defect() -> None:
    eps = torch.finfo(torch.float32).eps
    response_dim = 20
    psd_limit = 8 * eps + 4 * response_dim * eps
    within = torch.eye(response_dim, dtype=torch.float32)
    within[-1, -1] = -0.5 * psd_limit
    accepted = LatentPosteriorSummary(
        torch.zeros(response_dim, dtype=torch.float32),
        within.contiguous(),
        "transform",
        "model",
        covariance_validation_policy="dtype_dimension_scale_aware_v1",
    )
    assert accepted.covariance[-1, -1] < 0
    outside = torch.eye(response_dim, dtype=torch.float32)
    outside[-1, -1] = -2.0 * psd_limit
    with pytest.raises(InvalidPosteriorError, match="positive semidefinite"):
        LatentPosteriorSummary(
            torch.zeros(response_dim, dtype=torch.float32),
            outside.contiguous(),
            "transform",
            "model",
            covariance_validation_policy="dtype_dimension_scale_aware_v1",
        )


def test_covariance_validation_policy_rejects_unknown_identifier() -> None:
    with pytest.raises(InvalidPosteriorError, match="unknown covariance"):
        LatentPosteriorSummary(
            torch.zeros(2),
            torch.eye(2),
            "transform",
            "model",
            covariance_validation_policy="guess_from_dtype",
        )


def test_batched_covariance_failure_reports_one_consistent_worst_matrix() -> None:
    covariance = torch.stack(
        (
            torch.tensor([[10.0, 3e-6], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.01, 3e-6], [0.0, 0.01]], dtype=torch.float32),
        )
    ).contiguous()
    with pytest.raises(InvalidPosteriorError) as captured:
        LatentPosteriorSummary(
            torch.zeros((2, 2), dtype=torch.float32),
            covariance,
            "transform",
            "model",
            covariance_validation_policy="dtype_dimension_scale_aware_v1",
        )
    message = str(captured.value)
    assert "worst_batch_flat_index=1" in message
    assert "max_abs_covariance=0.00999999978" in message
    assert "exceedance_ratio=" in message


def test_structural_materialization_is_exact_recorded_and_noninvasive() -> None:
    raw = torch.tensor(
        [[1.0, 0.200002], [0.2, 0.5]], dtype=torch.float32
    ).contiguous()
    expected = (0.5 * (raw + raw.T)).contiguous()
    materialized, diagnostics = structurally_symmetric_materialize(raw)
    assert torch.equal(materialized, materialized.T)
    assert torch.equal(materialized, expected)
    assert diagnostics["raw_asymmetry_max"] > 0
    assert diagnostics["exact_output_symmetry"] is True
    assert diagnostics["jitter_added"] is False
    summary = LatentPosteriorSummary(
        torch.zeros(2, dtype=torch.float32),
        materialized,
        "transform",
        "model",
        covariance_validation_policy="structural_symmetric_materialization_v1",
        structural_materialization_diagnostics=diagnostics,
    )
    assert torch.equal(summary.covariance, expected)
    assert summary.validation_diagnostics["structural_materialization"] == diagnostics


def test_structural_materialization_rejects_material_asymmetry_and_non_psd() -> None:
    with pytest.raises(InvalidPosteriorError, match="materially asymmetric"):
        structurally_symmetric_materialize(
            torch.tensor([[1.0, 0.01], [0.0, 1.0]], dtype=torch.float32)
        )
    with pytest.raises(InvalidPosteriorError, match="positive semidefinite"):
        structurally_symmetric_materialize(
            torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)
        )
    with pytest.raises(InvalidPosteriorError, match="requires raw materialization"):
        LatentPosteriorSummary(
            torch.zeros(2, dtype=torch.float32),
            torch.eye(2, dtype=torch.float32),
            "transform",
            "model",
            covariance_validation_policy="structural_symmetric_materialization_v1",
        )


def test_structural_materialization_preserves_symmetric_acquisition_order() -> None:
    symmetric = torch.stack(
        (
            torch.diag(torch.tensor([0.2, 0.5], dtype=torch.float32)),
            torch.diag(torch.tensor([0.3, 0.8], dtype=torch.float32)),
        )
    )
    antisymmetric = torch.tensor(
        [[[0.0, 1e-6], [-1e-6, 0.0]], [[0.0, -1e-6], [1e-6, 0.0]]],
        dtype=torch.float32,
    )
    raw = (symmetric + antisymmetric).contiguous()
    materialized, _ = structurally_symmetric_materialize(raw)
    before = torch.linalg.eigvalsh(0.5 * (raw + raw.transpose(-1, -2)))[..., -1]
    after = torch.linalg.eigvalsh(materialized)[..., -1]
    assert torch.equal(before, after)
    assert torch.equal(torch.argsort(before), torch.argsort(after))


def test_beta_is_fixed_positive_heuristic_with_provenance() -> None:
    assert FixedBeta().provenance() == {
        "policy_id": "fixed_beta",
        "value": 2.0,
        "guarantee": None,
        "interpretation": "confidence_radius_multiplier",
    }
    with pytest.raises(ValueError, match="positive"):
        FixedBeta(0.0)
    with pytest.raises(ValueError, match="unknown"):
        FixedBeta(policy_id="scheduled_beta")
    assert FixedBeta(policy_id="fixed_beta_v1").provenance()["policy_id"] == (
        "fixed_beta_v1"
    )


def test_trace_effective_dimension_geometry_uses_beta_squared_trace() -> None:
    summary = posterior(
        torch.tensor([1.0, 2.0], dtype=DTYPE),
        torch.diag(torch.tensor([0.25, 1.0], dtype=DTYPE)),
    )
    result = evaluate_confidence_geometry(
        summary,
        target([0.0, 0.0]),
        beta=FixedBeta(2.0),
        radius_policy="trace_effective_dimension_radius_diagnostic_v1",
    )
    assert torch.equal(result.radius_squared, 4.0 * summary.trace)
    assert torch.allclose(
        result.effective_dimension,
        summary.trace / summary.largest_eigenvalue,
    )
