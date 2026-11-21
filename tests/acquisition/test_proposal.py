from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mutaco import (
    AcquisitionBranch,
    BranchReason,
    FallbackReason,
    GPFitConfig,
    GPFitStatus,
    LatentPosteriorSummary,
    ParameterAxis,
    ParameterSpec,
    ProposalConfig,
    ProposalFailureReason,
    ProposalStatus,
    RawCandidatePolicy,
    RestartDiagnostic,
    RestartStatus,
    RestartStopReason,
    ReturnedCandidateSource,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    StandardizedTarget,
    SurrogateTrainingData,
    candidate_policy_support_markdown,
    fit_exact_multitask_gp,
    generate_candidate_pool,
    propose_candidate,
    recover_best_ranked_unevaluated_raw,
)


DTYPE = torch.float64


def parameter_spec(dimension: int, lower: float = 0.0, upper: float = 1.0) -> ParameterSpec:
    return ParameterSpec(
        tuple(
            ParameterAxis(f"theta_{index}", lower, upper)
            for index in range(dimension)
        )
    )


def config(**settings: object) -> ProposalConfig:
    values: dict[str, object] = {
        "raw_samples": 16,
        "restarts": 3,
        "max_iterations": 8,
    }
    values.update(settings)
    return ProposalConfig.from_seed_deriver(
        SeedDeriver(ScientificIdentity.create({"experiment": "proposal-test"}), 193),
        SeedCoordinate("raw_candidates", ("round", 0)),
        **values,
    )


class SyntheticSurrogate:
    def __init__(
        self,
        dimension: int = 2,
        *,
        mode: str = "barrier",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model_id = f"synthetic-{mode}"
        self.transform_id = "transform-test"
        self.parameter_dim = dimension
        self.dtype = DTYPE
        self.device = device
        self.mode = mode
        self.call_shapes: list[tuple[int, ...]] = []

    def posterior(self, theta: torch.Tensor) -> LatentPosteriorSummary:
        self.call_shapes.append(tuple(theta.shape))
        scalar = theta.ndim == 1
        batch = theta.unsqueeze(0) if scalar else theta
        if self.mode == "raise" or (self.mode == "scalar_raise" and scalar):
            raise RuntimeError("deliberate posterior failure")
        if self.mode == "barrier":
            mean = batch[..., :2] - 0.5
            variance = torch.full(
                (batch.shape[0], 2), 0.05, dtype=self.dtype, device=self.device
            )
        elif self.mode == "exploration":
            coordinate = batch.sum(dim=-1)
            mean = torch.ones(
                (batch.shape[0], 2), dtype=self.dtype, device=self.device
            ) * (10.0 + 0.0 * coordinate.unsqueeze(-1))
            variance = torch.stack(
                (1.0 + coordinate, 0.5 + 0.25 * coordinate), dim=-1
            )
        elif self.mode == "ranking":
            high = batch[:, 0] >= 0.5
            first = torch.where(
                high,
                torch.full_like(batch[:, 0], 2.0),
                torch.full_like(batch[:, 0], 3.0),
            )
            second = torch.where(
                high,
                torch.full_like(batch[:, 0], 2.0),
                torch.zeros_like(batch[:, 0]),
            )
            mean = torch.ones(
                (batch.shape[0], 2), dtype=self.dtype, device=self.device
            ) * 10.0
            variance = torch.stack((first, second), dim=-1)
        elif self.mode in {"tie", "scalar_raise"}:
            coordinate = batch.sum(dim=-1)
            mean = 10.0 + torch.zeros(
                (batch.shape[0], 2), dtype=self.dtype, device=self.device
            ) * coordinate.unsqueeze(-1)
            variance = 1.0 + torch.zeros(
                (batch.shape[0], 2), dtype=self.dtype, device=self.device
            ) * coordinate.unsqueeze(-1)
        else:
            raise AssertionError(self.mode)
        covariance = torch.diag_embed(variance).contiguous()
        mean = mean.contiguous()
        if scalar:
            mean = mean.squeeze(0).contiguous()
            covariance = covariance.squeeze(0).contiguous()
        return LatentPosteriorSummary(
            mean,
            covariance,
            self.transform_id,
            self.model_id,
        )


class NonfiniteScorePosterior:
    def __init__(self, count: int) -> None:
        self.mean = torch.full((count, 2), 10.0, dtype=DTYPE)
        self.response_dim = 2
        self.transform_id = "transform-test"
        self.model_id = "synthetic-nonfinite"
        self.largest_eigenvalue = torch.full((count,), torch.nan, dtype=DTYPE)
        self.trace = torch.full((count,), torch.nan, dtype=DTYPE)


class NonfiniteScoreSurrogate(SyntheticSurrogate):
    def __init__(self) -> None:
        super().__init__(mode="tie")
        self.model_id = "synthetic-nonfinite"

    def posterior(self, theta: torch.Tensor) -> NonfiniteScorePosterior:  # type: ignore[override]
        self.call_shapes.append(tuple(theta.shape))
        return NonfiniteScorePosterior(theta.shape[0])


def target(device: torch.device = torch.device("cpu")) -> StandardizedTarget:
    return StandardizedTarget(
        torch.zeros(2, dtype=DTYPE, device=device),
        ("response_0", "response_1"),
        "transform-test",
    )


@pytest.mark.parametrize("dimension", [2, 6])
def test_candidate_pool_is_deterministic_bounded_dimension_generic_and_rng_neutral(
    dimension: int,
) -> None:
    before = torch.random.get_rng_state().clone()
    first = generate_candidate_pool(
        parameter_spec=parameter_spec(dimension, -2.0, 3.0),
        config=config(),
        dtype=DTYPE,
        device=torch.device("cpu"),
    )
    second = generate_candidate_pool(
        parameter_spec=parameter_spec(dimension, -2.0, 3.0),
        config=config(),
        dtype=DTYPE,
        device=torch.device("cpu"),
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(first.theta, second.theta)
    assert first.pool_id == second.pool_id
    assert first.theta.shape == (16, dimension)
    assert bool(((first.theta >= -2.0) & (first.theta <= 3.0)).all())
    assert first.generation_device == "cpu"
    assert first.provenance()["bounds"] == (
        tuple(-2.0 for _ in range(dimension)),
        tuple(3.0 for _ in range(dimension)),
    )


def test_raw_policies_and_seeds_are_discriminating_and_unknowns_fail() -> None:
    spec = parameter_spec(2)
    sobol = generate_candidate_pool(
        parameter_spec=spec,
        config=config(),
        dtype=DTYPE,
        device=torch.device("cpu"),
    )
    iid = generate_candidate_pool(
        parameter_spec=spec,
        config=config(raw_policy=RawCandidatePolicy.IID_UNIFORM),
        dtype=DTYPE,
        device=torch.device("cpu"),
    )
    other_seed = generate_candidate_pool(
        parameter_spec=spec,
        config=ProposalConfig(
            candidate_seed=config().candidate_seed + 1,
            seed_coordinate=SeedCoordinate("raw_candidates", ("round", 1)),
            raw_samples=16,
        ),
        dtype=DTYPE,
        device=torch.device("cpu"),
    )
    assert not torch.equal(sobol.theta, iid.theta)
    assert not torch.equal(sobol.theta, other_seed.theta)
    assert iid.classification == "diagnostic-only"
    with pytest.raises(ValueError, match="unknown raw-candidate"):
        ProposalConfig(
            candidate_seed=1,
            seed_coordinate=SeedCoordinate("raw_candidates"),
            raw_policy="latin_hypercube",  # type: ignore[arg-type]
        )


def test_barrier_branch_refines_with_backtracking_and_separate_call_accounting() -> None:
    surrogate = SyntheticSurrogate(mode="barrier")
    result = propose_candidate(
        surrogate,
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]], dtype=DTYPE
        ),
        config=config(learning_rate=5.0, max_backtracks=8),
    )
    assert result.status is ProposalStatus.SUCCESS
    assert result.branch is AcquisitionBranch.BARRIER
    assert result.candidate is not None
    assert bool(((result.candidate >= 0.0) & (result.candidate <= 1.0)).all())
    assert result.posterior_calls.raw_pool == 1
    assert result.posterior_calls.total == len(surrogate.call_shapes)
    assert result.posterior_calls.total == 1 + sum(result.posterior_calls.per_restart)
    assert surrogate.call_shapes[0] == (16, 2)
    assert any(item.backtracks > 0 for item in result.restarts)
    assert result.simulator_calls == 0
    assert result.branch_direction == "minimize"
    assert result.raw_counts["strict_interior_finite"] > 0  # type: ignore[operator]


def test_barrier_backtrack_exhaustion_retains_the_valid_raw_candidate() -> None:
    result = propose_candidate(
        SyntheticSurrogate(mode="barrier"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(
            raw_samples=64,
            restarts=1,
            max_iterations=2,
            learning_rate=100.0,
            max_backtracks=0,
        ),
    )
    assert result.status is ProposalStatus.SUCCESS
    assert result.branch is AcquisitionBranch.BARRIER
    assert result.restarts[0].stop_reason is RestartStopReason.BACKTRACK_EXHAUSTED
    assert result.candidate_source is ReturnedCandidateSource.REFINED
    assert result.fallback_reason is None
    raw_index = result.raw_rankings[0].raw_index
    assert torch.equal(result.candidate, result.pool.theta[raw_index])


def test_exploration_branch_refines_and_reports_exact_duplicate_without_rejecting() -> None:
    surrogate = SyntheticSurrogate(mode="exploration")
    proposal_config = config(learning_rate=5.0, max_iterations=3)
    evaluated = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]], dtype=DTYPE
    )
    result = propose_candidate(
        surrogate,
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=evaluated,
        config=proposal_config,
    )
    assert result.status is ProposalStatus.SUCCESS
    assert result.branch is AcquisitionBranch.EXPLORATION
    assert result.candidate_source is ReturnedCandidateSource.REFINED
    assert torch.equal(result.candidate, torch.ones(2, dtype=DTYPE))
    assert result.duplicate is not None
    assert result.duplicate.exact_duplicate
    assert result.duplicate.exact_duplicate_indices == (1,)
    assert result.duplicate.nearest_training_distance == 0.0
    assert result.duplicate.tolerance is None
    assert result.branch_direction == "maximize"

    recovered = recover_best_ranked_unevaluated_raw(
        result,
        evaluated_theta=evaluated,
        parameter_spec=parameter_spec(2),
        config=proposal_config,
        policy_id="best_ranked_unevaluated_raw_v1",
    )
    assert recovered is not None
    assert recovered.candidate_source is ReturnedCandidateSource.RAW_DUPLICATE_RECOVERY
    assert recovered.duplicate is not None and not recovered.duplicate.exact_duplicate
    assert recovered.duplicate_recovery is not None
    assert recovered.duplicate_recovery.recovered_raw_rank == 0
    assert recovered.posterior_calls == result.posterior_calls
    assert recovered.simulator_calls == 0
    assert recovered.duplicate_recovery.additional_posterior_calls == 0
    assert recovered.duplicate_recovery.additional_random_draws == 0

    exhausted = recover_best_ranked_unevaluated_raw(
        result,
        evaluated_theta=torch.cat((evaluated, result.pool.theta), dim=0).contiguous(),
        parameter_spec=parameter_spec(2),
        config=proposal_config,
        policy_id="best_ranked_unevaluated_raw_v1",
    )
    assert exhausted is None


def test_operator_norm_and_trace_produce_discriminating_raw_rankings() -> None:
    spec = parameter_spec(2)
    evaluated = torch.tensor([[0.0, 0.0]], dtype=DTYPE)
    operator = propose_candidate(
        SyntheticSurrogate(mode="ranking"),
        parameter_spec=spec,
        target=target(),
        evaluated_theta=evaluated,
        config=config(max_iterations=1),
    )
    trace = propose_candidate(
        SyntheticSurrogate(mode="ranking"),
        parameter_spec=spec,
        target=target(),
        evaluated_theta=evaluated,
        config=config(
            exploration_policy="trace_marginal_variance",  # type: ignore[arg-type]
            max_iterations=1,
        ),
    )
    operator_theta = operator.pool.theta[operator.raw_rankings[0].raw_index, 0]
    trace_theta = trace.pool.theta[trace.raw_rankings[0].raw_index, 0]
    assert operator_theta < 0.5
    assert trace_theta >= 0.5


def test_exact_ties_return_nonworsening_refinement_in_earliest_order() -> None:
    first = propose_candidate(
        SyntheticSurrogate(mode="tie"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(max_iterations=2),
    )
    second = propose_candidate(
        SyntheticSurrogate(mode="tie"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(max_iterations=2),
    )
    assert first.raw_rankings[0].raw_index == 0
    assert first.restarts[0].raw_index == 0
    assert first.restarts[0].restart_index == 0
    assert first.candidate_source is ReturnedCandidateSource.REFINED
    assert first.fallback_reason is None
    assert first.fallback_explanation is None
    assert torch.equal(first.candidate, first.pool.theta[0])
    assert torch.equal(first.candidate, second.candidate)
    assert first.candidate_id == second.candidate_id


def test_all_restart_failures_return_typed_raw_fallback() -> None:
    result = propose_candidate(
        SyntheticSurrogate(mode="scalar_raise"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(max_iterations=2),
    )
    assert result.status is ProposalStatus.SUCCESS
    assert all(item.status is RestartStatus.FAILED for item in result.restarts)
    assert result.candidate_source is ReturnedCandidateSource.RAW_FALLBACK
    assert result.fallback_reason is FallbackReason.ALL_RESTARTS_FAILED
    assert result.fallback_explanation
    assert torch.equal(
        result.candidate, result.pool.theta[result.raw_rankings[0].raw_index]
    )


def test_no_valid_nonworsening_outcome_returns_typed_raw_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mutaco.acquisition import proposal

    def worsening_restart(
        surrogate: object,
        start: torch.Tensor,
        *,
        raw_index: int,
        restart_index: int,
        initial_score: torch.Tensor,
        **kwargs: object,
    ) -> object:
        del surrogate, kwargs
        diagnostic = RestartDiagnostic(
            restart_index=restart_index,
            raw_index=raw_index,
            status=RestartStatus.COMPLETED,
            stop_reason=RestartStopReason.MAX_ITERATIONS,
            iterations=1,
            posterior_calls=0,
            accepted_steps=1,
            backtracks=0,
            initial_score=float(initial_score),
            final_score=float(initial_score - 1.0),
        )
        return proposal._RestartOutcome(  # type: ignore[attr-defined]
            start.detach().clone(),
            initial_score.detach().clone() - 1.0,
            diagnostic,
        )

    monkeypatch.setattr(proposal, "_exploration_restart", worsening_restart)
    result = propose_candidate(
        SyntheticSurrogate(mode="exploration"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(max_iterations=1),
    )
    assert result.candidate_source is ReturnedCandidateSource.RAW_FALLBACK
    assert (
        result.fallback_reason
        is FallbackReason.NO_VALID_NONWORSENING_REFINEMENT
    )


def test_tau_interior_changes_operational_branch_not_scientific_configuration() -> None:
    evaluated = torch.tensor([[0.0, 0.0]], dtype=DTYPE)
    interior = propose_candidate(
        SyntheticSurrogate(mode="barrier"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=evaluated,
        config=config(tau_interior=0.0, max_iterations=1),
    )
    strict_numerical_domain = propose_candidate(
        SyntheticSurrogate(mode="barrier"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=evaluated,
        config=config(tau_interior=0.999999, max_iterations=1),
    )
    assert interior.branch is AcquisitionBranch.BARRIER
    assert strict_numerical_domain.branch is AcquisitionBranch.EXPLORATION
    assert strict_numerical_domain.configuration["tau_interior"] == 0.999999


def test_full_proposal_is_dimension_generic_at_p6() -> None:
    result = propose_candidate(
        SyntheticSurrogate(dimension=6, mode="barrier"),
        parameter_spec=parameter_spec(6),
        target=target(),
        evaluated_theta=torch.zeros((1, 6), dtype=DTYPE),
        config=config(max_iterations=2),
    )
    assert result.status is ProposalStatus.SUCCESS
    assert result.candidate is not None and result.candidate.shape == (6,)
    assert result.pool.theta.shape == (16, 6)


def test_raw_posterior_and_no_finite_exploration_failures_are_typed() -> None:
    failed_call = propose_candidate(
        SyntheticSurrogate(mode="raise"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(),
    )
    assert failed_call.status is ProposalStatus.FAILED
    assert failed_call.failure_reason is ProposalFailureReason.RAW_POSTERIOR_ERROR
    assert failed_call.posterior_calls.total == 1
    nonfinite = propose_candidate(
        NonfiniteScoreSurrogate(),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(),
    )
    assert nonfinite.status is ProposalStatus.FAILED
    assert (
        nonfinite.failure_reason
        is ProposalFailureReason.NO_FINITE_EXPLORATION_CANDIDATE
    )
    assert nonfinite.candidate is None


def test_raw_barrier_scoring_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mutaco.acquisition import proposal

    class BrokenBarrier:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def evaluate(self, posterior: object, target_value: object) -> object:
            del posterior, target_value
            raise FloatingPointError("deliberate barrier scoring failure")

    surrogate = SyntheticSurrogate(mode="barrier")
    monkeypatch.setattr(proposal, "MatchedBarrierObjective", BrokenBarrier)
    result = propose_candidate(
        surrogate,
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(),
    )
    assert result.status is ProposalStatus.FAILED
    assert result.branch_reason is BranchReason.RAW_POSTERIOR_SCORING_FAILED
    assert (
        result.failure_reason
        is ProposalFailureReason.RAW_POSTERIOR_SCORING_ERROR
    )
    assert result.failure_explanation == (
        "FloatingPointError: deliberate barrier scoring failure"
    )
    assert result.posterior_calls.raw_pool == 1
    assert result.posterior_calls.per_restart == ()
    assert result.posterior_calls.total == 1 == len(surrogate.call_shapes)
    assert result.simulator_calls == 0


def test_raw_exploration_scoring_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mutaco.acquisition import proposal

    def fail_exploration(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ArithmeticError("deliberate exploration scoring failure")

    surrogate = SyntheticSurrogate(mode="exploration")
    monkeypatch.setattr(proposal, "evaluate_exploration", fail_exploration)
    result = propose_candidate(
        surrogate,
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(),
    )
    assert result.status is ProposalStatus.FAILED
    assert result.branch is AcquisitionBranch.NONE
    assert result.candidate_source is ReturnedCandidateSource.NONE
    assert result.branch_reason is BranchReason.RAW_POSTERIOR_SCORING_FAILED
    assert (
        result.failure_reason
        is ProposalFailureReason.RAW_POSTERIOR_SCORING_ERROR
    )
    assert result.posterior_calls.total == 1 == len(surrogate.call_shapes)
    assert result.restarts == ()
    assert result.simulator_calls == 0


def test_raw_ranking_preparation_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mutaco.acquisition import proposal

    def fail_ranking(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("deliberate ranking preparation failure")

    surrogate = SyntheticSurrogate(mode="barrier")
    monkeypatch.setattr(proposal, "_ranking", fail_ranking)
    result = propose_candidate(
        surrogate,
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(),
    )
    assert result.status is ProposalStatus.FAILED
    assert result.branch_reason is BranchReason.RAW_POSTERIOR_SCORING_FAILED
    assert (
        result.failure_reason
        is ProposalFailureReason.RAW_POSTERIOR_SCORING_ERROR
    )
    assert result.posterior_calls.total == 1 == len(surrogate.call_shapes)
    assert result.simulator_calls == 0


def test_timings_and_device_are_excluded_from_candidate_identity() -> None:
    first = propose_candidate(
        SyntheticSurrogate(mode="tie"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(max_iterations=1),
    )
    second = propose_candidate(
        SyntheticSurrogate(mode="tie"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(max_iterations=1),
    )
    assert first.candidate_id == second.candidate_id
    assert torch.equal(first.candidate, second.candidate)
    assert "timings" not in first.configuration
    assert "device" not in first.configuration
    assert first.execution["interpreter_path"]
    assert first.execution["python_version"]
    assert first.execution["torch_version"]
    assert first.execution["botorch_version"]
    assert first.execution["gpytorch_version"]
    extraction = first.execution["bounded_host_extraction"]
    assert extraction["policy"] == "bounded-control-identity-extraction-v1"
    assert extraction["cpu_objective_or_optimizer"] is False
    assert extraction["full_posterior_transfer"] is False
    assert extraction["selection_changed"] is False


def test_exact_gp_integrates_through_public_posterior_without_refit() -> None:
    spec = parameter_spec(2)
    theta = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.25],
            [0.25, 0.5],
        ],
        dtype=DTYPE,
    )
    response = torch.stack(
        (theta[:, 0] + theta[:, 1], theta[:, 0] - theta[:, 1]), dim=-1
    ).contiguous()
    training = SurrogateTrainingData(
        theta,
        response,
        spec,
        tuple(f"observation-{index}" for index in range(theta.shape[0])),
        "transform-test",
    )
    fit = fit_exact_multitask_gp(
        training,
        config=GPFitConfig(
            fit_seed=11,
            fit_seed_coordinate=SeedCoordinate("gp_fit", ("proposal-integration",)),
            max_iterations=5,
        ),
    )
    assert fit.diagnostic.status is GPFitStatus.SUCCESS, fit.diagnostic
    assert fit.surrogate is not None
    model_id = fit.surrogate.model_id
    result = propose_candidate(
        fit.surrogate,
        parameter_spec=spec,
        target=target(),
        evaluated_theta=theta,
        config=config(raw_samples=8, restarts=2, max_iterations=2),
    )
    assert result.status is ProposalStatus.SUCCESS
    assert result.model_id == model_id == fit.surrogate.model_id
    assert result.posterior_calls.raw_pool == 1
    assert result.simulator_calls == 0


def test_candidate_policy_support_document_is_generated() -> None:
    path = (
        Path(__file__).parents[2]
        / "scientific_contracts"
        / "mutaco_d_contract"
        / "CANDIDATE_POLICY_SUPPORT.md"
    )
    assert path.read_text(encoding="utf-8") == candidate_policy_support_markdown()


def test_prompt12_diagnostic_policy_identifiers_are_explicit() -> None:
    selected = config(
        confidence_radius_policy="trace_effective_dimension_radius_diagnostic_v1",
        infeasible_fallback_policy="minimum_normalized_target_miss_diagnostic_v1",
    )
    document = selected.scientific_document()
    assert document["confidence_radius_policy"].endswith("diagnostic_v1")
    assert document["infeasible_fallback_policy"].endswith("diagnostic_v1")
    with pytest.raises(ValueError, match="confidence-radius"):
        config(confidence_radius_policy="unknown")
    with pytest.raises(ValueError, match="infeasible-fallback"):
        config(infeasible_fallback_policy="unknown")


def test_target_aware_infeasible_fallback_is_minimized_without_extra_work() -> None:
    result = propose_candidate(
        SyntheticSurrogate(mode="exploration"),
        parameter_spec=parameter_spec(2),
        target=target(),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE),
        config=config(
            confidence_radius_policy="trace_effective_dimension_radius_diagnostic_v1",
            infeasible_fallback_policy="minimum_normalized_target_miss_diagnostic_v1",
            max_iterations=2,
        ),
    )
    assert result.status is ProposalStatus.SUCCESS
    assert result.branch is AcquisitionBranch.EXPLORATION
    assert result.branch_direction == "minimize"
    scores = [row.score for row in result.raw_rankings]
    assert scores == sorted(scores)
    assert result.simulator_calls == 0


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compatible Torch CUDA is unavailable"
)
def test_candidate_generation_scoring_and_refinement_remain_on_cuda() -> None:
    device = torch.device("cuda")
    surrogate = SyntheticSurrogate(mode="exploration", device=device)
    result = propose_candidate(
        surrogate,
        parameter_spec=parameter_spec(2),
        target=target(device),
        evaluated_theta=torch.tensor([[0.0, 0.0]], dtype=DTYPE, device=device),
        config=config(max_iterations=2),
    )
    assert result.status is ProposalStatus.SUCCESS
    assert result.pool.theta.device.type == "cuda"
    assert result.candidate is not None and result.candidate.device.type == "cuda"
    assert result.pool.generation_device == "cpu"
