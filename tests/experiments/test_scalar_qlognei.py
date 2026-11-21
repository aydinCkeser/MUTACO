from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from mutaco import (
    ParameterAxis,
    ParameterSpec,
    ResponseAxis,
    ResponseSpec,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
)
from experiments.benchmarking.methods.scalar_qlognei import (
    SCALAR_POSTERIOR_MEAN_LOSS_TERMINAL,
    SCALAR_QLOGNEI_POLICY,
    ScalarQLogNEIConfig,
    ScalarQLogNEIError,
    fit_scalar_loss_gp,
    observed_physical_mae,
    propose_scalar_qlognei,
    recommend_scalar_posterior_mean_loss,
)
import experiments.benchmarking.methods.primary_search as primary_search
import experiments.benchmarking.methods.scalar_qlognei as scalar_qlognei


def _config(seed: int = 31) -> ScalarQLogNEIConfig:
    deriver = SeedDeriver(ScientificIdentity("scalar-qlognei-test"), seed)
    return ScalarQLogNEIConfig(
        n_mc=16,
        raw_samples=8,
        restarts=2,
        max_iterations=8,
        fit_max_iterations=8,
        seed_deriver=deriver,
        fit_coordinate=SeedCoordinate("fit", (0,)),
        sampler_coordinate=SeedCoordinate("mc", (0,)),
        optimizer_coordinate=SeedCoordinate("optimizer", (0,)),
    )


def test_observed_physical_mae_is_scalar_physical_objective() -> None:
    responses = torch.tensor([[1.0, 3.0], [2.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([1.0, 1.0], dtype=torch.float32)
    assert torch.equal(observed_physical_mae(responses, target), torch.tensor([1.0, 1.0]))


def test_official_scalar_qlognei_stays_float32_q1_and_zero_call() -> None:
    parameters = ParameterSpec((ParameterAxis("theta", 0.0, 1.0),))
    theta = torch.linspace(0.02, 0.98, 16, dtype=torch.float32).unsqueeze(-1)
    responses = torch.cat((theta, 1.0 - theta), dim=1).contiguous()
    target = torch.tensor([0.4, 0.6], dtype=torch.float32)
    state_before = torch.random.get_rng_state().clone()

    fitted = fit_scalar_loss_gp(
        theta=theta,
        responses_physical=responses,
        target_physical=target,
        parameter_spec=parameters,
        config=_config(),
    )
    proposal = propose_scalar_qlognei(
        fitted=fitted,
        baseline=theta,
        parameter_spec=parameters,
        config=_config(),
    )
    recommendation = recommend_scalar_posterior_mean_loss(
        fitted=fitted,
        parameter_spec=parameters,
        reference=theta,
        seed_deriver=_config().seed_deriver,
        coordinate=SeedCoordinate("terminal", (0,)),
        raw_samples=8,
        restarts=2,
        max_iterations=8,
    )

    assert fitted.model.train_inputs[0].dtype is torch.float32
    assert proposal.candidate.dtype is torch.float32
    assert proposal.provenance["policy_id"] == SCALAR_QLOGNEI_POLICY
    assert proposal.provenance["batch_q"] == 1
    assert proposal.provenance["baseline_covariance_dimension"] == 16
    assert proposal.provenance["simulator_calls"] == 0
    assert proposal.provenance["fallback_trigger"] is None
    assert recommendation.selected_theta.dtype is torch.float32
    assert recommendation.document()["policy_id"] == SCALAR_POSTERIOR_MEAN_LOSS_TERMINAL
    assert recommendation.document()["simulator_calls"] == 0
    assert torch.equal(torch.random.get_rng_state(), state_before)


def _fitted_scalar_problem():
    parameters = ParameterSpec((ParameterAxis("theta", 0.0, 1.0),))
    theta = torch.linspace(0.02, 0.98, 16, dtype=torch.float32).unsqueeze(-1)
    responses = torch.cat((theta, 1.0 - theta), dim=1).contiguous()
    fitted = fit_scalar_loss_gp(
        theta=theta,
        responses_physical=responses,
        target_physical=torch.tensor([0.4, 0.6], dtype=torch.float32),
        parameter_spec=parameters,
        config=_config(),
    )
    return parameters, theta, fitted


def test_duplicate_optimizer_result_records_trigger_without_changing_fallback(
    monkeypatch,
) -> None:
    parameters, theta, fitted = _fitted_scalar_problem()

    def duplicate(*args, **kwargs):
        del args, kwargs
        return theta[3].reshape(1, -1).clone(), torch.tensor(0.25)

    monkeypatch.setattr(scalar_qlognei, "optimize_acqf", duplicate)
    state_before = torch.random.get_rng_state().clone()
    first = propose_scalar_qlognei(
        fitted=fitted, baseline=theta, parameter_spec=parameters, config=_config()
    )
    second = propose_scalar_qlognei(
        fitted=fitted, baseline=theta, parameter_spec=parameters, config=_config()
    )
    trigger = first.provenance["fallback_trigger"]
    assert first.provenance["candidate_source"] == (
        "deterministic_raw_fallback_after_ScalarQLogNEIError"
    )
    assert trigger == {
        "stage": "optimized_candidate_validation",
        "optimizer_returned": True,
        "exception_type": "ScalarQLogNEIError",
        "exception_message": "optimized qLogNEI candidate is invalid",
        "trigger_reasons": ["exact_baseline_duplicate"],
        "rejected_candidate": theta[3].tolist(),
        "rejected_candidate_finite": True,
        "rejected_candidate_nonfinite": [],
        "rejected_log_acquisition_value": 0.25,
        "rejected_log_acquisition_value_finite": True,
        "rejected_log_acquisition_value_nonfinite_kind": None,
        "exact_duplicate_indices_zero_based": [3],
    }
    assert not any(torch.equal(first.candidate, row) for row in theta)
    assert torch.equal(first.candidate, second.candidate)
    assert first.provenance["log_acquisition_value"] == second.provenance[
        "log_acquisition_value"
    ]
    assert torch.equal(torch.random.get_rng_state(), state_before)


@pytest.mark.parametrize("nonfinite", ("candidate", "value"))
def test_nonfinite_optimizer_result_is_path_safe_and_uses_same_fallback(
    monkeypatch, nonfinite: str
) -> None:
    parameters, theta, fitted = _fitted_scalar_problem()

    def invalid(*args, **kwargs):
        del args, kwargs
        candidate = torch.tensor([[float("nan")]], dtype=torch.float32)
        value = torch.tensor(0.25, dtype=torch.float32)
        if nonfinite == "value":
            candidate = torch.tensor([[0.375]], dtype=torch.float32)
            value = torch.tensor(float("-inf"), dtype=torch.float32)
        return candidate, value

    monkeypatch.setattr(scalar_qlognei, "optimize_acqf", invalid)
    proposal = propose_scalar_qlognei(
        fitted=fitted, baseline=theta, parameter_spec=parameters, config=_config()
    )
    trigger = proposal.provenance["fallback_trigger"]
    assert trigger["trigger_reasons"] == [
        f"nonfinite_{'candidate' if nonfinite == 'candidate' else 'acquisition_value'}"
    ]
    if nonfinite == "candidate":
        assert trigger["rejected_candidate"] is None
    else:
        assert trigger["rejected_candidate"] == [0.375]
    if nonfinite == "value":
        assert trigger["rejected_log_acquisition_value"] is None
    else:
        assert trigger["rejected_log_acquisition_value"] == 0.25
    assert trigger["rejected_candidate_nonfinite"] == (
        [{"flat_index": 0, "kind": "nan"}]
        if nonfinite == "candidate"
        else []
    )
    assert trigger["rejected_log_acquisition_value_nonfinite_kind"] == (
        "-inf" if nonfinite == "value" else None
    )
    assert bool(torch.isfinite(proposal.candidate).all())
    assert not any(torch.equal(proposal.candidate, row) for row in theta)


def test_optimizer_exception_is_persisted_without_policy_change(monkeypatch) -> None:
    parameters, theta, fitted = _fitted_scalar_problem()

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(scalar_qlognei, "optimize_acqf", fail)
    proposal = propose_scalar_qlognei(
        fitted=fitted, baseline=theta, parameter_spec=parameters, config=_config()
    )
    trigger = proposal.provenance["fallback_trigger"]
    assert proposal.provenance["candidate_source"] == (
        "deterministic_raw_fallback_after_RuntimeError"
    )
    assert trigger["stage"] == "continuous_optimizer"
    assert trigger["optimizer_returned"] is False
    assert trigger["exception_type"] == "RuntimeError"
    assert trigger["exception_message"] == "synthetic optimizer failure"
    assert trigger["trigger_reasons"] == ["optimizer_exception"]
    assert bool(torch.isfinite(proposal.candidate).all())
    assert not any(torch.equal(proposal.candidate, row) for row in theta)


def test_primary_scalar_qlognei_runner_is_16_plus_24_and_q1(monkeypatch) -> None:
    parameters = ParameterSpec((ParameterAxis("theta", 0.0, 1.0),))
    responses = ResponseSpec((ResponseAxis("a"), ResponseAxis("b")))
    initial = torch.linspace(0.02, 0.98, 16, dtype=torch.float32).unsqueeze(-1)
    calls = []
    fit_sizes = []

    def fake_fit(**kwargs):
        fit_sizes.append(kwargs["theta"].shape[0])
        return SimpleNamespace(
            model_id=f"model-{kwargs['theta'].shape[0]}",
            fit_diagnostic={"status": "success", "size": kwargs["theta"].shape[0]},
        )

    def fake_proposal(*, fitted, baseline, parameter_spec, config):
        del fitted, parameter_spec, config
        candidate = torch.tensor(
            [min(0.999, (baseline.shape[0] + 0.5) / 41.0)], dtype=torch.float32
        )
        return SimpleNamespace(
            candidate=candidate,
            provenance={
                "policy_id": SCALAR_QLOGNEI_POLICY,
                "batch_q": 1,
                "simulator_calls": 0,
            },
        )

    def fake_recommend(**kwargs):
        del kwargs
        return SimpleNamespace(
            document=lambda: {
                "policy_id": SCALAR_POSTERIOR_MEAN_LOSS_TERMINAL,
                "selected_theta": [0.5],
                "score": 0.0,
                "simulator_calls": 0,
                "gp_fits": 0,
            }
        )

    monkeypatch.setattr(primary_search, "fit_scalar_loss_gp", fake_fit)
    monkeypatch.setattr(primary_search, "propose_scalar_qlognei", fake_proposal)
    monkeypatch.setattr(
        primary_search, "recommend_scalar_posterior_mean_loss", fake_recommend
    )

    def evaluate(theta, index):
        calls.append((tuple(theta), index))
        return (theta[0], 1.0 - theta[0])

    settings = {
        "n_mc": 8,
        "raw_samples": 4,
        "restarts": 1,
        "max_iterations": 1,
        "fit_max_iterations": 1,
        "terminal_raw_samples": 4,
        "terminal_restarts": 1,
        "terminal_max_iterations": 1,
    }
    result = primary_search.run_primary_scalar_qlognei_search(
        initial_design=initial,
        budget=40,
        target_physical=torch.tensor([0.5, 0.5], dtype=torch.float32),
        target_id="target-a",
        target_index=0,
        parameter_spec=parameters,
        response_spec=responses,
        evaluate=evaluate,
        inference_seed_deriver=SeedDeriver(ScientificIdentity("inference"), 41),
        terminal_seed_deriver=SeedDeriver(ScientificIdentity("terminal"), 42),
        coordinate_prefix=("campaign", "target-a", 40, "scalar_qlognei"),
        qlognei_settings=settings,
    )
    assert result.status == "recommended"
    assert result.method == "scalar_qlognei"
    assert result.successful_calls == 40
    assert [item[1] for item in calls] == list(range(1, 41))
    assert fit_sizes == list(range(16, 41))
    assert len(result.proposal_diagnostics) == 24
    assert all(row["batch_q"] == 1 for row in result.proposal_diagnostics)
    assert result.recommendation["simulator_calls"] == 0
    assert tuple(result.timing) == primary_search.PRIMARY_TIMING_KEYS
    assert all(result.timing[key] > 0.0 for key in primary_search.PRIMARY_TIMING_KEYS[:4])
    assert sum(result.timing[key] for key in primary_search.PRIMARY_TIMING_KEYS[:-1]) == pytest.approx(
        result.timing["total"]
    )

    def fail_proposal(**kwargs):
        del kwargs
        raise ScalarQLogNEIError("synthetic acquisition failure")

    monkeypatch.setattr(primary_search, "propose_scalar_qlognei", fail_proposal)
    failed = primary_search.run_primary_scalar_qlognei_search(
        initial_design=initial,
        budget=40,
        target_physical=torch.tensor([0.5, 0.5], dtype=torch.float32),
        target_id="target-a",
        target_index=0,
        parameter_spec=parameters,
        response_spec=responses,
        evaluate=evaluate,
        inference_seed_deriver=SeedDeriver(ScientificIdentity("inference"), 41),
        terminal_seed_deriver=SeedDeriver(ScientificIdentity("terminal"), 42),
        coordinate_prefix=("campaign", "target-a", 40, "scalar_qlognei"),
        qlognei_settings=settings,
    )
    assert failed.status == "failed"
    assert failed.failure_stage == "acquisition"
    assert failed.timing["acquisition"] > 0.0
    assert sum(
        failed.timing[key] for key in primary_search.PRIMARY_TIMING_KEYS[:-1]
    ) == pytest.approx(failed.timing["total"])


def test_float32_policy_forbids_precision_fallback_and_records_jitter_contract() -> None:
    document = _config().document()
    assert document["dtype_policy"] == "float32_end_to_end_v1"
    assert document["precision_fallback"] == "forbidden"
    assert document["cholesky_policy"] == {
        "implementation": "gpytorch.settings.cholesky_jitter",
        "initial_float32_jitter": 1.0e-6,
        "maximum_attempts": 3,
        "unrecovered_action": "typed_failure_before_next_simulator_call",
    }


def test_fit_failure_is_typed_and_does_not_promote_precision(monkeypatch) -> None:
    parameters = ParameterSpec((ParameterAxis("theta", 0.0, 1.0),))
    theta = torch.linspace(0.02, 0.98, 16, dtype=torch.float32).unsqueeze(-1)
    responses = torch.cat((theta, 1.0 - theta), dim=1).contiguous()
    state_before = torch.random.get_rng_state().clone()

    def fail_fit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic factorization failure")

    monkeypatch.setattr(scalar_qlognei, "fit_gpytorch_mll", fail_fit)
    with pytest.raises(ScalarQLogNEIError, match="without precision fallback"):
        fit_scalar_loss_gp(
            theta=theta,
            responses_physical=responses,
            target_physical=torch.tensor([0.4, 0.6], dtype=torch.float32),
            parameter_spec=parameters,
            config=_config(),
        )
    assert torch.equal(torch.random.get_rng_state(), state_before)
