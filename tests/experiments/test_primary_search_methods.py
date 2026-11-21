from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mutaco import (
    GPFitStatus,
    ParameterAxis,
    ParameterSpec,
    ResponseAxis,
    ResponseSpec,
    ScientificIdentity,
    SeedDeriver,
)
import experiments.benchmarking.methods.primary_search as primary_search


def test_primary_sobol_search_is_nonadaptive_and_zero_fit() -> None:
    design = torch.linspace(0.0, 1.0, 40, dtype=torch.float64).unsqueeze(-1).contiguous()
    seen = []

    def evaluate(theta, index):
        seen.append((tuple(theta), index))
        return (theta[0], 1.0 - theta[0])

    result = primary_search.run_primary_sobol_search(
        design=design,
        target_physical=torch.tensor([0.5, 0.5], dtype=torch.float64).contiguous(),
        response_spec=ResponseSpec((ResponseAxis("a"), ResponseAxis("b"))),
        evaluate=evaluate,
    )
    assert result.status == "recommended"
    assert result.successful_calls == 40
    assert result.fit_diagnostics == result.proposal_diagnostics == ()
    assert [item[1] for item in seen] == list(range(1, 41))
    assert result.recommendation["policy_id"] == "evaluated_best_observed_physical_mae_v1"
    assert result.recommendation["simulator_calls"] == 0
    assert tuple(result.timing) == primary_search.PRIMARY_TIMING_KEYS
    assert result.timing["gp_fit"] == result.timing["acquisition"] == 0.0
    assert result.timing["terminal"] > 0.0
    assert sum(result.timing[key] for key in primary_search.PRIMARY_TIMING_KEYS[:-1]) == pytest.approx(
        result.timing["total"]
    )


def test_primary_nei_search_uses_16_plus_24_and_one_fit_per_version(monkeypatch) -> None:
    dtype = torch.float64
    parameters = ParameterSpec((ParameterAxis("theta", 0.0, 1.0),))
    responses = ResponseSpec((ResponseAxis("a"), ResponseAxis("b")))
    initial = torch.linspace(0.02, 0.98, 16, dtype=dtype).unsqueeze(-1).contiguous()
    calls = []
    fit_versions = []

    class FakeSurrogate:
        model_id = "fake-model"

        def __init__(self, transform_id):
            self.transform_id = transform_id

    def fake_fit(data, *, config, warm_start_states, training_version):
        del config, warm_start_states
        fit_versions.append(training_version)
        diagnostic = SimpleNamespace(
            status=GPFitStatus.SUCCESS,
            fit_id=f"fit-{training_version}",
            model_id=f"model-{training_version}",
            training_data_id=data.identity(),
            elapsed_seconds=0.01,
            failure_kind=None,
            failure_message=None,
            recovery_attempts=(),
        )
        return SimpleNamespace(
            diagnostic=diagnostic,
            surrogate=FakeSurrogate(data.transform_id),
        )

    def fake_proposal(surrogate, *, baseline, transform, target_physical, parameter_spec, config):
        del surrogate, transform, target_physical, parameter_spec, config
        value = min(0.999, (baseline.shape[0] + 0.5) / 41.0)
        candidate = torch.tensor([value], dtype=dtype)
        return SimpleNamespace(
            candidate=candidate,
            provenance=lambda: {
                "policy_id": "composite_mc_nei_physical_mae_v1",
                "candidate": candidate.tolist(),
                "value": 0.1,
                "simulator_calls": 0,
            },
        )

    def fake_recommend(**kwargs):
        theta = torch.tensor([0.5], dtype=dtype)
        return SimpleNamespace(
            document=lambda: {
                "policy_id": "continuous_posterior_mean_physical_l1_mae",
                "selected_theta": theta.tolist(),
                "score": 0.0,
                "simulator_calls": 0,
                "gp_fits": 0,
            }
        )

    monkeypatch.setattr(primary_search, "fit_exact_multitask_gp_with_recovery", fake_fit)
    monkeypatch.setattr(primary_search, "propose_composite_mc_nei", fake_proposal)
    monkeypatch.setattr(primary_search, "recommend_continuous_physical_mae", fake_recommend)

    def evaluate(theta, index):
        calls.append((tuple(theta), index))
        return (theta[0], 1.0 - theta[0])

    inference = SeedDeriver(ScientificIdentity("nei-inference-test"), 10)
    terminal = SeedDeriver(ScientificIdentity("terminal-test"), 11)
    result = primary_search.run_primary_nei_search(
        initial_design=initial,
        budget=40,
        target_physical=torch.tensor([0.5, 0.5], dtype=dtype).contiguous(),
        target_id="target-a",
        target_index=0,
        parameter_spec=parameters,
        response_spec=responses,
        evaluate=evaluate,
        inference_seed_deriver=inference,
        terminal_seed_deriver=terminal,
        coordinate_prefix=("campaign", "target-a", 40, "nei_bo"),
        gp_settings={"max_iterations": 1},
        nei_settings={
            "n_mc": 8,
            "raw_samples": 4,
            "restarts": 1,
            "max_iterations": 1,
            "learning_rate": 0.05,
        },
        snapshot_positions=(16, 28, 40),
        snapshot_callback=lambda position, surrogate, baseline, physical, transform: {
            "position": position,
            "baseline_size": baseline.shape[0],
            "response_size": physical.shape[0],
            "transform_id": transform.metadata.transform_id,
        },
    )
    assert result.status == "recommended"
    assert result.successful_calls == 40
    assert [item[1] for item in calls] == list(range(1, 41))
    assert fit_versions == list(range(25))
    assert len(result.fit_diagnostics) == 25
    assert len(result.proposal_diagnostics) == 24
    assert result.transform is not None
    assert result.recommendation["simulator_calls"] == 0
    assert [row["position"] for row in result.qualification_snapshots] == [16, 28, 40]
    assert tuple(result.timing) == primary_search.PRIMARY_TIMING_KEYS
    assert result.timing["simulator"] > 0.0
    assert result.timing["gp_fit"] > 0.0
    assert result.timing["acquisition"] > 0.0
    assert result.timing["terminal"] > 0.0
