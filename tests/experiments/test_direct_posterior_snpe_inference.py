from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver
import experiments.benchmarking.methods.direct_posterior_inference as inference_module
from experiments.paper.common.protocol_v3 import DecisionOutcome, PhysicalCallCounts


class _Writer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        pass


class _Posterior:
    def __init__(self, round_index):
        self.round_index = round_index

    def set_default_x(self, target):
        self.target = target
        return self

    def sample(self, shape, show_progress_bars=False):
        del show_progress_bars
        return torch.rand((shape[0], 2), dtype=torch.float32)


class _NPE:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.appended = []
        _NPE.instances.append(self)

    def append_simulations(self, theta, x, **kwargs):
        self.appended.append((theta.clone(), x.clone(), kwargs.get("proposal")))
        return self

    def train(self, **kwargs):
        self.training_kwargs = kwargs
        return object()

    def build_posterior(self, estimator, sample_with):
        del estimator
        assert sample_with == "direct"
        return _Posterior(len(self.appended) - 1)


class _BoxUniform:
    def __init__(self, lower, upper, device):
        self.lower, self.upper, self.device = lower, upper, device


class _Adapter:
    def build_simulator(self, config, *, device, dtype):
        del config, device, dtype
        spec = SimpleNamespace(lower=(0.0, 0.0), upper=(1.0, 1.0))
        return SimpleNamespace(capabilities=SimpleNamespace(parameter_spec=spec))


def test_cold_start_npe_preserves_paper120_identity_and_seed_coordinates() -> None:
    config = {
        "dtype": "float32",
        "campaign": {"campaign_seed": 81257},
        "simulation_budget_per_target": 40,
        "randomness": {
            "method_seed_namespace": "method_runs",
            "sbi_batch_context_policy": "per_logical_simulation_index_v1",
        },
        "sbi": {
            "density_estimator": "maf",
            "training_batch_size": 64,
            "learning_rate": 0.0005,
            "validation_fraction": 0.2,
            "stop_after_epochs": 20,
            "max_num_epochs": 200,
            "posterior_samples": 10000,
            "show_progress_bars": False,
            "posterior_sampling": "direct",
            "fallback_policy": "none",
            "training_diagnostics_policy": "sbi_training_summary_loss_v0261",
            "training_diagnostics_audit": "sbi_tensorboard_training_loss_audit_v1",
        },
    }
    target_family = ScientificIdentity(
        "scientific-v1-ab3b6488ee6ebfdc65261e63669ee70837aa62e88dcdb1d32dd22f4ee555c05f"
    )
    method_configuration = inference_module._method_configuration_for_identity(
        method="sbi_npe", config=config, budget=40, allocations=(40,)
    )
    assert "sequential_round_allocation" not in method_configuration
    method_science = ScientificIdentity.create(
        {
            "schema": "benchmark-method-run-v1",
            "target_family_id": target_family.value,
            "target_record_id": (
                "frozen-target-record-v2-e3a859e807a718a6749aae18930a5d1c4982970f272ad09cf46fd700ccda0ef6"
            ),
            "method": "sbi_npe",
            "method_configuration": method_configuration,
        }
    )
    assert method_science.value == (
        "scientific-v1-55697c18de2bf86ebb1f86dd34bd49b675b160182e35613102b557afe23a4aa8"
    )
    deriver = SeedDeriver(method_science, 81257)
    assert deriver.seed_for(SeedCoordinate("sbi_simulation", ("sbi_npe", 0, 0))) == (
        5393899087350487035
    )
    assert deriver.seed_for(SeedCoordinate("sbi_recommendation", ("sbi_npe", 0))) == (
        1965748830523254238
    )


def test_snpe_allocation_remains_identity_bearing() -> None:
    config = {
        "dtype": "float32",
        "randomness": {
            "method_seed_namespace": "method_runs",
            "sbi_batch_context_policy": "per_logical_simulation_index_v1",
        },
        "sbi": {"density_estimator": "maf"},
    }
    method_configuration = inference_module._method_configuration_for_identity(
        method="sbi_snpe", config=config, budget=40, allocations=(14, 13, 13)
    )
    assert method_configuration["sequential_round_allocation"] == [14, 13, 13]


@pytest.mark.parametrize(
    "budget, allocation, expected_simulated",
    [
        (40, [14, 13, 13], [(14, 0, 0), (13, 1, 14), (13, 2, 27)]),
        (9, [3, 3, 3], [(3, 0, 0), (3, 1, 3), (3, 2, 6)]),
    ],
)
def test_target_specific_snpe_uses_prior_then_proposal_rounds_and_preserves_rng(
    tmp_path, monkeypatch, budget, allocation, expected_simulated
):
    _NPE.instances.clear()
    simulated = []

    def fake_simulate(simulator, theta, **kwargs):
        del simulator
        simulated.append((theta.shape[0], kwargs["round_index"], kwargs["logical_start_index"]))
        return torch.zeros((theta.shape[0], 1), dtype=torch.float32)

    def fake_decision(**kwargs):
        theta = tuple(float(value) for value in kwargs["samples"][0])
        return SimpleNamespace(
            decision=DecisionOutcome(
                method="sbi_snpe",
                status="recommended",
                decision_policy_identifier="posterior_sample_map_physical_v1",
                method_configuration_identity=kwargs["method_configuration_identity"],
                inference_budget=kwargs["inference_budget"],
                inference_calls=kwargs["inference_calls"],
                recommended_theta=theta,
                decision_metadata={"candidate_count": 10000},
            ),
            timing_event={"task": "posterior_sample_map_scoring_selection"},
        )

    monkeypatch.setattr(
        inference_module,
        "_lazy_sbi",
        lambda: (_NPE, object, object, _BoxUniform, object, _Writer),
    )
    monkeypatch.setattr(inference_module, "_simulate_batch", fake_simulate)
    monkeypatch.setattr(inference_module, "safe_training_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(inference_module, "posterior_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(inference_module, "posterior_sample_map_decision", fake_decision)
    monkeypatch.setattr(inference_module, "version", lambda name: "0.26.1")
    monkeypatch.setattr(
        inference_module,
        "summarize_sbi_timing",
        lambda *a, **k: {"schema_version": "test-timing"},
    )
    config = {
        "dtype": "float32",
        "campaign": {"campaign_seed": 7},
        "simulation_budget_per_target": budget,
        "randomness": {
            "method_seed_namespace": "method_runs",
            "sbi_batch_context_policy": "per_logical_simulation_index_v1",
        },
        "sbi": {
            "density_estimator": "maf",
            "training_batch_size": 64,
            "learning_rate": 0.0005,
            "validation_fraction": 0.2,
            "stop_after_epochs": 20,
            "max_num_epochs": 200,
            "posterior_samples": 10000,
            "show_progress_bars": False,
            "sequential_round_allocation": allocation,
        },
    }
    before = torch.random.get_rng_state().clone()
    row = inference_module.run_direct_posterior_sample_map_target(
        method="sbi_snpe",
        adapter=_Adapter(),
        index=3,
        theta_star=torch.tensor([0.2, 0.4]),
        target_response=torch.tensor([0.5]),
        config=config,
        campaign_science=ScientificIdentity("target-family"),
        device=torch.device("cpu"),
        dtype=torch.float32,
        artifact_output=tmp_path,
        target_record_id="target-3",
    )
    assert torch.equal(torch.random.get_rng_state(), before)
    assert row["method"] == "sbi_snpe"
    assert row["inference_calls"] == budget
    assert row["assessment_calls"] == 0
    assert row["sequential_round_allocation"] == allocation
    assert row["posterior_samples"] == 10000
    assert simulated == expected_simulated
    appended = _NPE.instances[0].appended
    assert appended[0][2] is None
    assert isinstance(appended[1][2], _Posterior)
    assert isinstance(appended[2][2], _Posterior)


def test_each_snpe_target_constructs_a_fresh_estimator(tmp_path, monkeypatch):
    _NPE.instances.clear()
    monkeypatch.setattr(
        inference_module,
        "_lazy_sbi",
        lambda: (_NPE, object, object, _BoxUniform, object, _Writer),
    )
    # Construction occurs before any physical simulation, so a deliberate stop
    # proves separate target-local estimator ownership without scientific calls.
    monkeypatch.setattr(
        inference_module,
        "_simulate_batch",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop before simulation")),
    )
    config = {
        "dtype": "float32",
        "campaign": {"campaign_seed": 7},
        "simulation_budget_per_target": 40,
        "randomness": {
            "method_seed_namespace": "method_runs",
            "sbi_batch_context_policy": "per_logical_simulation_index_v1",
        },
        "sbi": {
            "density_estimator": "maf",
            "training_batch_size": 64,
            "learning_rate": 0.0005,
            "validation_fraction": 0.2,
            "stop_after_epochs": 20,
            "max_num_epochs": 200,
            "posterior_samples": 10000,
            "show_progress_bars": False,
            "sequential_round_allocation": [14, 13, 13],
        },
    }
    for index in (0, 1):
        try:
            inference_module.run_direct_posterior_sample_map_target(
                method="sbi_snpe",
                adapter=_Adapter(),
                index=index,
                theta_star=torch.tensor([0.2, 0.4]),
                target_response=torch.tensor([0.5]),
                config=config,
                campaign_science=ScientificIdentity("target-family"),
                device=torch.device("cpu"),
                dtype=torch.float32,
                artifact_output=tmp_path / str(index),
                target_record_id=f"target-{index}",
            )
        except RuntimeError as exc:
            assert str(exc) == "stop before simulation"
    assert len(_NPE.instances) == 2
    assert _NPE.instances[0] is not _NPE.instances[1]
