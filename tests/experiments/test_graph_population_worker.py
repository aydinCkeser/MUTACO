from __future__ import annotations

import json
import random
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from mutaco import SimulationContext, SimulationResult
import experiments.graph_inversion.paper_v3_callbacks as callback_module
from experiments.graph_inversion.paper_v3_callbacks import (
    GraphPopulationCallbacks,
    GraphRawEvidencePersistenceError,
    _CallbackGraphAdapter,
    _assessment_randomness_evidence,
    method_configuration_identity,
)
from experiments.graph_inversion.paper_policies import validate_graph_paper_configuration
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.population_v3 import (
    MethodDeclarationV3,
    load_population_declaration_v3,
    population_call_preview,
    shard_indices,
)
from experiments.paper.common.method_specs import (
    PRIMARY_COMPARISON_GROUP,
    load_method_execution_specs,
    selected_method_configuration_paths,
)
from experiments.paper.common.protocol_v3 import method_inference_context
from experiments.paper.common.protocol_v3 import DecisionOutcome, PhysicalCallCounts
from experiments.paper.common.validation import PaperContractError
from experiments.graph_inversion.run_paper_v3_worker import (
    _inference_contexts_are_distinct,
    _target_records,
    _validate_source_module_origins,
    _write_worker_terminal,
)


ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/paper120_b40_mutaco_npe_v3_v1.json"
MUTACO_CONFIG = ROOT / "experiments/graph_inversion/configs/paper/policy_upgrade_100_targets_v1.json"
NPE_CONFIG = ROOT / "experiments/graph_inversion/configs/d4_100_targets.json"
SNPE_CONFIG = ROOT / "experiments/graph_inversion/configs/paper/paper120_b40_snpe_proposed_v1.json"
METHOD_SPECS = ROOT / "experiments/graph_inversion/configs/paper/paper120_primary_method_specs_v1.json"
QUALIFICATION_TEMPLATE = ROOT / "experiments/graph_inversion/configs/paper/paper120_unified_qualification_v1.template.json"
QUALIFICATION_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/paper120_b40_snpe_qualification_v1.json"
TARGET_LEDGER = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3"


def test_paper120_declaration_is_canonical_and_exactly_82_calls_per_target():
    declaration = load_population_declaration_v3(json.loads(DECLARATION.read_text()))
    assert declaration.document() == json.loads(DECLARATION.read_text())
    assert len(declaration.target_record_identities) == 120
    assert population_call_preview(1, declaration.methods)["maximum_total_calls"] == 82
    assert population_call_preview(120, declaration.methods) == {
        "target_construction_calls": 0,
        "maximum_inference_calls": 9600,
        "maximum_assessment_calls": 240,
        "maximum_total_calls": 9840,
        "recommendation_calls": 0,
        "scientific_calls_executed": 0,
    }


def test_method_configuration_identities_are_exact_file_owned():
    declaration = load_population_declaration_v3(json.loads(DECLARATION.read_text()))
    paths = {"mutaco": MUTACO_CONFIG, "sbi_npe": NPE_CONFIG}
    for method in declaration.methods:
        assert method.configuration_identity == method_configuration_identity(
            method.method, paths[method.method]
        )


def test_standalone_snpe_proposal_and_qualification_preview_are_approval_gated(
    tmp_path,
):
    config = json.loads(SNPE_CONFIG.read_text())
    template = json.loads(QUALIFICATION_TEMPLATE.read_text())
    assert config["state"] == "proposed_unexecuted"
    assert config["scientific_policy_status"] == "pending_owner_approval"
    assert config["sbi"]["sequential_round_allocation"] == [14, 13, 13]
    assert config["sbi"]["posterior_samples"] == 10000
    assert config["call_accounting"] == {
        "target_construction": 0,
        "inference": 40,
        "recommendation": 0,
        "assessment": 1,
        "total_per_valid_target": 41,
    }
    method = MethodDeclarationV3(
        "sbi_snpe",
        method_configuration_identity("sbi_snpe", SNPE_CONFIG),
        "posterior_sample_map_physical_v1",
        40,
    )
    assert population_call_preview(3, (method,))["maximum_total_calls"] == 123
    assert template["state"] == "proposed_non_executable_template"
    assert template["execution_requires_approval"] is True
    assert template["call_preview"]["maximum_total_calls"] == 123
    GraphPopulationCallbacks(
        npe_configuration=NPE_CONFIG,
        snpe_configuration=SNPE_CONFIG,
        output=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_snpe_qualification_declaration_selects_noncontiguous_frozen_targets():
    document = json.loads(QUALIFICATION_DECLARATION.read_text())
    declaration = load_population_declaration_v3(document)
    assert declaration.document() == document
    assert population_call_preview(3, declaration.methods)["maximum_total_calls"] == 123
    records = _target_records(TARGET_LEDGER, declaration, (0, 1, 2))
    assert tuple(row["target_index"] for row in records) == (0, 49, 99)
    assert tuple(row["target_record_id"] for row in records) == declaration.target_record_identities


def test_unified_method_specs_cover_primary_group_and_select_declared_subset():
    document = json.loads(METHOD_SPECS.read_text())
    specs = load_method_execution_specs(document, source=METHOD_SPECS)
    assert [spec.method for spec in specs] == ["mutaco", "sbi_npe", "sbi_snpe"]
    assert {spec.comparison_group for spec in specs} == {PRIMARY_COMPARISON_GROUP}
    declaration = load_population_declaration_v3(json.loads(DECLARATION.read_text()))
    paths = selected_method_configuration_paths(specs, declaration.methods, root=ROOT)
    assert set(paths) == {"mutaco", "sbi_npe"}


def test_callback_adapter_has_no_uncapped_simulator_surface():
    class Capabilities:
        supports_batch = True

    calls = []
    adapter = _CallbackGraphAdapter(
        Capabilities(),
        lambda theta, context: calls.append((theta, context)) or [1.0, 2.0],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    simulator = adapter.build_simulator({}, device=torch.device("cpu"), dtype=torch.float32)
    assert not hasattr(adapter, "_base")
    assert not hasattr(simulator, "_simulator")
    result = simulator.evaluate(
        torch.tensor([0.1], dtype=torch.float32),
        SimulationContext("science", "inference", 0, 0, 7),
    )
    assert isinstance(result, SimulationResult)
    assert result.values.tolist() == [1.0, 2.0]
    assert len(calls) == 1


def test_assessment_randomness_evidence_separates_disorder_initial_and_thermal():
    context = {"family_identity": "assessment-family", "evaluation_seed": 123}
    metadata = {"seed_manifest": {
        "node_parameter/weight/B=0": 1,
        "pairwise_coupling/B=0": 2,
        "initial_state/B=0/R=0": 3,
        "thermal_noise/B=0/R=0/V=0": 4,
    }}
    evidence = _assessment_randomness_evidence(context, metadata)
    assert set(evidence["compact"]["coordinate_family_identities"]) == {
        "quenched_disorder", "initial_state", "thermal_noise"
    }
    assert set(evidence["raw"]["namespace_seed_manifests"]["quenched_disorder"]) == {
        "node_parameter/weight/B=0", "pairwise_coupling/B=0"
    }
    assert evidence["raw"]["namespace_seed_manifests"]["initial_state"] == {
        "initial_state/B=0/R=0": 3
    }
    assert evidence["raw"]["namespace_seed_manifests"]["thermal_noise"] == {
        "thermal_noise/B=0/R=0/V=0": 4
    }
    replay = _assessment_randomness_evidence(context, metadata)
    assert replay == evidence
    changed = _assessment_randomness_evidence(
        {"family_identity": "assessment-family", "evaluation_seed": 124}, metadata
    )
    assert changed["compact"]["coordinate_family_identities"] != evidence["compact"][
        "coordinate_family_identities"
    ]


def test_graph_callbacks_reject_unconfigured_method_before_dispatch(tmp_path):
    callbacks = GraphPopulationCallbacks(
        mutaco_configuration=MUTACO_CONFIG,
        npe_configuration=NPE_CONFIG,
        output=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    declaration = MethodDeclarationV3("sbi_snpe", "x", "x", 40)
    with pytest.raises(KeyError):
        callbacks.run_method(
            declaration,
            {"target_record_id": "target", "target_index": 0},
            {"coordinate": {"namespace": "method_inference", "coordinates": ["target", "sbi_snpe"]}},
            lambda theta, context: [0.0],
        )


def test_standalone_snpe_binding_uses_decision_only_direct_posterior_path(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "snpe.json"
    config = json.loads(NPE_CONFIG.read_text())
    config["sbi"]["sequential_round_allocation"] = [14, 13, 13]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    callbacks = GraphPopulationCallbacks(
        snpe_configuration=config_path,
        output=tmp_path / "output",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    target = {
        "target_record_id": "target-snpe",
        "target_index": 0,
        "target_family_id": "frozen-target-family",
        "target_seed": 11,
        "theta_star": [1.0, 1.0, 1.0, 0.5],
        "target_response": [0.0] * 20,
    }
    identity = method_configuration_identity("sbi_snpe", config_path)
    declaration = MethodDeclarationV3(
        "sbi_snpe", identity, "posterior_sample_map_physical_v1", 40
    )
    observed = {}

    def fake_direct(**kwargs):
        observed.update(kwargs)
        return {
            "status": "recommended",
            "method": "sbi_snpe",
            "inference_calls": 40,
            "decision_outcome": DecisionOutcome(
                method="sbi_snpe",
                status="recommended",
                decision_policy_identifier="posterior_sample_map_physical_v1",
                method_configuration_identity="internal",
                inference_budget=40,
                inference_calls=PhysicalCallCounts(40, 40),
                recommended_theta=(1.0, 1.0, 1.0, 0.5),
                decision_metadata={"candidate_count": 10000},
            ),
        }

    monkeypatch.setattr(
        callback_module, "run_direct_posterior_sample_map_target", fake_direct
    )
    context = {
        "root_seed": 1,
        "coordinate": {
            "namespace": "method_inference",
            "coordinates": ["target-snpe", "sbi_snpe"],
        },
    }
    outcome = callbacks.run_method(declaration, target, context, lambda *_: [0.0] * 20)
    assert outcome.status == "recommended"
    assert outcome.inference_calls.attempted == 40
    assert observed["method"] == "sbi_snpe"
    assert observed["config"]["sbi"]["sequential_round_allocation"] == [14, 13, 13]


def test_both_method_bindings_use_only_the_supplied_capped_callback(tmp_path, monkeypatch):
    population = load_population_declaration_v3(json.loads(DECLARATION.read_text()))
    callbacks = GraphPopulationCallbacks(
        mutaco_configuration=MUTACO_CONFIG,
        npe_configuration=NPE_CONFIG,
        output=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    target = {
        "target_record_id": population.target_record_identities[0],
        "target_index": 0,
        "target_family_id": "frozen-target-family",
        "target_seed": 11,
        "theta_star": [1.0, 1.0, 1.0, 0.5],
        "target_response": [0.0] * 20,
    }
    observed = []
    dispatch = []

    def physical(theta, context):
        observed.append(context["simulation_context"])
        return [0.0] * 20

    def fake_mutaco(**kwargs):
        dispatch.append(("mutaco", kwargs["config"], kwargs["campaign_science"].value))
        simulator = kwargs["adapter"].build_simulator(
            kwargs["config"], device=kwargs["device"], dtype=kwargs["dtype"]
        )
        simulator.evaluate(
            torch.tensor([1.0, 1.0, 1.0, 0.5], dtype=torch.float32),
            SimulationContext("mutaco-run", "inference", 0, 0, 12),
        )
        return {
            "status": "completed",
            "method_identity": "mutaco-execution",
            "inference_calls": 1,
            "recommended_theta": [1.0, 1.0, 1.0, 0.5],
            "graph_terminal_recommendation": {"policy": "physical_mae"},
        }

    def fake_npe(**kwargs):
        dispatch.append(("sbi_npe", kwargs["config"], kwargs["campaign_science"].value))
        simulator = kwargs["adapter"].build_simulator(
            kwargs["config"], device=kwargs["device"], dtype=kwargs["dtype"]
        )
        simulator.evaluate(
            torch.tensor([1.0, 1.0, 1.0, 0.5], dtype=torch.float32),
            SimulationContext("npe-run", "sbi_training", 0, 0, 13),
        )
        decision = DecisionOutcome(
            method="sbi_npe",
            status="recommended",
            decision_policy_identifier="posterior_sample_map_physical_v1",
            method_configuration_identity="internal-replaced-by-binding",
            inference_budget=40,
            inference_calls=PhysicalCallCounts(1, 1),
            recommended_theta=(1.0, 1.0, 1.0, 0.5),
            decision_metadata={"candidate_count": 10000},
        )
        return {
            "status": "completed",
            "method_identity": "npe-execution",
            "inference_calls": 1,
            "decision_outcome": decision,
        }

    monkeypatch.setattr(callback_module, "run_mutaco_target", fake_mutaco)
    monkeypatch.setattr(callback_module, "run_npe_sample_map_target", fake_npe)
    for declaration in population.methods:
        context = method_inference_context(
            population.comparison,
            target["target_record_id"],
            declaration.method,
            declaration.configuration_identity,
        )
        outcome = callbacks.run_method(declaration, target, context, physical)
        assert outcome.method_configuration_identity == declaration.configuration_identity
        assert outcome.inference_calls.attempted == 1
    assert [item["purpose"] for item in observed] == ["inference", "sbi_training"]
    assert dispatch == [
        ("mutaco", json.loads(MUTACO_CONFIG.read_text()), target["target_family_id"]),
        ("sbi_npe", json.loads(NPE_CONFIG.read_text()), target["target_family_id"]),
    ]
    assert len(tuple((tmp_path / "raw").rglob("*.json"))) == 2


def test_v3_context_does_not_move_accepted_configuration_or_ambient_rng(tmp_path):
    callbacks = GraphPopulationCallbacks(
        mutaco_configuration=MUTACO_CONFIG,
        npe_configuration=NPE_CONFIG,
        output=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    population = load_population_declaration_v3(json.loads(DECLARATION.read_text()))
    context = method_inference_context(
        population.comparison,
        population.target_record_identities[0],
        population.methods[0].method,
        population.methods[0].configuration_identity,
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    effective = callbacks._effective_configuration("mutaco", context)
    assert effective == json.loads(MUTACO_CONFIG.read_text())
    validate_graph_paper_configuration(effective)
    assert effective["campaign"]["campaign_seed"] != context["root_seed"]
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_raw_evidence_recursively_projects_tuple_and_immutable_mapping(tmp_path):
    callbacks = GraphPopulationCallbacks(
        mutaco_configuration=MUTACO_CONFIG,
        npe_configuration=NPE_CONFIG,
        output=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    row = {
        "posterior_diagnostics": MappingProxyType({
            "quantile_probabilities": (0.05, 0.5, 0.95),
            "nested": (MappingProxyType({"shape": (4, 20)}),),
        }),
        "finite": 1.0,
    }
    callbacks._persist_raw("target-json", "sbi_npe", row)
    persisted = read_json(callbacks._raw_path("target-json", "sbi_npe"))
    assert persisted["posterior_diagnostics"]["quantile_probabilities"] == [
        0.05, 0.5, 0.95
    ]
    assert persisted["posterior_diagnostics"]["nested"] == [{"shape": [4, 20]}]
    callbacks._persist_raw(
        "target-live-shape",
        "sbi_npe",
        {
            "training_diagnostics": [
                {"epoch_durations_seconds": (0.25, 0.5)}
            ]
        },
    )
    assert read_json(callbacks._raw_path("target-live-shape", "sbi_npe")) == {
        "training_diagnostics": [
            {"epoch_durations_seconds": [0.25, 0.5]}
        ]
    }
    with pytest.raises(ValueError, match="unsupported strict JSON type object"):
        callbacks._persist_raw("target-object", "sbi_npe", {"bad": object()})
    with pytest.raises(GraphRawEvidencePersistenceError) as staged:
        callbacks._persist_raw_staged("target-staged", "sbi_npe", {"bad": object()})
    assert staged.value.failure_stage == "post_decision_raw_evidence_persistence"
    with pytest.raises(ValueError, match="non-finite"):
        callbacks._persist_raw("target-nan", "sbi_npe", {"bad": float("nan")})


def test_inference_contexts_are_method_addressed_and_shards_cover_once():
    declaration = load_population_declaration_v3(json.loads(DECLARATION.read_text()))
    target = declaration.target_record_identities[0]
    contexts = [
        method_inference_context(
            declaration.comparison, target, method.method, method.configuration_identity
        )
        for method in declaration.methods
    ]
    assert contexts[0] != contexts[1]
    assert contexts[0]["coordinate"]["coordinates"] == [target, "mutaco"]
    assert contexts[1]["coordinate"]["coordinates"] == [target, "sbi_npe"]
    shards = [shard_indices(120, rank=rank, workers=40) for rank in range(40)]
    assert sorted(index for shard in shards for index in shard) == list(range(120))
    assert all(len(shard) == 3 for shard in shards)


def test_mapping_valued_inference_contexts_have_stable_distinctness():
    first = {"coordinate": {"namespace": "method_inference", "coordinates": ["t", "a"]}}
    second = {"coordinate": {"namespace": "method_inference", "coordinates": ["t", "b"]}}
    assert _inference_contexts_are_distinct([first])
    assert _inference_contexts_are_distinct([first, second])
    assert not _inference_contexts_are_distinct([first, dict(first)])


def test_worker_terminal_retains_post_population_failure_without_calls(tmp_path):
    failure = RuntimeError("qualification stopped at 81 of 82 calls")
    _write_worker_terminal(
        tmp_path,
        status="failed",
        phase="exact_successful_cap_validation",
        source_commit="a" * 40,
        declaration_sha256="b" * 64,
        physical_calls={
            "maximum_calls": 82,
            "attempted_calls": 81,
            "successful_calls": 81,
            "failed_calls": 0,
        },
        failure=failure,
    )
    terminal = read_json(tmp_path / "worker_terminal.json")
    assert terminal["status"] == "failed"
    assert terminal["phase"] == "exact_successful_cap_validation"
    assert terminal["physical_calls"]["successful_calls"] == 81
    assert terminal["failure_type"] == "RuntimeError"
    assert terminal["scientific_calls_added_by_terminal_evidence"] == 0
    assert (tmp_path / "worker_terminal.sha256").read_text().strip() == file_hash(
        tmp_path / "worker_terminal.json"
    )


def test_worker_uses_packages_from_exact_source_checkout():
    origins = _validate_source_module_origins(ROOT)
    assert Path(origins["mutaco"]).is_relative_to(ROOT)
    assert Path(origins["mutaco_graph"]).is_relative_to(ROOT)
    worker = (ROOT / "infrastructure/truba/run_paper_v3_worker.sh").read_text()
    assert '${ROOT}/src' in worker
    assert '${ROOT}/simulators/graph-simulator/src' in worker


def test_truba_population_uses_the_production_worker():
    worker = (ROOT / "infrastructure/truba/run_paper_v3_worker.sh").read_text()
    campaign = (ROOT / "infrastructure/truba/paper_v3_population_template.sbatch").read_text()
    assert "experiments.graph_inversion.run_paper_v3_worker" in worker
    assert "MUTACO_V3_MAXIMUM_CALLS" in worker
    assert "INERT TEMPLATE" not in campaign and "exit 64" not in campaign
    assert ': "${MUTACO_V3_MAXIMUM_CALLS:' in campaign
    assert 'srun --ntasks="${MUTACO_V3_WORKERS}"' in campaign
    assert "run_paper120_v3_worker.sh" in campaign
    assert "collect_paper120_v3" in campaign
    assert "experiments.paper.workflow.cli ingest export" in campaign
    assert "compact-export/paper-v3-${SLURM_JOB_ID:-local}" in campaign
    assert "MUTACO_V3_METHOD_SPECS" in worker
    assert '--method-specs "${MUTACO_V3_METHOD_SPECS}"' in worker
    assert "--mutaco-config" not in worker
    assert "--npe-config" not in worker
    assert "--snpe-config" not in worker
    runner_files = sorted(
        path.name
        for path in (ROOT / "experiments/graph_inversion").glob("run_paper_v3*worker.py")
    )
    assert runner_files == ["run_paper_v3_worker.py"]
    source = (ROOT / "experiments/graph_inversion/run_paper_v3_worker.py").read_text()
    assert "standalone snpe" not in source.lower()
    assert "_inference_contexts_are_distinct(inference_contexts)" in source
