from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

import experiments.paper.common.target_freeze_v2 as target_freeze_module
import experiments.paper.common.population_v3 as population_v3_module
from experiments.graph_inversion.run_paper_v3_worker import _validate_thread_configuration
from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter
from experiments.benchmarking.artifacts import file_hash
from experiments.paper.common.aggregate import PaperRecordSet, summarize_population_v3
from experiments.paper.common.cohort_v2 import (
    CohortDeclarationSpec,
    materialize_cohort_declaration,
    validate_cohort_declaration,
)
from experiments.paper.common.population_v3 import (
    MethodDeclarationV3,
    PopulationDeclarationV3,
    collect_population_v3,
    execute_population_v3,
    population_call_preview,
    load_population_declaration_v3,
    shard_indices,
)
from experiments.paper.common.protocol_v3 import (
    AssessmentEvaluationV3,
    DecisionOutcome,
    PaperComparisonConfigV3,
    PhysicalCallCounts,
)
from experiments.graph_inversion.paper_v3_callbacks import (
    _assessment_randomness_evidence,
)
from experiments.paper.common.target_freeze_v2 import (
    collect_target_freeze,
    execute_target_freeze,
    preview_target_freeze,
)
from experiments.paper.common.validation import PaperContractError
from infrastructure.truba.paper_v3_layout import BarbunV3Layout


ROOT = Path(__file__).resolve().parents[2]


def test_worker_thread_guard_accepts_profile_driven_five_threads(monkeypatch):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "5")

    _validate_thread_configuration(intraop_threads=5, interop_threads=1)


def test_worker_thread_guard_rejects_environment_mismatch(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "5")

    with pytest.raises(SystemExit, match="matching Torch intra-op threads \\(5\\)"):
        _validate_thread_configuration(intraop_threads=5, interop_threads=1)


def test_graph_target_only_config_matches_campaign_config_without_method_policy():
    campaign = json.loads(
        (ROOT / "experiments/graph_inversion/configs/d4_100_targets.json").read_text()
    )
    target_only = {
        "dtype": campaign["dtype"],
        "device": "cpu",
        "simulator": campaign["simulator"],
    }
    adapter = GraphScientificD4Adapter()
    target_config = adapter.target_graph_config(
        target_only, device=torch.device("cpu"), dtype=torch.float32
    )
    campaign_config = adapter._graph_config(
        campaign, device=torch.device("cpu"), dtype=torch.float32
    )
    assert target_config == campaign_config
    assert target_config.identity_document() == campaign_config.identity_document()
    assert "comparison" not in target_only and "runner" not in target_only


def _declaration(count: int = 3):
    return materialize_cohort_declaration(CohortDeclarationSpec(
        cohort_id="synthetic-cohort",
        simulator_adapter="synthetic-adapter",
        simulator_definition_id="synthetic-simulator-v1",
        simulator_configuration_identity="synthetic-config-v1",
        simulator_configuration_sha256="0" * 64,
        parameter_profile="p2-test",
        parameter_names=("a", "b"),
        bounds=((0.0, -1.0), (1.0, 2.0)),
        target_count=count,
        campaign_seed=1234,
        target_seed_namespace="synthetic_target",
    ))


def test_generic_cohort_is_exactly_regenerated_and_rejects_mutation():
    declaration = _declaration()
    assert validate_cohort_declaration(declaration) == declaration
    assert [row["sobol_index"] for row in declaration["records"]] == [0, 1, 2]
    altered = json.loads(json.dumps(declaration))
    altered["records"][1]["theta_star"][0] += 0.01
    with pytest.raises(PaperContractError, match="exact regeneration"):
        validate_cohort_declaration(altered)


def test_target_preview_and_collection_are_zero_call(tmp_path):
    calls = []
    declaration = _declaration()
    preview = preview_target_freeze(declaration, tmp_path, maximum_calls=3)
    assert preview.planned_calls == 3
    assert preview.document()["scientific_calls_executed"] == 0
    manifest = collect_target_freeze(declaration, tmp_path)
    assert manifest["missing_count"] == 3
    assert manifest["scientific_calls_executed_by_collection"] == 0
    assert calls == []


def test_target_only_surface_has_no_method_dispatch():
    sources = (
        ROOT / "experiments/paper/protocol/target_freeze_v2.py",
        ROOT / "experiments/graph_inversion/freeze_targets_v2.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for forbidden in ("run_mutaco_target", "run_npe", "run_sbi", "methods_to_execute"):
        assert forbidden not in text
    assert "--torch-intraop-threads" in text
    assert "--torch-interop-threads" in text


def test_target_cap_fails_before_first_call(tmp_path):
    calls = []
    with pytest.raises(PaperContractError, match="requires 3 calls"):
        execute_target_freeze(
            _declaration(), tmp_path, maximum_calls=2,
            evaluate_target=lambda row: (calls.append(row) or [1.0], {}),
            execution_provenance={"source_commit": "test"},
        )
    assert calls == []
    assert not (tmp_path / "target_attempts").exists()


def test_target_freeze_is_append_only_and_recovery_is_zero_call(tmp_path):
    calls = []

    def evaluate(row):
        calls.append(row["target_index"])
        return [sum(row["theta_star"]), 2.0], {"adapter": "stub"}

    first = execute_target_freeze(
        _declaration(), tmp_path, maximum_calls=3, evaluate_target=evaluate,
        execution_provenance={"source_commit": "test"},
    )
    assert calls == [0, 1, 2]
    assert first["manifest"]["completed_count"] == 3
    calls.clear()
    second = execute_target_freeze(
        _declaration(), tmp_path, maximum_calls=0, evaluate_target=evaluate,
        execution_provenance={"source_commit": "test"},
    )
    assert calls == []
    assert second["preview"]["planned_target_construction_calls"] == 0
    attempts = sorted((tmp_path / "target_attempts").glob("*.json"))
    assert len(attempts) == 2


def test_target_failure_is_preserved_without_automatic_retry(tmp_path):
    def fail(_row):
        raise RuntimeError("injected target failure")

    with pytest.raises(RuntimeError, match="injected"):
        execute_target_freeze(
            _declaration(1), tmp_path, maximum_calls=1, evaluate_target=fail,
            execution_provenance={"source_commit": "test"},
        )
    attempt = json.loads((tmp_path / "target_attempts/attempt_0000.json").read_text())
    assert attempt["status"] == "failed_preserved_no_automatic_retry"
    assert attempt["calls_attempted"] == attempt["calls_failed"] == 1
    assert attempt["failed_indices"][0]["failure_phase"] == "evaluation"


def test_target_freeze_thaws_nested_strict_metadata_before_persistence(tmp_path):
    metadata = MappingProxyType({
        "simulator": MappingProxyType({"shape": (2, 3), "nested": ("a", 1)}),
        "seed_manifest": MappingProxyType({"thermal_noise/0": 123}),
    })
    execute_target_freeze(
        _declaration(1), tmp_path, maximum_calls=1,
        evaluate_target=lambda _row: ([1.0, 2.0], metadata),
        execution_provenance={"source_commit": "test"},
    )
    record = json.loads((tmp_path / "target_ledger/target_000.json").read_text())
    assert record["simulator_result_metadata"] == {
        "seed_manifest": {"thermal_noise/0": 123},
        "simulator": {"nested": ["a", 1], "shape": [2, 3]},
    }


def test_target_freeze_rejects_invalid_metadata_as_persistence_failure(tmp_path):
    metadata = MappingProxyType({"invalid": object()})
    with pytest.raises(ValueError, match="unsupported strict JSON type object"):
        execute_target_freeze(
            _declaration(1), tmp_path, maximum_calls=1,
            evaluate_target=lambda _row: ([1.0], metadata),
            execution_provenance={"source_commit": "test"},
        )
    attempt = json.loads((tmp_path / "target_attempts/attempt_0000.json").read_text())
    assert attempt["calls_attempted"] == attempt["calls_failed"] == 1
    assert attempt["calls_successful"] == 0
    assert attempt["failed_indices"][0]["failure_phase"] == "persistence"


def test_post_evaluation_write_failure_is_counted_and_stops(monkeypatch, tmp_path):
    original = target_freeze_module._write_hashed

    def fail_record(path, document):
        if path.name == "target_000.json":
            raise OSError("injected record persistence failure")
        original(path, document)

    monkeypatch.setattr(target_freeze_module, "_write_hashed", fail_record)
    calls = []
    with pytest.raises(OSError, match="injected record persistence failure"):
        execute_target_freeze(
            _declaration(2), tmp_path, maximum_calls=2,
            evaluate_target=lambda row: (calls.append(row["target_index"]) or [1.0], {}),
            execution_provenance={"source_commit": "test"},
        )
    assert calls == [0]
    attempt = json.loads((tmp_path / "target_attempts/attempt_0000.json").read_text())
    assert attempt["calls_attempted"] == attempt["calls_failed"] == 1
    assert attempt["calls_successful"] == 0
    assert attempt["failed_indices"][0]["failure_phase"] == "persistence"


def test_target_success_requires_exact_hash_valid_reload(monkeypatch, tmp_path):
    original = target_freeze_module._load_hashed

    def reject_persisted_reload(path):
        loaded = original(path)
        if path.name == "target_000.json" and loaded is not None:
            return None
        return loaded

    monkeypatch.setattr(target_freeze_module, "_load_hashed", reject_persisted_reload)
    with pytest.raises(PaperContractError, match="failed exact persisted reload"):
        execute_target_freeze(
            _declaration(1), tmp_path, maximum_calls=1,
            evaluate_target=lambda _row: ([1.0], {}),
            execution_provenance={"source_commit": "test"},
        )
    attempt = json.loads((tmp_path / "target_attempts/attempt_0000.json").read_text())
    assert attempt["calls_attempted"] == attempt["calls_failed"] == 1
    assert attempt["calls_successful"] == 0
    assert attempt["completed_indices"] == []
    assert attempt["failed_indices"][0]["failure_phase"] == "persistence"


def _comparison():
    return PaperComparisonConfigV3(
        simulator_definition_identity="stub-simulator-v1",
        simulator_configuration_identity="stub-config-v1",
        assessment_fidelity="stub-response-v1",
        assessment_campaign_seed=9002,
        inference_campaign_seed=9001,
    )


def _methods():
    return (
        MethodDeclarationV3("mutaco", "mutaco-test-config", "synthetic-zero-call", 2),
        MethodDeclarationV3("sbi_npe", "npe-test-config", "synthetic-zero-call", 2),
    )


def test_v3_population_freezes_all_decisions_then_pairs_assessment(tmp_path):
    targets = (
        {"target_record_id": "target-a", "theta_star": [0.1, 0.2], "target_response": [1.0, 2.0]},
        {"target_record_id": "target-b", "theta_star": [0.3, 0.4], "target_response": [1.5, 2.5]},
    )
    events = []

    def inference(method, target, theta, context):
        events.append(("inference", target["target_record_id"], method))
        return [float(theta[0]), float(theta[1])]

    def run_method(declaration, target, context, evaluate):
        evaluate([0.2, 0.3], context)
        evaluate([0.3, 0.4], context)
        return DecisionOutcome(
            method=declaration.method,
            status="recommended",
            decision_policy_identifier=declaration.decision_policy_identifier,
            method_configuration_identity=declaration.configuration_identity,
            inference_budget=declaration.inference_budget,
            inference_calls=PhysicalCallCounts(2, 2),
            recommended_theta=(0.2, 0.3),
        )

    def assessment(target, theta, context):
        events.append(("assessment", target["target_record_id"], context["family_identity"]))
        freeze_files = tuple((tmp_path / "decision_freezes").glob("*.json"))
        assert freeze_files
        return [1.1, 2.1]

    result = execute_population_v3(
        comparison=_comparison(), targets=targets, methods=_methods(), output=tmp_path,
        maximum_calls=12, run_method=run_method,
        evaluate_inference=inference, evaluate_assessment=assessment,
    )
    assert result["physical_calls"]["attempted_calls"] == 12
    assert result["physical_calls_by_category"] == {
        "method_inference_or_training": {
            "attempted_calls": 8, "successful_calls": 8, "failed_calls": 0,
        },
        "paired_assessment_and_metric": {
            "attempted_calls": 4, "successful_calls": 4, "failed_calls": 0,
        },
    }
    execution_attempt = json.loads(
        (tmp_path / "population_execution_attempt.json").read_text()
    )
    assert execution_attempt["status"] == "completed"
    assert execution_attempt["physical_calls"] == result["physical_calls"]
    assert result["target_construction_calls"] == result["recommendation_calls"] == 0
    manifest, rows = collect_population_v3(
        tmp_path, target_identities=("target-a", "target-b"),
        methods=("mutaco", "sbi_npe"),
    )
    assert manifest["complete"] and len(rows) == 4
    for target in ("target-a", "target-b"):
        pair = [row for row in rows if row["target_record_identity"] == target]
        assert pair[0]["assessment_context"] == pair[1]["assessment_context"]
        assert pair[0]["decision"]["inference_context"] != pair[1]["decision"]["inference_context"]
    assert all(event[0] == "inference" for event in events[:4])


def test_v3_population_persists_shared_disorder_and_thermal_assessment_evidence(
    tmp_path,
):
    target = {
        "target_record_id": "target-randomness",
        "theta_star": [0.1, 0.2],
        "target_response": [1.0, 2.0],
    }

    def runner(declaration, target, context, evaluate):
        evaluate([0.2, 0.3], context)
        evaluate([0.3, 0.4], context)
        return DecisionOutcome(
            method=declaration.method,
            status="recommended",
            decision_policy_identifier=declaration.decision_policy_identifier,
            method_configuration_identity=declaration.configuration_identity,
            inference_budget=2,
            inference_calls=PhysicalCallCounts(2, 2),
            recommended_theta=(0.2, 0.3),
        )

    manifest = {
        "seed_manifest": {
            "node_parameter/weight/B=0": 1,
            "pairwise_coupling/B=0": 2,
            "initial_state/B=0/R=0": 3,
            "thermal_noise/B=0/R=0/V=0": 4,
        }
    }

    def assess(target, theta, context):
        del target, theta
        evidence = _assessment_randomness_evidence(context, manifest)
        return AssessmentEvaluationV3(
            (1.1, 2.1), evidence["compact"], evidence["raw"]
        )

    execute_population_v3(
        comparison=_comparison(),
        targets=(target,),
        methods=_methods(),
        output=tmp_path,
        maximum_calls=6,
        run_method=runner,
        evaluate_inference=lambda *args: [0.0, 0.0],
        evaluate_assessment=assess,
    )
    _, rows = collect_population_v3(
        tmp_path,
        target_identities=(target["target_record_id"],),
        methods=("mutaco", "sbi_npe"),
    )
    evidence = [row["assessment_randomness_evidence"] for row in rows]
    assert evidence[0]["coordinate_family_identities"] == evidence[1][
        "coordinate_family_identities"
    ]
    assert evidence[0]["namespace_seed_manifest_hashes"] == evidence[1][
        "namespace_seed_manifest_hashes"
    ]
    assert "namespace_seed_manifests" not in evidence[0]
    raw_files = tuple((tmp_path / "assessment_randomness").rglob("*.json"))
    assert len(raw_files) == 2
    raw = json.loads(raw_files[0].read_text())
    assert set(raw["namespace_seed_manifests"]) == {
        "quenched_disorder", "initial_state", "thermal_noise"
    }


def test_population_preview_and_shards_are_computation_free():
    assert population_call_preview(120, _methods()) == {
        "target_construction_calls": 0,
        "maximum_inference_calls": 480,
        "maximum_assessment_calls": 240,
        "maximum_total_calls": 720,
        "recommendation_calls": 0,
        "scientific_calls_executed": 0,
    }
    shards = [shard_indices(120, rank=rank, workers=40) for rank in range(40)]
    assert sorted(index for shard in shards for index in shard) == list(range(120))
    assert all(len(shard) == 3 for shard in shards)


def test_population_exact_cap_fails_before_method_dispatch(tmp_path):
    calls = []
    with pytest.raises(PaperContractError, match="exact declared maximum"):
        execute_population_v3(
            comparison=_comparison(),
            targets=({"target_record_id": "ta", "theta_star": [0.0], "target_response": [0.0]},),
            methods=_methods(), output=tmp_path, maximum_calls=5,
            run_method=lambda *args: calls.append(args),
            evaluate_inference=lambda *args: calls.append(args),
            evaluate_assessment=lambda *args: calls.append(args),
        )
    assert calls == []
    evidence = json.loads((tmp_path / "population_execution_attempt.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["physical_calls"]["attempted_calls"] == 0


def test_population_persists_terminal_ledger_when_decision_freeze_fails(
    tmp_path, monkeypatch
):
    target = {"target_record_id": "ta", "theta_star": [0.0], "target_response": [0.0]}

    def runner(declaration, target, context, evaluate):
        evaluate([0.0], context)
        evaluate([0.0], context)
        return DecisionOutcome(
            method=declaration.method,
            status="recommended",
            decision_policy_identifier=declaration.decision_policy_identifier,
            method_configuration_identity=declaration.configuration_identity,
            inference_budget=2,
            inference_calls=PhysicalCallCounts(2, 2),
            recommended_theta=(0.0,),
        )

    monkeypatch.setattr(
        population_v3_module,
        "freeze_decisions",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("freeze fault")),
    )
    with pytest.raises(RuntimeError, match="freeze fault"):
        execute_population_v3(
            comparison=_comparison(), targets=(target,), methods=_methods(),
            output=tmp_path, maximum_calls=6, run_method=runner,
            evaluate_inference=lambda *args: [0.0],
            evaluate_assessment=lambda *args: [0.0],
            execution_provenance={"source_commit": "test-commit", "declaration_identity": "test-declaration"},
        )
    evidence = json.loads((tmp_path / "population_execution_attempt.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["last_completed_phase"] == "method_attempt_persistence"
    assert evidence["physical_calls"] == {
        "maximum_calls": 6, "attempted_calls": 4, "successful_calls": 4,
        "failed_calls": 0,
    }
    assert evidence["execution_provenance"]["source_commit"] == "test-commit"


def test_population_counts_contained_assessment_failures(tmp_path):
    target = {"target_record_id": "ta", "theta_star": [0.0], "target_response": [0.0]}

    def runner(declaration, target, context, evaluate):
        evaluate([0.0], context)
        evaluate([0.0], context)
        return DecisionOutcome(
            method=declaration.method,
            status="recommended",
            decision_policy_identifier=declaration.decision_policy_identifier,
            method_configuration_identity=declaration.configuration_identity,
            inference_budget=2,
            inference_calls=PhysicalCallCounts(2, 2),
            recommended_theta=(0.0,),
        )

    result = execute_population_v3(
        comparison=_comparison(), targets=(target,), methods=_methods(),
        output=tmp_path, maximum_calls=6, run_method=runner,
        evaluate_inference=lambda *args: [0.0],
        evaluate_assessment=lambda *args: (_ for _ in ()).throw(RuntimeError("assessment fault")),
    )
    assert result["completed_slots"] == 0 and result["failed_slots"] == 2
    assert result["physical_calls_by_category"]["paired_assessment_and_metric"] == {
        "attempted_calls": 2, "successful_calls": 0, "failed_calls": 2,
    }


def test_population_counts_mid_inference_failures_without_assessment(tmp_path):
    target = {"target_record_id": "ta", "theta_star": [0.0], "target_response": [0.0]}

    def runner(declaration, target, context, evaluate):
        evaluate([0.0], context)
        raise AssertionError("unreachable after injected evaluation failure")

    result = execute_population_v3(
        comparison=_comparison(), targets=(target,), methods=_methods(),
        output=tmp_path, maximum_calls=6, run_method=runner,
        evaluate_inference=lambda *args: (_ for _ in ()).throw(RuntimeError("inference fault")),
        evaluate_assessment=lambda *args: pytest.fail("failed decisions are not assessed"),
    )
    assert result["completed_slots"] == 0 and result["failed_slots"] == 2
    assert result["physical_calls"] == {
        "maximum_calls": 6, "attempted_calls": 2, "successful_calls": 0,
        "failed_calls": 2,
    }
    assert result["physical_calls_by_category"]["paired_assessment_and_metric"][
        "attempted_calls"
    ] == 0
    attempts = [json.loads(path.read_text()) for path in (tmp_path / "attempts_v3").rglob("*.json")]
    assert {item["failure_stage"] for item in attempts} == {"method_dispatch"}
    assert {item["failure_type"] for item in attempts} == {"RuntimeError"}


def test_population_persists_terminal_ledger_when_method_record_write_fails(
    tmp_path, monkeypatch
):
    target = {"target_record_id": "ta", "theta_star": [0.0], "target_response": [0.0]}

    def runner(declaration, target, context, evaluate):
        evaluate([0.0], context)
        evaluate([0.0], context)
        return DecisionOutcome(
            method=declaration.method,
            status="recommended",
            decision_policy_identifier=declaration.decision_policy_identifier,
            method_configuration_identity=declaration.configuration_identity,
            inference_budget=2,
            inference_calls=PhysicalCallCounts(2, 2),
            recommended_theta=(0.0,),
        )

    monkeypatch.setattr(
        population_v3_module,
        "append_method_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("record fault")),
    )
    with pytest.raises(RuntimeError, match="record fault"):
        execute_population_v3(
            comparison=_comparison(), targets=(target,), methods=_methods(),
            output=tmp_path, maximum_calls=6, run_method=runner,
            evaluate_inference=lambda *args: [0.0],
            evaluate_assessment=lambda *args: [0.0],
        )
    evidence = json.loads((tmp_path / "population_execution_attempt.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["last_completed_phase"] == "assessment_and_record_construction"
    assert evidence["physical_calls"] == {
        "maximum_calls": 6, "attempted_calls": 6, "successful_calls": 6,
        "failed_calls": 0,
    }


def test_population_stub_success_path_is_exactly_40_plus_40_plus_2(tmp_path):
    methods = (
        MethodDeclarationV3("mutaco", "mc", "synthetic-zero-call", 40),
        MethodDeclarationV3("sbi_npe", "nc", "synthetic-zero-call", 40),
    )
    target = {"target_record_id": "ta", "theta_star": [0.0], "target_response": [0.0]}

    def runner(declaration, target, context, evaluate):
        for _ in range(40):
            evaluate([0.0], context)
        return DecisionOutcome(
            method=declaration.method,
            status="recommended",
            decision_policy_identifier=declaration.decision_policy_identifier,
            method_configuration_identity=declaration.configuration_identity,
            inference_budget=40,
            inference_calls=PhysicalCallCounts(40, 40),
            recommended_theta=(0.0,),
        )

    result = execute_population_v3(
        comparison=_comparison(), targets=(target,), methods=methods,
        output=tmp_path, maximum_calls=82, run_method=runner,
        evaluate_inference=lambda *args: [0.0],
        evaluate_assessment=lambda *args: [0.0],
    )
    assert result["physical_calls"] == {
        "maximum_calls": 82, "attempted_calls": 82, "successful_calls": 82,
        "failed_calls": 0,
    }
    assert result["physical_calls_by_category"] == {
        "method_inference_or_training": {
            "attempted_calls": 80, "successful_calls": 80, "failed_calls": 0,
        },
        "paired_assessment_and_metric": {
            "attempted_calls": 2, "successful_calls": 2, "failed_calls": 0,
        },
    }


def test_population_declaration_is_explicit_v3_and_has_no_v2_fallback():
    declaration = PopulationDeclarationV3(
        campaign_id="synthetic-v3",
        comparison=_comparison(),
        target_ledger_hash="ledger-hash",
        target_record_identities=("ta", "tb"),
        methods=_methods(),
        output_root="runs/synthetic-v3",
    ).document()
    assert load_population_declaration_v3(declaration).document() == declaration
    legacy = dict(declaration)
    legacy["schema_version"] = "mutaco-paper-campaign-declaration-v1"
    with pytest.raises(PaperContractError, match="explicit v3"):
        load_population_declaration_v3(legacy)


def test_v3_population_statistics_are_deterministic_and_keep_denominators(tmp_path):
    # Reuse the strict v3 execution test to generate two complete targets.
    targets = (
        {"target_record_id": "ta", "theta_star": [0.0], "target_response": [0.0, 0.0]},
        {"target_record_id": "tb", "theta_star": [0.0], "target_response": [0.0, 0.0]},
    )
    methods = (
        MethodDeclarationV3("mutaco", "mc", "synthetic-zero-call", 1),
        MethodDeclarationV3("sbi_npe", "nc", "synthetic-zero-call", 1),
    )

    def runner(declaration, target, context, evaluate):
        evaluate([0.0], context)
        return DecisionOutcome(
            method=declaration.method, status="recommended",
            decision_policy_identifier=declaration.decision_policy_identifier,
            method_configuration_identity=declaration.configuration_identity,
            inference_budget=1, inference_calls=PhysicalCallCounts(1, 1),
            recommended_theta=(0.0,),
        )

    execute_population_v3(
        comparison=_comparison(), targets=targets, methods=methods, output=tmp_path,
        maximum_calls=8, run_method=runner,
        evaluate_inference=lambda method, target, theta, context: [0.0, 0.0],
        evaluate_assessment=lambda target, theta, context: [0.1, 0.2],
    )
    _, rows = collect_population_v3(tmp_path, target_identities=("ta", "tb"), methods=("mutaco", "sbi_npe"))
    record_set = PaperRecordSet(tmp_path, rows)
    first = summarize_population_v3(
        record_set, declared_target_identities=("ta", "tb", "missing"),
        bootstrap_seed=77, bootstrap_resamples=100,
    )
    second = summarize_population_v3(
        record_set, declared_target_identities=("ta", "tb", "missing"),
        bootstrap_seed=77, bootstrap_resamples=100,
    )
    assert first == second
    assert first["methods"]["mutaco"]["missing_count"] == 1
    assert first["paired"][0]["complete_pair_count"] == 2
    assert first["paired"][0]["excluded_pair_count"] == 1
    assert first["paired"][0]["excluded_pairs"][0]["reasons"][0]["reason"] == "missing_record"
    assert first["methods"]["mutaco"]["call_ledger"]["terminal_recommendation"]["attempted_calls"] == 0
    assert first["methods"]["mutaco"]["call_ledger"]["method_inference_or_training"]["attempted_calls"] == 2
    assert first["methods"]["mutaco"]["timing"]["paired_held_out_assessment"]["wall_seconds"] is not None
    assert first["scientific_calls"] == 0


def test_population_v3_summary_has_public_paper_import():
    from experiments.paper import summarize_population_v3 as public_summary

    assert public_summary is summarize_population_v3


def test_truba_v3_campaign_resources_and_worker_geometry_are_exact():
    job = (ROOT / "infrastructure/truba/paper_v3_population_template.sbatch").read_text()
    worker = (ROOT / "infrastructure/truba/run_paper_v3_worker.sh").read_text()
    assert "#SBATCH --nodes=8" in job
    assert "#SBATCH --exclude=barbun[112-119]" in job
    assert "#SBATCH --ntasks=40" in job
    assert "#SBATCH --ntasks-per-node=5" in job
    assert "#SBATCH --cpus-per-task=8" in job
    assert "#SBATCH --mem=64G" in job
    assert "INERT TEMPLATE" not in job and "exit 64" not in job
    assert ': "${MUTACO_V3_MAXIMUM_CALLS:' in job
    assert 'srun --ntasks="${MUTACO_V3_WORKERS}"' in job
    assert '--cpus-per-task="${MUTACO_V3_SBATCH_CPUS_PER_TASK}"' in job
    assert "collect_paper120_v3" in job
    assert "INERT TEMPLATE" not in worker
    assert "--execute --confirm-approved" in worker
    assert "MUTACO_EXPECTED_COMMIT" in worker
    assert "MUTACO_V3_MAXIMUM_CALLS" in worker
    assert "--execute --confirm-approved" in worker
    assert "OMP_NUM_THREADS" in worker and "MKL_NUM_THREADS" in worker
    assert '--torch-intraop-threads "${SLURM_CPUS_PER_TASK}"' in worker
    layout = BarbunV3Layout().validate(120)
    assert layout["workers"] == 40 and layout["waves"] == 3
    assert layout["scheduler_queried"] is layout["job_submitted"] is False
