from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.graph_inversion import sbi_zoo_p1 as p1
from experiments.benchmarking.methods.sbi_zoo_accounting import WriteAheadScientificCallLedger
from experiments.benchmarking.methods.sbi_zoo_contract import BUDGETS
from experiments.benchmarking.methods.sbi_zoo_failures import (
    PhysicalSimulatorTripwireTriggered,
    ScientificBudgetViolation,
)
from infrastructure.truba.lifecycle import _write_receipt


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3"
CONFIG = ROOT / "experiments/graph_inversion/configs/paper/paper120_v3_target_config.json"


@pytest.fixture(autouse=True)
def physical_simulator_tripwire(monkeypatch: pytest.MonkeyPatch):
    from mutaco_graph.scientific import OverdampedScientificSimulator
    from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter
    from experiments.graph_inversion.paper_v3_callbacks import _CallbackGraphAdapter, _CallbackSimulator

    state = {"triggered": False, "expected": False}

    def guard(*args, **kwargs):
        del args, kwargs
        state["triggered"] = True
        raise PhysicalSimulatorTripwireTriggered("P1 zero-call simulator tripwire")

    for cls, name in (
        (OverdampedScientificSimulator, "__post_init__"),
        (OverdampedScientificSimulator, "evaluate"),
        (OverdampedScientificSimulator, "evaluate_batch"),
        (GraphScientificD4Adapter, "build_simulator"),
        (GraphScientificD4Adapter, "build_target_simulator"),
        (_CallbackGraphAdapter, "build_simulator"),
        (_CallbackSimulator, "evaluate"),
        (_CallbackSimulator, "evaluate_batch"),
    ):
        monkeypatch.setattr(cls, name, guard)
    yield state
    assert state["triggered"] is state["expected"]


def _receipt(tmp_path: Path, source: str = "a" * 40) -> Path:
    path = tmp_path / "lifecycle.json"
    _write_receipt(
        path,
        {
            "schema_version": "mutaco-truba-validation-receipt-v1",
            "status": "validated_zero_call",
            "source_commit": source,
            "source_materialization": {
                "status": "materialized_clean",
                "source_commit": source,
            },
            "worker_preview": {
                "schema_version": p1.P1_PREFLIGHT_SCHEMA,
                "p1_execution_source": source,
                "scientific_calls_executed": 0,
                "job_submitted": False,
                "declaration_published": False,
            },
            "scientific_calls": 0,
            "job_submitted": False,
            "submission_authorized": False,
        },
    )
    return path


def _declaration(tmp_path: Path) -> dict:
    return p1.build_declaration(
        cohort_root=COHORT,
        simulator_configuration_path=CONFIG,
        output_root=tmp_path / "workers",
        execution_source="a" * 40,
        execution_tree="b" * 40,
        lifecycle_receipt_path=_receipt(tmp_path),
    )


def _successful_receipts(document: dict) -> list[dict]:
    return [
        {
            "target_position": spec["target_position"],
            "target_record_id": spec["target_record_id"],
            "budget": spec["budget"],
            "dataset_identity": f"dataset-{spec['target_position']}-{spec['budget']}",
            "canonical_content_sha256": f"{spec['target_position']:02x}{spec['budget']:02x}".ljust(64, "0"),
            "freeze_identity": f"freeze-{spec['target_position']}-{spec['budget']}",
            "accounting": {
                "maximum_calls": spec["budget"],
                "attempted_calls": spec["budget"],
                "successful_calls": spec["budget"],
                "failed_calls": 0,
                "attempted_outcome_unknown": 0,
                "retry_calls": 0,
            },
        }
        for spec in document["ordered_dataset_declarations"]
    ]


def test_cohort_declaration_enumeration_and_randomness_are_exact(tmp_path):
    first = _declaration(tmp_path)
    second = _declaration(tmp_path / "second")
    # Output roots are provenance, so declarations differ, while every frozen
    # cohort and coordinate-addressing rule remains identical.
    assert first["cohort"] == second["cohort"]
    assert first["cohort"]["target_count"] == 120
    assert [
        target["logical_worker_index"] for target in first["cohort"]["targets"]
    ] == list(range(120))
    assert [
        target["physical_array_task_index"] for target in first["cohort"]["targets"]
    ] == list(range(60)) + list(range(60))
    assert [
        target["physical_task_sequence_index"] for target in first["cohort"]["targets"]
    ] == [0] * 60 + [1] * 60
    assert first["execution"]["physical_task_mapping"] == p1.physical_task_mapping()
    assert first["execution"]["slurm"] == p1.SLURM_GEOMETRY
    assert first["execution"]["collector_slurm"] == p1.COLLECTOR_SLURM_GEOMETRY
    datasets = first["ordered_dataset_declarations"]
    assert len(datasets) == 240
    assert sum(item["budget"] == 40 for item in datasets) == 120
    assert sum(item["budget"] == 100 for item in datasets) == 120
    slots = [slot for item in datasets for slot in item["slots"]]
    assert len(slots) == 16_800
    assert len({slot["scientific_slot_identity"] for slot in slots}) == 16_800
    b40, b100 = datasets[:2]
    assert b40["target_record_id"] == b100["target_record_id"]
    assert b40["theta_seed"] != b100["theta_seed"]
    assert [s["theta_canonical_sha256"] for s in b40["slots"]] != [
        s["theta_canonical_sha256"] for s in b100["slots"][:40]
    ]
    assert first["heldout_firewall"] == {
        "heldout_calls": 0,
        "heldout_accesses": 0,
        "repaired_family_roots_forbidden": [92021540, 92021600],
    }
    assert p1.validate_declaration(first) == first


def test_declaration_roundtrip_lifecycle_binding_and_zero_call_preview(tmp_path):
    document = _declaration(tmp_path)
    path = tmp_path / "declaration" / "declaration.json"
    p1.write_declaration(path, document)
    assert p1.load_declaration(path) == document
    with pytest.raises(p1.P1ExecutionError, match="append-only"):
        p1.write_declaration(path, document)
    preview = p1.preview(document)
    assert preview["paper120_targets"] == 120
    assert preview["total_slots"] == 16_800
    assert preview["scientific_calls_executed"] == 0
    assert preview["heldout_accesses"] == 0
    assert preview["slurm"]["job_exclusive"] is False
    assert preview["collector_slurm"]["job_exclusive"] is False

    wrong = _receipt(tmp_path / "wrong", source="c" * 40)
    with pytest.raises(p1.P1ExecutionError, match="mismatched"):
        p1.build_declaration(
            cohort_root=COHORT,
            simulator_configuration_path=CONFIG,
            output_root=tmp_path / "wrong-workers",
            execution_source="a" * 40,
            execution_tree="b" * 40,
            lifecycle_receipt_path=wrong,
        )


def test_population_gate_requires_exactly_240_complete_datasets(tmp_path):
    document = _declaration(tmp_path)
    receipts = _successful_receipts(document)
    passed = p1.evaluate_population_receipts(document, receipts)
    assert passed["status"] == "PASS"
    assert passed["p2_gate"] == "OPEN"
    assert passed["attempted_calls"] == 16_800
    assert passed["successful_calls"] == 16_800

    assert p1.evaluate_population_receipts(document, receipts[:-1])["p2_gate"] == "BLOCKED"
    assert p1.evaluate_population_receipts(document, receipts + [receipts[0]])["status"] == "BLOCKED"
    wrong_target = [dict(item) for item in receipts]
    wrong_target[0]["target_record_id"] = "wrong"
    assert p1.evaluate_population_receipts(document, wrong_target)["status"] == "BLOCKED"
    failed = [dict(item) for item in receipts]
    failed[0] = {**failed[0], "accounting": {**failed[0]["accounting"], "successful_calls": 39, "failed_calls": 1}}
    assert p1.evaluate_population_receipts(document, failed)["status"] == "BLOCKED"
    unknown = [dict(item) for item in receipts]
    unknown[0] = {**unknown[0], "accounting": {**unknown[0]["accounting"], "successful_calls": 39, "attempted_outcome_unknown": 1}}
    assert p1.evaluate_population_receipts(document, unknown)["status"] == "BLOCKED"


def test_restart_only_allows_new_or_intent_only_slots(tmp_path):
    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)
    intent = {"identity": "slot-0"}
    ledger.persist_intent("row-000", intent)
    calls = []
    assert ledger.invoke_unattempted_once(
        "row-000",
        intent=intent,
        invoke=lambda: calls.append("called") or torch.zeros(20),
        persist_outcome=lambda value: {"persisted": bool(value.shape == (20,))},
    ).shape == (20,)
    assert calls == ["called"]
    with pytest.raises(ScientificBudgetViolation, match="non-retryable"):
        ledger.invoke_unattempted_once(
            "row-000", intent=intent, invoke=lambda: None, persist_outcome=lambda value: {}
        )
    ledger.persist_intent("row-001", {"identity": "slot-1"})
    ledger.mark_attempted("row-001")
    with pytest.raises(ScientificBudgetViolation, match="non-retryable"):
        ledger.invoke_unattempted_once(
            "row-001",
            intent={"identity": "slot-1"},
            invoke=lambda: None,
            persist_outcome=lambda value: {},
        )
    assert ledger.slot_status("row-001") == "attempted_outcome_unknown"
    assert ledger.accounting()["attempted_calls"] == 2


def test_mock_target_freezes_b40_before_b100_failure(tmp_path, monkeypatch):
    document = _declaration(tmp_path)
    monkeypatch.setattr(p1, "_require_exact_clean_source", lambda document: None)
    response_spec = SimpleNamespace(
        axes=tuple(SimpleNamespace(name=name) for name in p1.RESPONSE_NAMES)
    )
    fake_simulator = SimpleNamespace(
        capabilities=SimpleNamespace(simulator_identity=p1.SIMULATOR_ID, response_spec=response_spec)
    )
    calls = []

    def fake_checked(simulator, theta, *, device, dtype, context):
        del simulator, device, dtype
        calls.append(context)
        if len(calls) == 41:
            raise RuntimeError("synthetic B100 invocation failure")
        return SimpleNamespace(values=(torch.arange(20, dtype=torch.float32) + theta.sum()).contiguous())

    monkeypatch.setattr(p1, "checked_evaluate", fake_checked)
    with pytest.raises(RuntimeError, match="synthetic B100"):
        p1.execute_target(document, target_index=0, simulator_factory=lambda config: fake_simulator)
    b40_root = tmp_path / "workers/targets/target-000/b40"
    assert (b40_root / "freeze.json").is_file()
    assert p1._validate_dataset(document, document["ordered_dataset_declarations"][0])["budget"] == 40
    b100_spec = document["ordered_dataset_declarations"][1]
    b100_ledger = WriteAheadScientificCallLedger(
        tmp_path / "workers/targets/target-000/b100/ledger",
        b100_spec["dataset_generation_declaration_identity"],
        100,
    )
    assert b100_ledger.accounting()["failed_calls"] == 1
    before = len(calls)
    with pytest.raises(ScientificBudgetViolation, match="non-retryable"):
        p1.execute_target(document, target_index=0, simulator_factory=lambda config: fake_simulator)
    assert len(calls) == before
    assert (b40_root / "freeze.json").is_file()
    response_path = b40_root / "artifact/response.f32le"
    response_path.write_bytes(response_path.read_bytes()[:-1] + b"\x00")
    with pytest.raises(Exception, match="hash|identity|content|bytes"):
        p1._validate_dataset(document, document["ordered_dataset_declarations"][0])


def test_mock_target_success_publishes_completion_after_both_freezes(tmp_path, monkeypatch):
    document = _declaration(tmp_path)
    monkeypatch.setattr(p1, "_require_exact_clean_source", lambda document: None)
    response_spec = SimpleNamespace(
        axes=tuple(SimpleNamespace(name=name) for name in p1.RESPONSE_NAMES)
    )
    simulator = SimpleNamespace(
        capabilities=SimpleNamespace(simulator_identity=p1.SIMULATOR_ID, response_spec=response_spec)
    )
    calls = []

    def fake_checked(simulator, theta, *, device, dtype, context):
        del simulator, device, dtype
        calls.append(context)
        return SimpleNamespace(values=(torch.arange(20, dtype=torch.float32) + theta.sum()).contiguous())

    monkeypatch.setattr(p1, "checked_evaluate", fake_checked)
    result = p1.execute_target(document, target_index=0, simulator_factory=lambda config: simulator)
    assert len(calls) == 140
    assert [item["budget"] for item in result["datasets"]] == [40, 100]
    target_root = tmp_path / "workers/targets/target-000"
    assert (target_root / "b40/freeze.json").is_file()
    assert (target_root / "b100/freeze.json").is_file()
    completion = p1._load_target_completion(
        target_root,
        document=document,
        target_index=0,
        receipts=[
            p1._validate_dataset(document, document["ordered_dataset_declarations"][0]),
            p1._validate_dataset(document, document["ordered_dataset_declarations"][1]),
        ],
    )
    assert completion["target_completion_identity"] == result["target_completion_identity"]


def test_scheduler_geometry_is_canonical_policy_compatible_and_no_requeue():
    submit = (ROOT / "infrastructure/truba/submit_sbi_zoo_p1.sh").read_text()
    worker = (ROOT / "infrastructure/truba/sbi_zoo_p1_worker.sbatch").read_text()
    collector = (ROOT / "infrastructure/truba/sbi_zoo_p1_collector.sbatch").read_text()
    profile = (ROOT / "infrastructure/truba/sbi_zoo_p1_v1.env").read_text()
    lifecycle = (ROOT / "infrastructure/truba/validate_exact_source.sh").read_text()
    machine_contract = json.loads(
        (ROOT / "experiments/paper/studies/graph_paper120/analysis/sbi_zoo_extension_v1.json").read_text()
    )["implementation"]["p1_execution"]
    worker_submit, collector_submit = submit.split('collector_job="$(sbatch', maxsplit=1)
    assert "--exclusive" not in worker
    assert "--exclusive" not in worker_submit
    assert "--exclusive" not in collector
    assert "--exclusive" not in collector_submit
    assert "--exclusive" not in profile
    assert "--array=\"${MUTACO_SBI_ZOO_P1_ARRAY}%${MUTACO_SBI_ZOO_P1_CONCURRENCY}\"" in submit
    assert "--cpus-per-task=20" in worker
    assert "--mem=64G" in worker
    assert "--time=00:30:00" in worker
    assert "--cpus-per-task=20" in collector
    assert "--mem=64G" in collector
    assert "--time=00:15:00" in collector
    assert "--no-requeue" in submit and "--no-requeue" in worker and "--no-requeue" in collector
    assert 'MUTACO_SBI_ZOO_P1_MEMORY="64G"' in profile
    assert 'MUTACO_SBI_ZOO_P1_COLLECTOR_MEMORY="64G"' in profile
    assert 'MUTACO_SBI_ZOO_P1_ARRAY="0-59"' in profile
    assert 'MUTACO_SBI_ZOO_P1_CONCURRENCY=20' in profile
    assert 'MUTACO_SBI_ZOO_P1_RUNTIME_THREADS_PER_WORKER=20' in profile
    assert 'OMP_NUM_THREADS="${MUTACO_SBI_ZOO_P1_RUNTIME_THREADS_PER_WORKER}"' in worker
    assert 'MKL_NUM_THREADS="${MUTACO_SBI_ZOO_P1_RUNTIME_THREADS_PER_WORKER}"' in worker
    assert 'OPENBLAS_NUM_THREADS="${MUTACO_SBI_ZOO_P1_RUNTIME_THREADS_PER_WORKER}"' in worker
    assert "physical-task" in worker
    assert "--physical-task-index" in worker
    assert p1.SLURM_GEOMETRY["array"] == "0-59%20"
    assert p1.SLURM_GEOMETRY["array_elements"] == 60
    assert p1.SLURM_GEOMETRY["logical_targets"] == 120
    assert p1.SLURM_GEOMETRY["logical_target_mapping"] == ["task_id", "task_id+60"]
    assert p1.SLURM_GEOMETRY["runtime_threads_per_worker"] == 20
    assert p1.SLURM_GEOMETRY["slurm_cpus_per_task"] == 20
    assert p1.SLURM_GEOMETRY["requested_memory_per_node"] == "64G"
    assert p1.SLURM_GEOMETRY["logical_workers_per_allocation"] == 1
    assert p1.SLURM_GEOMETRY["node_colocation"] == "scheduler_controlled"
    assert p1.SLURM_GEOMETRY["expected_slurm_alloc_cpus_per_worker"] == 20
    assert p1.SLURM_GEOMETRY["observed_slurm_alloc_cpus_per_worker"] is None
    assert p1.SLURM_GEOMETRY["job_exclusive"] is False
    assert p1.COLLECTOR_SLURM_GEOMETRY["job_exclusive"] is False
    assert p1.COLLECTOR_SLURM_GEOMETRY["dependency"] == "afterany_worker_array"
    assert machine_contract["exclusive"] is False
    assert machine_contract["expected_slurm_alloc_cpus_per_worker"] == 20
    assert machine_contract["observed_slurm_alloc_cpus_per_worker"] is None
    assert "sbi_zoo_p1)" in lifecycle
    assert "submit_sbi_zoo_p1.sh --validate-only" in lifecycle
    lifecycle_p1 = lifecycle.split("sbi_zoo_p1)", maxsplit=1)[1].split(";;", maxsplit=1)[0]
    assert "--exclusive" not in lifecycle_p1


def test_physical_task_mapping_is_exact_and_second_target_runs_after_failure(tmp_path):
    assert [
        target
        for task_index in range(60)
        for target in p1.logical_target_positions(task_index)
    ] == [item for pair in zip(range(60), range(60, 120)) for item in pair]
    assert sorted(
        target
        for task_index in range(60)
        for target in p1.logical_target_positions(task_index)
    ) == list(range(120))
    with pytest.raises(p1.P1ExecutionError, match="outside"):
        p1.logical_target_positions(60)

    calls = []

    def runner(command, *, check):
        assert check is False
        target_index = int(command[command.index("--target-index") + 1])
        calls.append(target_index)
        return SimpleNamespace(returncode=1 if target_index == 7 else 0)

    returncode, terminal = p1.execute_physical_task(
        tmp_path / "declaration.json", physical_task_index=7, runner=runner
    )
    assert calls == [7, 67]
    assert returncode == 1
    assert terminal["logical_target_positions"] == [7, 67]
    assert terminal["logical_worker_results"] == [
        {"target_position": 7, "returncode": 1},
        {"target_position": 67, "returncode": 0},
    ]
    assert terminal["all_logical_workers_succeeded"] is False
    assert terminal["automatic_retry"] is False
    assert terminal["slurm_requeue"] is False


def test_negative_tripwire_probe_fails_closed(physical_simulator_tripwire):
    from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter

    physical_simulator_tripwire["expected"] = True
    with pytest.raises(PhysicalSimulatorTripwireTriggered):
        GraphScientificD4Adapter().build_target_simulator(
            {}, device=torch.device("cpu"), dtype=torch.float32
        )
