from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.graph_inversion import sbi_zoo_p2 as p2


ROOT = Path(__file__).resolve().parents[2]


def test_component_enumeration_uses_frozen_p1_randomness():
    targets = [
        {"target_position": i, "target_record_id": f"target-{i:03d}"}
        for i in range(120)
    ]
    declaration = {
        "p1_declaration_identity": "p1-declaration",
        "randomness": {"root_seed": 12345},
        "cohort": {"targets": targets},
    }
    receipts = [
        {
            "target_position": i,
            "target_record_id": f"target-{i:03d}",
            "budget": budget,
            "dataset_identity": f"dataset-{i}-{budget}",
            "canonical_content_sha256": f"hash-{i}-{budget}",
            "freeze_identity": f"freeze-{i}-{budget}",
        }
        for i in range(120)
        for budget in (40, 100)
    ]
    first = p2._component_specs(declaration, {"dataset_receipts": receipts})
    second = p2._component_specs(declaration, {"dataset_receipts": receipts})
    assert first == second
    assert len(first) == 480
    assert len({item["component_identity"] for item in first}) == 480
    assert len({(item["target_position"], item["budget"], item["method"]) for item in first}) == 480
    assert all(item["training_seed"] != item["recommendation_seed"] for item in first)


def test_task_mapping_and_independent_component_progression():
    calls = []

    def runner(command, *, check):
        assert check is False
        target = int(command[command.index("--target-position") + 1])
        budget = int(command[command.index("--budget") + 1])
        method = command[command.index("--method") + 1]
        calls.append((target, budget, method))
        return SimpleNamespace(returncode=1 if (target, budget, method) == (7, 40, p2.METHODS[0]) else 0)

    code, result = p2.execute_physical_task(
        Path("declaration.json"), physical_task_index=7, runner=runner
    )
    assert code == 1
    assert len(calls) == 8
    assert calls[:4] == [
        (7, 40, p2.METHODS[0]),
        (7, 40, p2.METHODS[1]),
        (7, 100, p2.METHODS[0]),
        (7, 100, p2.METHODS[1]),
    ]
    assert calls[4:] == [
        (67, 40, p2.METHODS[0]),
        (67, 40, p2.METHODS[1]),
        (67, 100, p2.METHODS[0]),
        (67, 100, p2.METHODS[1]),
    ]
    assert result["physical_simulator_calls"] == 0
    assert result["heldout_accesses"] == 0


def test_scheduler_surface_is_nonexclusive_explicit_and_zero_call():
    profile = (ROOT / "infrastructure/truba/sbi_zoo_p2_v1.env").read_text()
    submit = (ROOT / "infrastructure/truba/submit_sbi_zoo_p2.sh").read_text()
    worker = (ROOT / "infrastructure/truba/sbi_zoo_p2_worker.sbatch").read_text()
    collector = (ROOT / "infrastructure/truba/sbi_zoo_p2_collector.sbatch").read_text()
    lifecycle = (ROOT / "infrastructure/truba/validate_exact_source.sh").read_text()
    assert "--exclusive" not in profile + submit + worker + collector
    assert 'MUTACO_SBI_ZOO_P2_ARRAY="0-59"' in profile
    assert "MUTACO_SBI_ZOO_P2_CONCURRENCY=10" in profile
    assert "MUTACO_SBI_ZOO_P2_RUNTIME_THREADS_PER_WORKER=20" in profile
    assert "--cpus-per-task=20" in worker
    assert "--mem=64G" in worker
    assert "--time=01:00:00" in worker
    assert "--no-requeue" in worker and "--no-requeue" in collector and "--no-requeue" in submit
    assert "OMP_NUM_THREADS" in worker and "MKL_NUM_THREADS" in worker and "OPENBLAS_NUM_THREADS" in worker
    assert "sbi_zoo_p2)" in lifecycle
    assert p2.SLURM_GEOMETRY["array"] == "0-59%10"
    assert p2.SLURM_GEOMETRY["job_exclusive"] is False
    assert p2.COLLECTOR_SLURM_GEOMETRY["job_exclusive"] is False


def test_collection_requires_exactly_480(monkeypatch, tmp_path):
    specs = [
        {
            "target_position": i,
            "target_record_id": f"target-{i:03d}",
            "budget": budget,
            "method": method,
            "component_identity": f"component-{i}-{budget}-{method}",
            "dataset_identity": f"dataset-{i}-{budget}",
        }
        for i in range(120)
        for budget in (40, 100)
        for method in p2.METHODS
    ]
    document = {
        "p2_declaration_identity": "p2",
        "p2_execution_source": "source",
        "p1": {"collection_identity": "p1"},
        "output_root": str(tmp_path),
        "components": specs,
    }
    monkeypatch.setattr(p2, "validate_declaration", lambda value: value)
    monkeypatch.setattr(p2, "_terminal_status", lambda root: ("success", {"recommendation_identity": f"rec-{root.name}"}))

    def load(document, spec):
        return {"recommendation_identity": f"rec-{spec['method']}-b{spec['budget']}"}

    monkeypatch.setattr(p2, "_load_recommendation", load)
    result = p2.collect_population(document)
    assert result["status"] == "PASS"
    assert result["recommendation_count"] == 480
    assert result["counts"] == {
        "NLE_B40": 120,
        "NLE_B100": 120,
        "NRE_B40": 120,
        "NRE_B100": 120,
    }
    assert result["physical_simulator_calls"] == 0
    assert result["heldout_accesses"] == 0


def test_no_simulator_or_heldout_execution_surface():
    source = (ROOT / "experiments/graph_inversion/sbi_zoo_p2.py").read_text()
    assert "checked_evaluate" not in source
    assert "GraphScientificD4Adapter" not in source
    assert "build_target_simulator" not in source
    assert "92021540" in source and "92021600" in source
    assert "physical_simulator_calls\": 0" in source
