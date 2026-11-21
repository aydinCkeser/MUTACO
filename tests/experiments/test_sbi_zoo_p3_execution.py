from __future__ import annotations

from pathlib import Path

import pytest

from experiments.graph_inversion import sbi_zoo_p3 as p3


ROOT = Path(__file__).resolve().parents[2]


def test_repaired_family_contexts_are_exact_and_method_paired():
    target = "frozen-target-record-v2-test"
    for budget, seed in ((40, 92021540), (100, 92021600)):
        first = p3.assessment_context(budget=budget, target_position=3, target_id=target)
        second = p3.assessment_context(budget=budget, target_position=3, target_id=target)
        assert first == second
        assert first["family_definition"]["assessment_campaign_seed"] == seed
        assert first["coordinate"]["namespace"] == "paired_heldout_assessment"


def test_mapping_is_exactly_60_by_2():
    assert sorted(target for task in range(60) for target in p3.logical_target_positions(task)) == list(range(120))
    assert p3.logical_target_positions(0) == (0, 60)
    assert p3.logical_target_positions(59) == (59, 119)
    with pytest.raises(p3.P3ExecutionError):
        p3.logical_target_positions(60)


def test_scheduler_surface_is_simple_nonexclusive_and_no_retry():
    profile = (ROOT / "infrastructure/truba/sbi_zoo_p3_v1.env").read_text()
    submit = (ROOT / "infrastructure/truba/submit_sbi_zoo_p3.sh").read_text()
    worker = (ROOT / "infrastructure/truba/sbi_zoo_p3_worker.sbatch").read_text()
    collector = (ROOT / "infrastructure/truba/sbi_zoo_p3_collector.sbatch").read_text()
    assert "--exclusive" not in profile + submit + worker + collector
    assert 'MUTACO_SBI_ZOO_P3_ARRAY="0-59"' in profile
    assert "MUTACO_SBI_ZOO_P3_CONCURRENCY=20" in profile
    assert "--cpus-per-task=20" in worker and "--mem=64G" in worker
    assert "--time=00:30:00" in worker
    assert "--no-requeue" in submit and "--no-requeue" in worker and "--no-requeue" in collector
    assert "OMP_NUM_THREADS" in worker and "MKL_NUM_THREADS" in worker and "OPENBLAS_NUM_THREADS" in worker
    assert p3.SLURM_GEOMETRY["calls_per_array_task"] == 8
    assert p3.SLURM_GEOMETRY["array"] == "0-59%20"


def test_collection_requires_exactly_480_successes(monkeypatch, tmp_path):
    tasks = [
        {
            "target_position": position,
            "target_record_id": f"target-{position}",
            "budget": budget,
            "method": method,
            "task_identity": f"task-{position}-{budget}-{method}",
            "recommendation_identity": f"rec-{position}-{budget}-{method}",
            "assessment_context": {"family_identity": f"family-{position}-{budget}"},
        }
        for position in range(120)
        for budget in p3.BUDGETS
        for method in p3.METHODS
    ]
    document = {
        "p3_declaration_identity": "p3",
        "p3_execution_source": "source",
        "p2": {"collection_identity": "p2"},
        "output_root": str(tmp_path),
        "tasks": tasks,
    }
    monkeypatch.setattr(p3, "validate_declaration", lambda value: value)
    class Ledger:
        def __init__(self, *args, **kwargs): pass
        def slot_status(self, slot): return "success"
        def terminal_payload(self, slot):
            return {"result_identity": f"result-{slot}", "result_sha256": "a" * 64}
        def accounting(self):
            return {"maximum_calls":480,"attempted_calls":480,"successful_calls":480,"failed_calls":0,"attempted_outcome_unknown":0,"retry_calls":0}
    monkeypatch.setattr(p3, "WriteAheadScientificCallLedger", Ledger)
    def load(document, task):
        slot = f"target-{task['target_position']:03d}-{task['method']}-b{task['budget']}"
        return {
            "result_identity": f"result-{slot}", "result_sha256": "a" * 64,
            "task_identity": task["task_identity"], "target_position": task["target_position"],
            "target_record_id": task["target_record_id"], "budget": task["budget"],
            "method": task["method"], "recommendation_identity": task["recommendation_identity"],
            "recommended_theta": [1,1,1,0.5], "assessment_context": task["assessment_context"],
            "assessment_response": [0.0]*20, "assessment_randomness_evidence": {},
            "response_error_l1_sum": 0.0, "response_error_mae": 0.0,
            "response_error_l2": 0.0, "parameter_error_l2": 0.0, "timing": {},
        }
    monkeypatch.setattr(p3, "_load_result", load)
    result = p3.collect_population(document)
    assert result["status"] == "PASS"
    assert result["record_count"] == 480
    assert result["counts"] == {"NLE_B40":120,"NLE_B100":120,"NRE_B40":120,"NRE_B100":120}


def test_exact_global_cap_and_no_retry(tmp_path):
    ledger = p3.WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 480)
    assert ledger.accounting() == {"maximum_calls":480,"attempted_calls":0,"successful_calls":0,"failed_calls":0,"attempted_outcome_unknown":0,"retry_calls":0}


def test_no_p2_training_or_recommendation_path():
    source = (ROOT / "experiments/graph_inversion/sbi_zoo_p3.py").read_text()
    assert "train_single_round" not in source
    assert "recommend_sample_map" not in source
    assert "build_mcmc_posterior" not in source
