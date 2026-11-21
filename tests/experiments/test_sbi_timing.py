from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from experiments.benchmarking.timing import (
    TimingLedger,
    TimingValidationError,
    attach_paper_dispatch_timing,
    summarize_sbi_timing,
)
from experiments.paper.common import campaign as paper_campaign
from experiments.paper.common.aggregate import load_paper_records
from experiments.paper.common.campaign import ExecutionSelection, load_paper_campaign
from experiments.paper.common.timing_report import summarize_persisted_timings


ROOT = Path(__file__).resolve().parents[2]


def _event(
    event_id: str,
    task: str,
    wall: float,
    cpu: float,
    *,
    parent: str | None = None,
    round_index: int | None = None,
    status: str = "completed",
) -> dict:
    return {
        "event_id": event_id,
        "parent_event_id": parent,
        "task": task,
        "method": "sbi_snpe",
        "target_index": 0,
        "round_index": round_index,
        "device": "cpu",
        "status": status,
        "work_count": 4,
        "simulator_calls": 4 if task == "inference_simulation" else 0,
        "budget_classification": "algorithm",
        "started_utc": "2026-08-07T00:00:00+00:00",
        "ended_utc": "2026-08-07T00:00:01+00:00",
        "inclusive_seconds": wall,
        "process_cpu_seconds": cpu,
        "process_cpu_inclusive_seconds": cpu,
    }


def test_timing_summary_aggregates_rounds_and_avoids_nested_double_counting():
    events = [
        _event("root", "neural_training", 3.0, 2.5, round_index=0),
        _event(
            "child", "posterior_construction", 1.0, 0.5,
            parent="root", round_index=0,
        ),
        _event("round-1", "neural_training", 2.0, 1.5, round_index=1),
        _event("sample", "posterior_sampling", 1.0, 0.8),
        _event("recommend", "posterior_diagnostics_recommendation", 0.5, 0.4),
        _event("assessment", "held_out_assessment", 0.5, 0.3),
        _event("simulation", "inference_simulation", 1.0, 0.8, round_index=0),
    ]
    result = summarize_sbi_timing(
        events,
        method_total_wall_seconds=8.5,
        method_total_process_cpu_seconds=7.0,
        status="completed",
        failure_phase=None,
    )
    assert result["schema_version"] == "mutaco-paper-sbi-timing-v1"
    assert result["phases"]["neural_training"]["wall_seconds"] == 4.0
    assert result["phases"]["posterior_construction"]["wall_seconds"] == 1.0
    assert result["reconciliation"]["measured_phase_wall_seconds"] == 8.0
    assert result["within_method_unattributed_overhead"]["wall_seconds"] == 0.5
    assert {(row["phase"], row["round_index"]) for row in result["rounds"]} >= {
        ("neural_training", 0), ("neural_training", 1),
    }


@pytest.mark.parametrize(
    "events,total,match",
    [
        ([_event("x", "unknown", 1.0, 1.0)], 2.0, "unknown timing task"),
        ([_event("x", "neural_training", 3.0, 1.0)], 2.0, "exceed"),
        (
            [_event("x", "neural_training", 1.0, 1.0, parent="missing")],
            2.0,
            "missing parent",
        ),
    ],
)
def test_materially_invalid_timing_is_rejected(events, total, match):
    with pytest.raises(TimingValidationError, match=match):
        summarize_sbi_timing(
            events,
            method_total_wall_seconds=total,
            method_total_process_cpu_seconds=total,
            status="completed",
            failure_phase=None,
        )


def test_failed_event_keeps_exception_and_partial_timing():
    ledger = TimingLedger()
    with pytest.raises(RuntimeError, match="injected"):
        with ledger.event(
            "posterior_construction", method="sbi_npe", target_index=0,
            device="cpu",
        ):
            raise RuntimeError("injected posterior construction failure")
    assert ledger.events[0]["status"] == "failed"
    assert ledger.events[0]["failure_type"] == "RuntimeError"
    result = summarize_sbi_timing(
        ledger.events,
        method_total_wall_seconds=ledger.events[0]["inclusive_seconds"] + 0.01,
        method_total_process_cpu_seconds=(
            ledger.events[0]["process_cpu_inclusive_seconds"] + 0.01
        ),
        status="failed",
        failure_phase="round_0_posterior_build",
    )
    assert result["completeness"] == "exact_partial_failure"
    assert result["failure_phase"] == "round_0_posterior_build"


def test_dispatch_timing_reconciles_and_rejects_impossible_total():
    summary = summarize_sbi_timing(
        [_event("x", "neural_training", 1.0, 0.5)],
        method_total_wall_seconds=2.0,
        method_total_process_cpu_seconds=1.0,
        status="completed",
        failure_phase=None,
    )
    result = attach_paper_dispatch_timing(
        summary,
        paper_dispatch_wall_seconds=2.5,
        paper_dispatch_process_cpu_seconds=1.2,
    )
    assert result["paper_dispatch_overhead"]["wall_seconds"] == 0.5
    with pytest.raises(TimingValidationError, match="exceeds"):
        attach_paper_dispatch_timing(
            summary,
            paper_dispatch_wall_seconds=1.0,
            paper_dispatch_process_cpu_seconds=0.5,
        )


def test_persisted_report_supports_detailed_and_historical_total_only_records():
    detailed = summarize_sbi_timing(
        [_event("x", "neural_training", 1.0, 0.5)],
        method_total_wall_seconds=2.0,
        method_total_process_cpu_seconds=1.0,
        status="completed",
        failure_phase=None,
    )
    detailed = attach_paper_dispatch_timing(
        detailed,
        paper_dispatch_wall_seconds=2.1,
        paper_dispatch_process_cpu_seconds=1.1,
    )
    report = summarize_persisted_timings(
        [
            {"method": "sbi_snpe", "timings": {"total": 2.0, "sbi": detailed}},
            {"method": "sbi_npe", "timings": {"total": 55.0}},
        ]
    )
    assert report["methods"]["sbi_snpe"]["phase_decomposition_available"]
    assert report["methods"]["sbi_npe"]["method_total_wall_seconds"] == 55.0
    assert report["methods"]["sbi_npe"]["timing_completeness"] == {
        "historical_total_only": 1
    }
    assert not report["methods"]["sbi_npe"]["phase_decomposition_available"]


def _real_smoke_declaration():
    declaration = load_paper_campaign(
        ROOT / "experiments/paper/analytic_handwritten/configs/campaign.json"
    )
    config = json.loads(json.dumps(declaration.benchmark_configuration))
    config["campaign"]["target_count"] = 1
    config["comparison_set"] = {"policy": "full_bank"}
    config["simulation_budget_per_target"] = 8
    config["randomness"]["sbi_batch_context_policy"] = "single_batch_context_v1"
    config["sbi"].update(
        {
            "training_batch_size": 4,
            "stop_after_epochs": 1,
            "max_num_epochs": 2,
            "posterior_samples": 64,
            "show_progress_bars": False,
        }
    )
    return replace(declaration, benchmark_configuration=config)


def test_real_cpu_npe_and_two_round_snpe_paper_smoke_and_exact_replay(
    tmp_path, monkeypatch
):
    pytest.importorskip("sbi")
    from experiments.benchmarking.methods import sbi_suite

    original_allocation = sbi_suite.sequential_allocation

    def two_round_allocation(total: int, rounds: int) -> tuple[int, int]:
        assert rounds == 3
        return total // 2, total - total // 2

    monkeypatch.setattr(sbi_suite, "sequential_allocation", two_round_allocation)
    declaration = _real_smoke_declaration()
    selection = ExecutionSelection(
        ROOT / "experiments/paper/analytic_handwritten/configs/primary_execution.json",
        declaration.document["campaign_id"],
        ("sbi_npe", "sbi_snpe"),
    )
    projections = []
    try:
        for replay in range(2):
            output = tmp_path / f"replay-{replay}"
            monkeypatch.setattr(
                paper_campaign, "_output_path", lambda ignored, path=output: path
            )
            execution = paper_campaign.execute_paper_campaign(
                declaration, selection, progress_callback=lambda event: None
            )
            assert execution.executed_records == 2
            records = load_paper_records(output)
            report = summarize_persisted_timings(records)
            assert set(report["methods"]) == {"sbi_npe", "sbi_snpe"}
            replay_projection = {}
            for row in records:
                assert row["status"] == "completed", row.get("failure_explanation")
                timing = row["timings"]["sbi"]
                assert timing["completeness"] == "exact_complete"
                assert set(timing["phases"]) == {
                    "inference_simulation",
                    "neural_training",
                    "posterior_construction",
                    "posterior_sampling",
                    "posterior_diagnostics_recommendation",
                    "held_out_assessment",
                }
                expected_rounds = 2 if row["method"] == "sbi_snpe" else 1
                assert len(
                    {
                        item["round_index"]
                        for item in timing["rounds"]
                        if item["phase"] == "neural_training"
                    }
                ) == expected_rounds
                assert timing["reconciliation"]["wall_residual_seconds"] >= 0.0
                replay_projection[row["method"]] = {
                    "recommended_theta": row["recommended_theta"],
                    "assessment_response": row["assessment_response"],
                    "target_error_l2": row["target_error_l2"],
                    "design_error_l2": row["design_error_l2"],
                    "seeds": row["seeds"],
                    "accounting": row["accounting"],
                }
            projections.append(replay_projection)
    finally:
        monkeypatch.setattr(sbi_suite, "sequential_allocation", original_allocation)
    assert projections[0] == projections[1]
