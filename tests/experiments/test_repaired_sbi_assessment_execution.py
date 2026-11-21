from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmarking.artifacts import read_json
from experiments.paper.common.repaired_sbi_assessment_execution import (
    execute_worker,
    shard_tasks,
    worker_preview,
)
from experiments.paper.common.repaired_sbi_assessment_preparation import OUTPUT
from experiments.paper.common.validation import PaperContractError
from infrastructure.truba.collect_repaired_sbi_assessment import collect


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ID = "repaired-sbi-assessment-preparation-package-v1-15191325a32642bc9da1f6e4514cc7f84208263f62e8bc642832a18722f8eb46"
DECLARATION_ID = "repaired-sbi-assessment-declaration-v1-acd75ed8d2fcc424f5b2bd68f092c3f2f4fa61fdbec4195d355a7b033f01492f"
COMMIT = "a" * 40


def _tasks() -> list[dict]:
    return [json.loads(line) for line in (ROOT / OUTPUT / "assessment_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]


def _targets() -> dict[str, dict]:
    result = {}
    for position in range(120):
        row = read_json(ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3" / f"target_{position:03d}.json")
        result[row["target_record_id"]] = row
    return result


def _evaluate(theta, context, target):
    del theta, context
    return target["target_response"]


def test_assessment_shards_keep_three_methods_and_budgets_together() -> None:
    tasks = _tasks()
    seen = set()
    for rank in range(120):
        selected = shard_tasks(tasks, rank=rank, workers=120)
        assert [(task["budget"], task["target_position"]) for task in selected] == [
            (40, rank), (40, rank), (40, rank),
            (100, rank), (100, rank), (100, rank),
        ]
        assert [task["method"] for task in selected] == ["mutaco", "sbi_npe", "sbi_snpe"] * 2
        assert worker_preview(tasks, rank=rank, workers=120, maximum_calls=6)["scientific_calls_executed"] == 0
        seen.update(task["task_identity"] for task in selected)
    assert len(seen) == 720


def test_assessment_worker_preserves_failures_and_terminal(tmp_path: Path) -> None:
    tasks, targets = _tasks(), _targets()
    calls = 0

    def fail_once(theta, context, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic assessment failure")
        return _evaluate(theta, context, target)

    output = tmp_path / "worker-000"
    with pytest.raises(PaperContractError, match="evidence was preserved"):
        execute_worker(
            tasks=tasks, targets=targets, output=output, rank=0, workers=120,
            maximum_calls=6, source_commit=COMMIT,
            declaration_identity=DECLARATION_ID, package_identity=PACKAGE_ID,
            evaluate=fail_once,
        )
    assert len(list((output / "attempts").glob("b*/target-*/*.json"))) == 6
    terminal = read_json(output / "worker_terminal.json")
    assert terminal["status"] == "failed"
    assert terminal["physical_calls"] == {
        "maximum_calls": 6, "attempted_calls": 6, "successful_calls": 5,
        "failed_calls": 1, "retry_calls": 0,
    }


def test_assessment_full_synthetic_lifecycle_and_collection(tmp_path: Path) -> None:
    tasks, targets = _tasks(), _targets()
    workers_root = tmp_path / "workers"
    for rank in range(120):
        execute_worker(
            tasks=tasks, targets=targets, output=workers_root / f"worker-{rank:03d}",
            rank=rank, workers=120, maximum_calls=6, source_commit=COMMIT,
            declaration_identity=DECLARATION_ID, package_identity=PACKAGE_ID,
            evaluate=_evaluate,
        )
    collection = collect(
        tasks=tasks, workers_root=workers_root, output=workers_root / "collection",
        workers=120, expected_commit=COMMIT, package_identity=PACKAGE_ID,
    )
    assert collection["status"] == "completed"
    assert collection["record_count"] == 720
    assert collection["physical_calls"] == {
        "maximum_calls": 720, "attempted_calls": 720, "successful_calls": 720,
        "failed_calls": 0, "retry_calls": 0,
    }
    compact = read_json(workers_root / "collection" / "compact_export.json")
    assert len(compact["records"]) == 720
    assert all(record["response_error_mae"] == 0.0 for record in compact["records"])
