from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.repaired_sbi_assessment_preparation import (
    ASSESSMENT_SEEDS,
    DECLARATION_PATH,
    METHODS,
    OUTPUT,
    assessment_declaration,
    build,
)


ROOT = Path(__file__).resolve().parents[2]


def _tasks() -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / OUTPUT / "assessment_tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def test_assessment_preparation_has_exact_complete_matrix() -> None:
    tasks = _tasks()
    assert len(tasks) == 720
    assert len({task["task_identity"] for task in tasks}) == 720
    assert [
        (task["budget"], task["target_position"], task["method"])
        for task in tasks
    ] == [
        (budget, position, method)
        for budget in (40, 100)
        for position in range(120)
        for method in METHODS
    ]
    assert all(task["nominal_assessment_calls"] == 1 for task in tasks)
    assert all(not task["automatic_retry"] for task in tasks)


def test_assessment_randomness_is_paired_by_target_budget_only() -> None:
    tasks = _tasks()
    contexts: dict[tuple[int, int], list[dict]] = {}
    for task in tasks:
        contexts.setdefault((task["budget"], task["target_position"]), []).append(
            task["assessment_context"]
        )
    assert len(contexts) == 240
    unique_contexts = set()
    for (budget, position), group in contexts.items():
        assert len(group) == 3
        assert group[0] == group[1] == group[2]
        context = group[0]
        assert context["budget"] == budget
        assert context["logical_index"] == position
        assert context["purpose"] == "paired_heldout_assessment"
        assert context["family_definition"]["assessment_campaign_seed"] == ASSESSMENT_SEEDS[budget]
        assert context["coordinate"]["coordinates"] == [
            budget,
            context["target_record_identity"],
            0,
        ]
        assert "method" not in context
        assert "method" not in context["family_definition"]
        unique_contexts.add(context["family_identity"])
    assert len(unique_contexts) == 240
    assert not (
        {contexts[(40, position)][0]["family_identity"] for position in range(120)}
        & {contexts[(100, position)][0]["family_identity"] for position in range(120)}
    )


def test_assessment_declaration_and_package_forbid_execution() -> None:
    declaration = assessment_declaration()
    assert read_json(ROOT / DECLARATION_PATH) == declaration
    assert declaration["call_preview"]["maximum_assessment_calls"] == 720
    assert declaration["call_preview"]["scientific_calls_executed"] == 0
    assert not any(declaration["authorization"].values())
    manifest = read_json(ROOT / OUTPUT / "manifest.json")
    assert manifest["task_count"] == 720
    assert manifest["scientific_calls"] == 0
    assert not manifest["assessment_execution_authorized"]
    assert not manifest["truba_preparation_authorized"]
    assert not manifest["truba_submission_authorized"]


def test_assessment_preparation_sidecars_and_reproduction() -> None:
    output = ROOT / OUTPUT
    for name in (
        "declaration.json",
        "assessment_tasks.jsonl",
        "preparation.json",
        "manifest.json",
    ):
        path = output / name
        assert (output / f"{name}.sha256").read_text(encoding="ascii") == file_hash(path) + "\n"
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["task_count"] == 720
    assert result["scientific_calls"] == 0
    assert result["truba_jobs_submitted"] == 0
