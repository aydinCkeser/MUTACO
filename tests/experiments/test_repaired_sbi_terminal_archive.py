from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.repaired_sbi_terminal_archive import OUTPUT, build


ROOT = Path(__file__).resolve().parents[2]


def _records() -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / OUTPUT / "recommendations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def test_corrected_terminal_archive_is_complete_and_unique() -> None:
    records = _records()
    assert [(row["budget"], row["target_position"]) for row in records] == [
        (budget, position) for budget in (40, 100) for position in range(120)
    ]
    assert len({row["record_identity"] for row in records}) == 240
    assert len({row["recommendation_identity"] for row in records}) == 240
    assert all(len(row["selected_theta"]) == 4 for row in records)
    assert all(len(row["predicted_physical_response"]) == 20 for row in records)


def test_corrected_terminal_archive_preserves_source_and_call_accounting() -> None:
    summary = read_json(ROOT / OUTPUT / "summary.json")
    assert summary["source_counts"] == {
        "2b067c33720ed7c6b30ee91ae6087081d5ad0776": 3,
        "9235d2fefefa6af0d35cc3af2dfaa01a2d56a3a9": 120,
        "e7baa235fd7282ea478acc83f9457cfd95b76185": 117,
    }
    assert summary["return_source_counts"] == {"refined": 240}
    assert summary["terminal_optimizations"] == 240
    assert summary["failed_recommendations"] == 0
    assert summary["automatic_retries"] == 0
    assert summary["simulator_calls"] == 0
    assert summary["gp_fit_attempts"] == 0
    assert summary["search_calls"] == 0
    assert summary["heldout_assessment_calls"] == 0


def test_corrected_terminal_archive_sidecars_and_authorization_boundary() -> None:
    output = ROOT / OUTPUT
    for name in ("recommendations.jsonl", "summary.json", "manifest.json"):
        path = output / name
        assert (output / f"{name}.sha256").read_text(encoding="ascii") == file_hash(path) + "\n"
    manifest = read_json(output / "manifest.json")
    assert manifest["record_count"] == 240
    assert manifest["terminal_recommendations_frozen"] == 240
    assert manifest["simulator_calls"] == 0
    assert manifest["heldout_assessment_calls"] == 0
    assert not manifest["assessment_authorized"]
    assert manifest["truba_jobs_submitted"] == 0


def test_corrected_terminal_archive_reproduces_from_append_only_attempts() -> None:
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["terminal_recommendations_frozen"] == 240
    assert result["simulator_calls"] == 0
    assert result["heldout_assessment_calls"] == 0
