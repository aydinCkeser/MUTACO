from __future__ import annotations

from pathlib import Path

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.repaired_sbi_recovery import OUTPUT, build


ROOT = Path(__file__).resolve().parents[2]


def test_b100_recovery_compact_evidence_is_complete() -> None:
    recovery = read_json(ROOT / OUTPUT / "b100_raw_recovery.json")
    assert recovery["coverage"] == {
        "targets": 120,
        "qualification3": 3,
        "remaining117": 117,
        "missing": 0,
        "raw_bytes": 582037498,
        "sha256_mismatches": 0,
    }
    assert [entry["target_position"] for entry in recovery["entries"]] == list(range(120))
    assert len({entry["raw_sha256"] for entry in recovery["entries"]}) == 120


def test_exact_state_qualification_passes_full_population() -> None:
    qualification = read_json(ROOT / OUTPUT / "exact_state_qualification.json")
    assert qualification["positions"] == list(range(120))
    assert qualification["qualification_positions"] == [0, 49, 99]
    assert qualification["budgets"] == [40, 100]
    assert qualification["cases_passed"] == 240
    assert qualification["cases_failed"] == 0
    assert all(case["required_equalities_passed"] for case in qualification["cases"])
    b100 = [case for case in qualification["cases"] if case["budget"] == 100]
    assert {
        case["source_commit"] for case in b100 if case["target_position"] in {0, 49, 99}
    } == {"2b067c33720ed7c6b30ee91ae6087081d5ad0776"}
    assert {
        case["source_commit"] for case in b100 if case["target_position"] not in {0, 49, 99}
    } == {"e7baa235fd7282ea478acc83f9457cfd95b76185"}
    assert qualification["optimizer_rerun_is_not_an_identity_test"]
    assert not qualification["terminal_recommendation_executed"]


def test_recovery_package_sidecars_and_zero_call_boundary() -> None:
    output = ROOT / OUTPUT
    for name in ("b100_raw_recovery.json", "exact_state_qualification.json", "manifest.json"):
        path = output / name
        assert (output / f"{name}.sha256").read_text(encoding="ascii") == file_hash(path) + "\n"
    manifest = read_json(output / "manifest.json")
    assert manifest["b100_raw_trajectories_verified"] == 120
    assert manifest["exact_state_cases_passed"] == 240
    assert manifest["population_exact_state_replay_complete"]
    assert manifest["search_reruns_required"] == 0
    assert not manifest["terminal_generation_authorized"]
    assert manifest["scientific_calls"] == 0
    assert manifest["truba_jobs_submitted"] == 0


def test_recovery_package_reproduces_from_local_archive() -> None:
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["scientific_calls"] == 0
