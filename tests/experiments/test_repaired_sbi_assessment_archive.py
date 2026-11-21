from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.repaired_sbi_assessment_archive_cli import RAW, OUTPUT, build


ROOT = Path(__file__).resolve().parents[2]


def test_tracked_repaired_sbi_assessment_evidence_is_complete() -> None:
    output = ROOT / OUTPUT
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "complete_collected_archived_and_analyzed"
    assert manifest["call_accounting"] == {
        "maximum_calls": 720, "attempted_calls": 720, "successful_calls": 720,
        "failed_calls": 0, "retry_calls": 0, "analysis_calls": 0,
    }
    assert manifest["integrity"]["json_hash_pairs_verified"] == 962
    assert manifest["reports"]["40"]["row_count"] == 360
    assert manifest["reports"]["100"]["row_count"] == 360
    for path in output.glob("*.json*"):
        if path.suffix == ".sha256":
            continue
        sidecar = output / f"{path.name}.sha256"
        assert sidecar.read_text(encoding="ascii").strip() == file_hash(path)


def test_repaired_sbi_primary_contrasts_are_paired_and_directional() -> None:
    manifest = read_json(ROOT / OUTPUT / "manifest.json")
    for budget, npe_wins, snpe_wins in (("40", 113, 110), ("100", 102, 95)):
        report = manifest["reports"][budget]
        assert report["paired_primary_endpoint"]["mutaco__sbi_npe"]["left_wins"] == npe_wins
        assert report["paired_primary_endpoint"]["mutaco__sbi_snpe"]["left_wins"] == snpe_wins
        assert report["paired_primary_endpoint"]["mutaco__sbi_npe"]["paired_mean_bootstrap_95_percentile_interval"][1] < 0
        assert report["paired_primary_endpoint"]["mutaco__sbi_snpe"]["paired_mean_bootstrap_95_percentile_interval"][1] < 0


def test_repaired_sbi_archive_reproduces_when_raw_archive_is_present() -> None:
    if not (ROOT / RAW).is_dir():
        pytest.skip("raw repaired SBI assessment archive is not present")
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["scientific_calls"] == 0
    assert result["truba_jobs_submitted"] == 0
