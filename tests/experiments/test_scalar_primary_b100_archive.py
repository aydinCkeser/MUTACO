from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmarking.artifacts import file_hash, read_json
from reproducibility.hash_pair_inventory import build_inventory
from experiments.paper.common.scalar_primary_b100_archive_cli import OUTPUT, RAW, build


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_IDENTITY = (
    "scalar-primary-population-evidence-v1-"
    "c72415fcb29e8d7379604d3cf0a33a3b0e962fed94ead446088c55ece83a9375"
)


def test_b100_compact_evidence_is_complete_and_hash_paired() -> None:
    output = ROOT / OUTPUT
    for path in sorted(output.glob("*.json*")):
        if path.suffix == ".sha256":
            continue
        sidecar = output / f"{path.name}.sha256"
        assert sidecar.read_text(encoding="ascii").strip() == file_hash(path)
    manifest_path = output / "manifest.json"
    assert file_hash(manifest_path) == "7f1c79a1351457cd8f0ee918bcfc0077375635e334803f54beee6fd1428436e8"
    manifest = read_json(manifest_path)
    assert manifest["evidence_identity"] == EVIDENCE_IDENTITY
    assert manifest["call_accounting"] == {
        "attempted": 36_360,
        "calls_added_by_archival_and_analysis": 0,
        "failed": 0,
        "maximum_authorized": 36_360,
        "successful": 36_360,
    }
    assert manifest["coverage"]["completed_targets"] == 120
    assert manifest["coverage"]["method_records"] == 360
    assert manifest["coverage"]["trajectory_results"] == 36_000
    assert manifest["integrity"]["remote_worker_hash_inventory"]["hash_mismatches"] == []


def test_b100_primary_results_and_diagnostics_are_frozen() -> None:
    manifest = read_json(ROOT / OUTPUT / "manifest.json")
    primary = manifest["primary_endpoint_heldout_physical_response_component_mean_mae"]
    assert primary["mutaco"]["mean"] == pytest.approx(0.019611348367865662)
    assert primary["scalar_qlognei"]["mean"] == pytest.approx(0.025521009362459088)
    assert primary["sobol"]["mean"] == pytest.approx(0.05905393938087813)
    assert manifest["target_winners"] == {"mutaco": 80, "scalar_qlognei": 33, "sobol": 7}
    pair = manifest["paired_primary_endpoint"]["mutaco__scalar_qlognei"]
    assert pair["left_wins"] == 84
    assert pair["left_losses"] == 36
    assert pair["paired_mean_bootstrap_95_percentile_interval"][1] < 0
    diagnostics = manifest["scalar_qlognei_diagnostics"]
    assert diagnostics["candidate_sources"] == {
        "botorch_optimize_acqf": 10_078,
        "deterministic_raw_fallback_after_ScalarQLogNEIError": 2,
    }
    assert diagnostics["precision_fallbacks"] == 0
    assert diagnostics["unrecovered_algorithm_failures"] == 0
    assert diagnostics["maximum_direct_numerical_jitter"] == 1e-4


def test_b100_archive_reproduces_when_verified_fetch_is_available() -> None:
    if not (ROOT / RAW / "population/collection/compact_export.json").is_file():
        pytest.skip("verified untracked B100 compact fetch is not present")
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["evidence_identity"] == EVIDENCE_IDENTITY
    assert result["scientific_calls"] == 0
    assert result["truba_jobs_submitted"] == 0


def test_hash_pair_inventory_is_read_only_and_deterministic(tmp_path: Path) -> None:
    for name in ("worker_start", "worker_terminal"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"status":"completed"}\n', encoding="utf-8")
        path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")
    first = build_inventory(tmp_path)
    second = build_inventory(tmp_path)
    assert first == second
    assert first["json_files"] == 2
    assert first["sha256_sidecars"] == 2
    assert first["hash_mismatches"] == []
    assert first["missing_sidecars"] == []
    assert first["orphan_sidecars"] == []
    assert first["scientific_calls"] == 0
