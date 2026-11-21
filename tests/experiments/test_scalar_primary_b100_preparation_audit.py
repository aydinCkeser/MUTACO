from __future__ import annotations

from pathlib import Path

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.scalar_primary_b100_preparation_audit_cli import OUTPUT, build


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = (
    "experiments/paper/exploration/outputs/scalar_primary_smoke/"
    "population_b100_v3r2_source_validation_receipt.json"
)


def test_b100_preparation_audit_is_complete_and_reproducible() -> None:
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["scientific_calls"] == 0
    assert result["truba_jobs_submitted"] == 0
    manifest = read_json(ROOT / OUTPUT / "manifest.json")
    assert manifest["maximum_total_calls"] == 36_360
    assert manifest["logical_workers"] == 120
    assert manifest["physical_array_tasks"] == 60
    assert manifest["cross_experiment_contract_audit"]["status"] == "valid"
    assert manifest["execution_authorized"] is False


def test_b100_preparation_artifact_hash_pairs_match() -> None:
    output = ROOT / OUTPUT
    for name in ("analysis_spec.json", "manifest.json"):
        path = output / name
        assert (output / f"{name}.sha256").read_text(encoding="ascii").strip() == file_hash(path)


def test_b100_exact_source_receipt_is_zero_call_and_non_submitting() -> None:
    receipt_path = ROOT / RECEIPT
    assert (
        file_hash(receipt_path)
        == "91213dba23c23d12c9a1e9fd8f96b29f13aa3302fd88cb9c1173fb26f3960a50"
    )
    receipt = read_json(receipt_path)
    assert receipt["status"] == "validated_zero_call"
    assert receipt["source_commit"] == "649235bcb7e8773f5f4b0425ad111314d7d3b1f9"
    assert receipt["source_materialization"]["status"] == "materialized_clean"
    assert receipt["worker_preview"]["all_shards_validated"] is True
    assert receipt["worker_preview"]["workers"] == 120
    assert receipt["worker_preview"]["call_preview"]["maximum_total_calls"] == 36_360
    assert receipt["scientific_calls"] == 0
    assert receipt["job_submitted"] is False
    assert receipt["submission_authorized"] is False
