"""Saved results retain their declared paths, package membership and scientific roles."""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from reproducibility.recorded_paths import resolve_recorded_path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "results/paper"


def test_failed_precursor_and_accepted_v2_are_separate_complete_packages() -> None:
    v1 = PAPER / "terminal_domain_heldout_assessment_b40_v1"
    v2 = PAPER / "terminal_domain_heldout_assessment_b40_v2"
    for package in (v1, v2):
        assert package.is_dir() and not package.is_symlink()
        assert resolve_recorded_path(ROOT, f"benchmarks/paper/{package.name}") == package
        record = json.loads((package / "manifest.json").read_text())
        assert hashlib.sha256((package / "analysis_report.json").read_bytes()).hexdigest() == record["analysis_report_sha256"]
    manifest = json.loads((v2 / "manifest.json").read_text())
    assert manifest["analysis_report_sha256"] == (
        "19e126c932c134934713273dbff0495f646c6dccafa8595da373283348930f66"
    )
    assert hashlib.sha256((v2 / "analysis_report.json").read_bytes()).hexdigest() == manifest["analysis_report_sha256"]
    assert sum(p.is_file() for p in v2.rglob("*")) == 978
    failed = "graph_primary_nei_q1_qualification_attempt_6244497"
    archive = ROOT / "scientific_archive/superseded_results" / failed
    registry = tomllib.loads((ROOT / "results/experiment_registry.toml").read_text())
    record = next(item for item in registry["experiments"] if item["id"] == failed)
    assert ROOT / record["evidence"] == archive / "manifest.json"
    assert not (ROOT / "benchmarks/paper" / failed).exists()
    assert json.loads((archive / "manifest.json").read_text())["status"] == "failed_preserved_no_automatic_retry"


def test_compact_input_resolved_names_and_frozen_producer_depth() -> None:
    # The original inventory binds the old addresses and the exact bytes independently.
    salvage = json.loads((PAPER / "repaired_mutaco_sbi_salvage_design/v1/inventory.json").read_bytes())
    for budget, canonical, declared in (
        ("b40", "graph_paper120/paper120_mutaco_npe_snpe_v1", "benchmarks/graph/paper120_mutaco_npe_snpe_v1"),
        ("b100", "graph_paper120/b100_v1", "benchmarks/paper/graph_paper120/b100_v1"),
    ):
        source = salvage[budget]["compact_source"]
        assert source["path"] == f"{declared}/paired_records.jsonl"
        located = resolve_recorded_path(ROOT, source["path"])
        assert located.resolve() == PAPER / canonical / "paired_records.jsonl"
        assert hashlib.sha256(located.read_bytes()).hexdigest() == source["sha256"]
    package = PAPER / "oscillator_physics_multi_target_sensitivity/paper120_maximin12_job6353452"
    producer = (package / "build_cohort_figure.py").resolve()
    assert producer.parents[5] == ROOT
    inventory = json.loads((package / "provenance/file_hashes.json").read_text())
    entry = next(x for x in inventory["files"] if x["path"] == producer.name)
    assert hashlib.sha256(producer.read_bytes()).hexdigest() == entry["sha256"]


def test_exploration_workspace_does_not_redirect_new_outputs_into_results() -> None:
    # Native declarations put new fetches and validation receipts in the same workspace.
    workspace = ROOT / "experiments/paper/exploration/outputs/scalar_primary_smoke"
    assert not workspace.is_symlink()
    receipt = "population_b40_v3r2_source_validation_receipt.json"
    assert (workspace / receipt).samefile(PAPER / "scalar_primary_source_validation_receipts" / receipt)
    registry = tomllib.loads((ROOT / "results/experiment_registry.toml").read_text())
    records = [entry for entry in registry["experiments"] if "population_b40_validation_receipt" in entry]
    assert len(records) == 1
    record = records[0]
    assert (ROOT / record["population_b40_validation_receipt"]).samefile(workspace / receipt)
    assert hashlib.sha256((workspace / receipt).read_bytes()).hexdigest() == (
        record["population_b40_validation_receipt_sha256"]
    )


def test_original_qualification_commands_and_checksum_pins_resolve():
    current = ROOT / "results/reproducibility"
    assert not (ROOT / "benchmarks").exists()
    for package in ("shareable_milestone_v1", "b100_regression_v1"):
        assert resolve_recorded_path(ROOT, f"benchmarks/reproducibility/{package}").samefile(current / package)

    source = json.loads((current / "documented_core_qualification_v1/manifest.json").read_text())
    closure = json.loads((current / "physical_export_closure_v1/manifest.json").read_text())
    pins = [
        (source[key]["path"], source[key]["sha256"])
        for key in ("documentation_revision", "historical_source_hashes", "test_identity_reference")
    ]
    pins.append((closure["approved_plan"]["source"], closure["approved_plan"]["sha256"]))
    for recorded_path, digest in pins:
        assert hashlib.sha256(resolve_recorded_path(ROOT, recorded_path).read_bytes()).hexdigest() == digest
