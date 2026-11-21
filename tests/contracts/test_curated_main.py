from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

from experiments.graph_inversion.paper_policies import (
    CANONICAL_PROFILE,
    PAPER120_PROFILE,
    load_graph_policy_profile,
    ordinary_graph_paper_policy_bundle,
    validate_graph_paper_configuration,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_machine_state_and_registries_point_to_existing_authorities() -> None:
    state = tomllib.loads((ROOT / "scientific_archive/project_records/state.toml").read_text(encoding="utf-8"))
    assert state["schema_version"] == "mutaco-project-state-v1"
    assert state["execution_authorized"] is False
    assert state["flagship"]["status"] == "complete"
    assert state["flagship"]["successful_physical_calls"] == 9840
    for registry_name, table in (
        ("scientific_contracts/scientific_registry.toml", "contracts"),
        ("results/experiment_registry.toml", "experiments"),
    ):
        registry = tomllib.loads(
            (ROOT / registry_name).read_text(encoding="utf-8")
        )
        assert registry[table]


def test_canonical_and_paper120_profiles_share_one_loader() -> None:
    canonical = load_graph_policy_profile(CANONICAL_PROFILE)
    paper120 = load_graph_policy_profile(PAPER120_PROFILE)
    assert canonical["simulation_budget"] == paper120["simulation_budget"] == 40
    assert canonical["sequential_points"] == 40 - canonical["initial_points"] == 32
    assert paper120["sequential_points"] == 40 - paper120["initial_points"] == 24
    assert canonical["beta_policy"] == {"id": "fixed_beta_v1", "value": 2.0}
    assert paper120["beta_policy"]["resolved_beta"] == 5.604501123581913
    assert ordinary_graph_paper_policy_bundle(CANONICAL_PROFILE)["profile_id"] == CANONICAL_PROFILE
    assert ordinary_graph_paper_policy_bundle(PAPER120_PROFILE)["profile_id"] == PAPER120_PROFILE


def test_executed_paper120_mutaco_configuration_remains_valid() -> None:
    path = ROOT / "results/paper/graph_paper120/paper120/executed_mutaco_configuration.json"
    validate_graph_paper_configuration(json.loads(path.read_text(encoding="utf-8")))


def test_frozen_paper120_cohort_is_complete_and_byte_verified() -> None:
    cohort = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3"
    assert _sha256(cohort / "manifest.json") == (
        cohort / "manifest.sha256"
    ).read_text(encoding="ascii").strip()
    manifest = json.loads((cohort / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed_count"] == 120
    assert manifest["missing_count"] == 0
    assert manifest["selected_indices"] == list(range(120))
    for index in range(120):
        record = cohort / f"target_{index:03d}.json"
        assert _sha256(record) == record.with_suffix(".sha256").read_text(
            encoding="ascii"
        ).strip()


def test_compact_paper120_evidence_is_complete_and_hash_verified() -> None:
    evidence = ROOT / "results/paper/graph_paper120/paper120"
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["paired_row_count"] == 120
    assert manifest["physical_calls"] == {
        "attempted": 9840,
        "successful": 9840,
        "failed": 0,
        "retry": 0,
    }
    for item in manifest["files"]:
        path = evidence / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert _sha256(path) == item["sha256"]
    rows = [
        json.loads(line)
        for line in (evidence / "paired_records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 120
    assert all(set(row["methods"]) == {"mutaco", "sbi_npe"} for row in rows)


def test_historical_workspaces_are_not_tracked_on_curated_main() -> None:
    assert not (ROOT / "colab").exists()
    assert not (ROOT / "benchmarks/graph/paper100").exists()
    assert not (ROOT / "docs/planning").exists()
    assert not (ROOT / "docs/audits").exists()
