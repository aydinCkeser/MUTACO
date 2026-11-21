"""Offline checks for canonical contract lookup and preserved promotion evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

from experiments.paper.workflow.cli import main


ROOT = Path(__file__).resolve().parents[2]


def read_toml(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_registry_targets_and_legacy_scientific_paths_resolve():
    authority = ROOT / "scientific_contracts" / "scientific_registry.toml"
    assert authority.is_file() and not authority.is_symlink()
    registry = read_toml(authority)
    for record in registry["contracts"] + registry["profiles"]:
        for key in ("path", "machine_contract"):
            if key in record:
                assert (ROOT / record[key]).is_file(), record
    fixture = Path("graph_oscillator/reference_fixtures/retained_cpu_small_v1.json")
    plan = json.loads((ROOT / "results/reproducibility/physical_export_recovery_v1/recapture-plan.json").read_text())
    original = plan["retained_fixture"]
    assert original["path"] == "scientific/" + fixture.as_posix()
    assert hashlib.sha256((ROOT / "scientific_contracts" / fixture).read_bytes()).hexdigest() == original["sha256"]

    # The frozen studies name their original authorities; readers use the active registry.
    studies = ROOT / "experiments/paper/studies/graph_paper120/studies"
    authorities = {record["path"] for record in registry["contracts"]}
    for study, protocol in (
        ("mutaco_nei_sobol_primary_v1", "primary_benchmark_protocol"),
        ("mutaco_scalar_qlognei_sobol_primary_v1", "scalar_qlognei_primary_benchmark_protocol"),
    ):
        declaration = json.loads((studies / f"{study}.json").read_text())
        assert declaration["authority"] == f"scientific/{protocol}/README.md"
        assert f"scientific_contracts/{protocol}/README.md" in authorities


def test_project_snapshot_can_be_copied_with_its_historical_schema(tmp_path):
    archived = ROOT / "scientific_archive" / "project_records"
    copied = shutil.copytree(archived, tmp_path / "project")
    # Hashes of the five original coordination files at 9f3ce18.
    hashes = {
        "state.toml": "6fe34c31ee46481edbd2d9d3b2b2b2fd0383f232a404e809ae5e244ddd779fc4",
        "state.schema.json": "379b963ca6c7512f2a7dbac0b55b61d2b66f3b46a2e5e3d7bbc10ee9e6d7d935",
        "HANDOFF.md": "347f741f89b023cdd3036f45aa5bbb499fec7d370a6a29b0a77cb15e0265afb0",
        "FRESH_CHAT_PROMPT.md": "cbebfd46a27e041c6678dd0afbf8e4263af21bf91dba5a7b0dba5bde5d44f035",
        "UBUNTU_CONTINUATION.md": "bed517d8ada6b77916477b330b0f60e61b73b44b77581c1c3b8844e32f166775",
    }
    for filename, expected in hashes.items():
        assert hashlib.sha256((copied / filename).read_bytes()).hexdigest() == expected
    state = read_toml(copied / "state.toml")
    schema = json.loads((copied / "state.schema.json").read_text())
    assert all(field in state for field in schema["required"])
    assert state["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert state["execution_authorized"] is False
    assert state["license_status"] == "unresolved"


def test_promote_uses_original_independent_bindings_and_active_registry(capsys):
    snapshot = read_toml(ROOT / "scientific_archive/project_records/state.toml")
    bindings = read_toml(ROOT / "results/paper/promotion_state.toml")
    expected = {
        key: {name: record[name] for name in ("compact_evidence", "bundle_identity")}
        for key, record in snapshot.items() if isinstance(record, dict) and "bundle_identity" in record
    }
    assert bindings == expected
    reports = ROOT / "experiments/paper/studies/graph_paper120/reports"
    for spec in (reports / "b40_standardization_ablation_v1.json", reports / "b100_v1.json"):
        main(["promote", "verify", str(spec), "--repository-root", str(ROOT)])
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "promotion_verified"
        assert result["files_written"] == result["scientific_calls"] == 0
