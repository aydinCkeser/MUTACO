from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.paper.workflow.cli import main
from experiments.paper.workflow.consolidation import (
    build_consolidated_bundle,
    verify_consolidated_bundle,
)
from experiments.paper.workflow.identities import document_identity
from experiments.paper.workflow.ingestion import validate_ingestion_bindings
from experiments.benchmarking.artifacts import file_hash
from experiments.paper.workflow.promotion import verify_promotion
from experiments.paper.workflow.reporting import build_report, validate_report_source
from experiments.paper.workflow.schemas import load_and_validate, validate_document
from experiments.paper.workflow.transport import (
    export_compact_population,
    fetch_compact_export,
)


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/paper/studies/graph_paper120"


def test_b40_report_is_bound_to_tracked_accepted_evidence() -> None:
    spec = load_and_validate(STUDY / "reports/b40_v1.json")
    result = validate_report_source(spec, ROOT)
    assert result["status"] == "accepted_source_validated"
    assert result["scientific_calls"] == 0
    assert result["report_written"] is False
    assert result["claims"]["physical_calls"]["successful_calls"] == 14760
    assert result["claims"]["paired_response_mae"]["mutaco__sbi_snpe"]["left_wins"] == 102


def test_b100_report_is_bound_to_canonical_accepted_evidence() -> None:
    spec = load_and_validate(STUDY / "reports/b100_v1.json")
    result = validate_report_source(spec, ROOT)
    assert result["status"] == "accepted_source_validated"
    assert result["source_sha256"] == spec["source"]["sha256"]
    assert result["claims"]["target_count"] == 120
    assert result["claims"]["physical_calls"]["successful_calls"] == 36360
    assert result["claims"]["paired_response_mae"]["mutaco__sbi_npe"]["left_wins"] == 78
    assert result["claims"]["paired_response_mae"]["mutaco__sbi_npe"][
        "paired_mean_bootstrap_95_percentile_interval"
    ] == [-0.040837308308665964, -0.016941881667029073]
    report = (ROOT / spec["output"]).read_text(encoding="utf-8")
    assert "accepted canonical repository evidence" in report


def test_b100_canonical_bundle_independently_verifies() -> None:
    result = verify_consolidated_bundle(
        ROOT / "results/paper/graph_paper120/b100_v1"
    )
    assert result["status"] == "verified"
    assert result["target_count"] == 120
    assert result["record_count"] == 360
    assert result["assessment_coordinates_shared_across_methods_per_target"] is True
    assert result["scientific_calls"] == 0


def test_b100_report_is_deterministically_reproduced() -> None:
    spec = load_and_validate(STUDY / "reports/b100_v1.json")
    result = build_report(spec, ROOT, confirm_authorized=False, check=True)
    assert result["status"] == "report_reproduced"
    assert result["report_written"] is False
    assert result["scientific_calls"] == 0


def test_b100_promotion_state_is_aligned() -> None:
    spec = load_and_validate(STUDY / "reports/b100_v1.json")
    result = verify_promotion(
        spec,
        ROOT,
        state_path=ROOT / "results/paper/promotion_state.toml",
        registry_path=ROOT / "results/experiment_registry.toml",
    )
    assert result["status"] == "promotion_verified"
    assert result["files_written"] == 0
    assert result["scientific_calls"] == 0


def test_b100_ingestion_is_identity_bound_and_requires_runtime_authorization(
    tmp_path: Path,
) -> None:
    spec = load_and_validate(STUDY / "ingestion/b100_v1.json")
    assert spec["expected_target_count"] == 120
    assert spec["expected_physical_calls"]["successful_calls"] == 36360
    assert spec["consolidation_authorized"] is False
    with pytest.raises(Exception, match="explicit runtime authorization"):
        build_consolidated_bundle(
            spec,
            ["qualification3=missing", "remaining117=missing"],
            repository_root=ROOT,
            output=tmp_path / "bundle",
            confirm_authorized=False,
        )


@pytest.mark.parametrize("budget", ("b40", "b100"))
def test_figure_scaffolds_are_inert(budget: str) -> None:
    spec = load_and_validate(STUDY / f"figures/{budget}_v1.toml")
    assert spec["state"] == "discussion_pending"
    assert spec["rendering_authorized"] is False
    assert spec["panels"] == []
    assert not list((ROOT / spec["output_root"]).glob("*.png"))
    assert not list((ROOT / spec["output_root"]).glob("*.pdf"))
    assert not list((ROOT / spec["output_root"]).glob("*.svg"))


def test_identity_changes_with_semantic_content() -> None:
    document = load_and_validate(STUDY / "reports/b40_v1.json")
    changed = deepcopy(document)
    changed["inference_budget_per_method_target"] = 41
    assert document_identity(changed) != document["report_identity"]


def test_exploration_template_is_valid_and_not_promoted() -> None:
    spec = load_and_validate(
        ROOT / "experiments/paper/workflow/templates/exploration_v1.json"
    )
    assert spec["classification"] == "exploratory_non_paper"
    assert spec["promotion_status"] == "not_promoted"
    assert spec["scientific_execution_authorized"] is False


def test_ingestion_interface_binds_sources_without_copying(tmp_path: Path) -> None:
    compact = tmp_path / "compact"
    compact.mkdir()
    (compact / "manifest.json").write_text("{}\n", encoding="utf-8")
    (compact / "paired_records.jsonl").write_text("", encoding="utf-8")
    plan = {
        "schema_version": "mutaco-paper-ingestion-spec-v1",
        "ingestion_id": "test-ingestion",
        "bundle_id": "test-bundle",
        "cohort_identity": "test-cohort",
        "methods": ["mutaco"],
        "source_fragments": [{"source_id": "existing", "kind": "compact_bundle"}],
        "consolidation_authorized": False,
    }
    plan["ingestion_identity"] = document_identity(plan)
    validated = validate_document(plan)
    result = validate_ingestion_bindings(validated, [f"existing={compact}"])
    assert result["status"] == "validated"
    assert result["scientific_calls"] == 0
    assert result["files_copied"] == 0
    assert result["consolidation_performed"] is False


def _write_hash_pair(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.with_suffix(".sha256").write_text(f"{file_hash(path)}\n", encoding="ascii")


def test_compact_export_and_fetch_are_deterministic_and_exclude_raw_data(
    tmp_path: Path,
) -> None:
    population = tmp_path / "workers"
    _write_hash_pair(population / "collection/population_collection.json", "{}\n")
    _write_hash_pair(population / "worker-000/records/t/mutaco.json", "{}\n")
    _write_hash_pair(
        population / "worker-000/assessment_randomness/t.json", "{}\n"
    )
    (population / "worker-000/methods/t/model.pt").parent.mkdir(parents=True)
    (population / "worker-000/methods/t/model.pt").write_bytes(b"raw model")
    first = export_compact_population(
        population, tmp_path / "export/first", confirm_authorized=True
    )
    second = export_compact_population(
        population, tmp_path / "export/second", confirm_authorized=True
    )
    assert first["archive_sha256"] == second["archive_sha256"]
    fetched = fetch_compact_export(
        (tmp_path / "export/first").as_posix(),
        tmp_path / "fetched",
        confirm_authorized=True,
    )
    extracted = Path(fetched["population_root"])
    assert (extracted / "collection/population_collection.json").is_file()
    assert (extracted / "worker-000/records/t/mutaco.json").is_file()
    assert not list(extracted.rglob("*.pt"))
    assert fetched["scientific_calls"] == 0


def test_compact_fetch_rejects_tampered_archive(tmp_path: Path) -> None:
    population = tmp_path / "workers"
    _write_hash_pair(population / "collection/population_collection.json", "{}\n")
    prefix = tmp_path / "export/population"
    export_compact_population(population, prefix, confirm_authorized=True)
    with Path(f"{prefix}.tar.gz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(Exception, match="archive hash mismatch"):
        fetch_compact_export(
            prefix.as_posix(), tmp_path / "fetched", confirm_authorized=True
        )


def test_cli_validates_inert_figure_without_writing(capsys: pytest.CaptureFixture[str]) -> None:
    spec = STUDY / "figures/b40_v1.toml"
    main(["figure", "validate", str(spec)])
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "discussion_pending"
    assert output["figures_written"] == 0


def test_cli_run_plan_is_identity_bound_and_zero_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study = {
        "schema_version": "mutaco-paper-study-spec-v1",
        "study_id": "test-study",
        "revision": "v1",
        "classification": "test_only",
        "domain": "synthetic",
        "scientific_inputs": {"methods": ["mutaco"], "budget": 1},
    }
    study["study_identity"] = document_identity(study)
    execution = {
        "schema_version": "mutaco-paper-execution-spec-v1",
        "execution_id": "test-execution",
        "study_identity": study["study_identity"],
        "source_commit": "0" * 40,
        "execution_authorized": False,
        "runtime": {"python": "test"},
        "scheduler": {"site": "none"},
    }
    execution["execution_identity"] = document_identity(execution)
    study_path = tmp_path / "study.json"
    execution_path = tmp_path / "execution.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    main(["run", "plan", str(study_path), str(execution_path)])
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "planned_unexecuted"
    assert output["scientific_calls"] == 0
    assert output["job_submitted"] is False


def test_cli_scientific_and_materializing_commands_are_unavailable() -> None:
    for arguments in (("run", "submit"), ("figure", "build")):
        with pytest.raises(SystemExit, match="deliberately unavailable"):
            main(list(arguments))
