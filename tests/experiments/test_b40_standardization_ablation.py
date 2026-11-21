from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.benchmarking.artifacts import file_hash
from experiments.graph_inversion.paper_policies import (
    load_graph_policy_profile,
    validate_graph_paper_configuration,
)
from experiments.graph_inversion.paper_v3_callbacks import (
    GraphPopulationCallbacks,
    method_configuration_identity,
)
from experiments.graph_inversion.run_paper_v3_worker import _target_records
from experiments.paper.common.method_specs import (
    load_method_execution_specs,
    selected_method_configuration_paths,
)
from experiments.paper.common.population_v3 import (
    load_population_declaration_v3,
    population_call_preview,
)
from experiments.paper.common.protocol_v3 import paired_assessment_context
from experiments.paper.workflow.schemas import load_and_validate


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "experiments/graph_inversion/configs/paper"
PROFILE_DIR = ROOT / "experiments/graph_inversion/configs/profiles"
STANDARD_CONFIG = PAPER / "policy_upgrade_100_targets_v1.json"
RAW_CONFIG = PAPER / "paper120_b40_mutaco_unstandardized_v1.json"
METHOD_SPECS = PAPER / "paper120_b40_standardization_ablation_method_specs_v1.json"
DECLARATION = PAPER / "paper120_b40_standardization_ablation_qualification_v1.json"
REMAINING_DECLARATION = PAPER / "paper120_b40_standardization_ablation_remaining117_v1.json"
TARGET_LEDGER = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3"
STUDY = (
    ROOT
    / "experiments/paper/studies/graph_paper120/studies/b40_standardization_ablation_v1.json"
)
B40_RECORDS = ROOT / "results/paper/graph_paper120/paper120_mutaco_npe_snpe_v1/paired_records.jsonl"
B100_RECORDS = ROOT / "results/paper/graph_paper120/b100_v1/paired_records.jsonl"
ABLATION_RECORDS = (
    ROOT
    / "results/paper/graph_paper120/b40_standardization_ablation_v1/paired_records.jsonl"
)
QUALIFICATION_EVIDENCE = (
    ROOT / "results/paper/graph_paper120/b40_standardization_ablation_qualification_v1"
)


def _rows(path: Path) -> dict[str, dict[str, object]]:
    return {
        row["target_record_identity"]: row
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }


def test_standardization_profiles_are_authoritative_and_distinct() -> None:
    standard = load_graph_policy_profile("graph_paper_joint95_mae_initial16_v1")
    unstandardized = load_graph_policy_profile(
        "graph_paper_joint95_mae_initial16_no_standardization_v1"
    )
    assert standard["standardization"] == {"ddof": 1, "scale_floor": 1e-12}
    assert unstandardized["standardization"] == {"policy": "identity_physical_v1"}
    standard_config = json.loads(STANDARD_CONFIG.read_text(encoding="utf-8"))
    raw_config = json.loads(RAW_CONFIG.read_text(encoding="utf-8"))
    validate_graph_paper_configuration(standard_config)
    validate_graph_paper_configuration(raw_config)
    assert standard_config["runner"]["transform"] == standard["standardization"]
    assert raw_config["runner"]["transform"] == unstandardized["standardization"]
    broken = deepcopy(standard_config)
    broken["runner"]["transform"]["ddof"] = 0
    with pytest.raises(ValueError, match="runner transform differs"):
        validate_graph_paper_configuration(broken)


def test_ablation_method_specs_and_identities_use_one_mutaco_adapter(tmp_path: Path) -> None:
    declaration = load_population_declaration_v3(
        json.loads(DECLARATION.read_text(encoding="utf-8"))
    )
    specs = load_method_execution_specs(
        json.loads(METHOD_SPECS.read_text(encoding="utf-8")), source=METHOD_SPECS
    )
    paths = selected_method_configuration_paths(specs, declaration.methods, root=ROOT)
    assert tuple(paths) == ("mutaco_standardized", "mutaco_unstandardized")
    assert {spec.adapter for spec in specs} == {"mutaco"}
    assert len({method.configuration_identity for method in declaration.methods}) == 2
    for method in declaration.methods:
        assert method.configuration_identity == method_configuration_identity(
            method.method, paths[method.method]
        )
        spec = next(item for item in specs if item.method == method.method)
        assert spec.configuration_sha256 == file_hash(paths[method.method])
    GraphPopulationCallbacks(
        method_configurations=paths,
        output=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_qualification_targets_and_cap_are_exact() -> None:
    document = json.loads(DECLARATION.read_text(encoding="utf-8"))
    declaration = load_population_declaration_v3(document)
    assert declaration.document() == document
    assert population_call_preview(3, declaration.methods) == {
        "target_construction_calls": 0,
        "maximum_inference_calls": 240,
        "maximum_assessment_calls": 6,
        "maximum_total_calls": 246,
        "recommendation_calls": 0,
        "scientific_calls_executed": 0,
    }
    assert population_call_preview(1, declaration.methods)["maximum_total_calls"] == 82
    records = _target_records(TARGET_LEDGER, declaration, (0, 1, 2))
    assert tuple(record["target_index"] for record in records) == (0, 49, 99)
    assert declaration.comparison.assessment_campaign_seed == 92020402


def test_remaining117_is_exact_complement_at_exact_worker_and_global_caps() -> None:
    full = load_population_declaration_v3(
        json.loads((PAPER / "paper120_b40_mutaco_npe_v3_v1.json").read_text(encoding="utf-8"))
    )
    qualification = load_population_declaration_v3(
        json.loads(DECLARATION.read_text(encoding="utf-8"))
    )
    document = json.loads(REMAINING_DECLARATION.read_text(encoding="utf-8"))
    remaining = load_population_declaration_v3(document)
    assert remaining.document() == document
    assert len(remaining.target_record_identities) == 117
    assert set(remaining.target_record_identities).isdisjoint(
        qualification.target_record_identities
    )
    assert set(remaining.target_record_identities) | set(
        qualification.target_record_identities
    ) == set(full.target_record_identities)
    assert remaining.methods == qualification.methods
    assert population_call_preview(117, remaining.methods)["maximum_total_calls"] == 9594
    from experiments.paper.common.population_v3 import shard_indices

    shards = tuple(shard_indices(117, rank=rank, workers=39) for rank in range(39))
    assert all(len(shard) == 3 for shard in shards)
    assert sorted(index for shard in shards for index in shard) == list(range(117))
    assert all(
        population_call_preview(len(shard), remaining.methods)["maximum_total_calls"] == 246
        for shard in shards
    )


def test_promoted_qualification_payload_is_complete_and_hash_paired() -> None:
    manifest = json.loads((QUALIFICATION_EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    population = QUALIFICATION_EVIDENCE / "population"
    collection_path = population / "collection/population_collection.json"
    summary_path = population / "collection/population_summary.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    assert manifest["role"] == "engineering_qualification_not_population_performance_claim"
    assert manifest["scientific_calls_added_by_fetch_and_promotion"] == 0
    assert manifest["collection_sha256"] == file_hash(collection_path)
    assert manifest["summary_sha256"] == file_hash(summary_path)
    assert collection["status"] == "completed"
    assert collection["declared_target_count"] == 3
    assert collection["declared_slot_count"] == collection["collected_slot_count"] == 6
    assert collection["physical_calls"] == {
        "attempted_calls": 246,
        "failed_calls": 0,
        "maximum_calls": 246,
        "successful_calls": 246,
    }
    json_files = sorted(population.rglob("*.json"))
    assert len(json_files) == 15
    for path in json_files:
        sidecar = path.with_suffix(".sha256")
        assert sidecar.read_text(encoding="ascii").strip() == file_hash(path)


def test_all_120_assessment_coordinates_match_historical_b40_and_b100() -> None:
    declaration = load_population_declaration_v3(
        json.loads(DECLARATION.read_text(encoding="utf-8"))
    )
    b40 = _rows(B40_RECORDS)
    b100 = _rows(B100_RECORDS)
    ablation = _rows(ABLATION_RECORDS)
    assert set(b40) == set(b100) == set(ablation)
    assert len(b40) == 120
    for target, historical_b40 in b40.items():
        expected = paired_assessment_context(declaration.comparison, target)
        b40_method = historical_b40["methods"]["sbi_snpe"]
        b100_methods = b100[target]["methods"]
        assert b40_method["assessment_context"] == expected
        for row in b100_methods.values():
            assert row["assessment_context"] == expected
            assert row["assessment_randomness_evidence"] == b40_method[
                "assessment_randomness_evidence"
            ]
        for row in ablation[target]["methods"].values():
            assert row["assessment_context"] == expected
            assert row["assessment_randomness_evidence"] == b40_method[
                "assessment_randomness_evidence"
            ]


def test_study_and_truba_validation_profile_are_locked_and_unexecuted() -> None:
    study = load_and_validate(STUDY)
    assert study["scientific_inputs"]["target_count"] == 120
    assert study["scientific_inputs"]["primary_assessment_metric"] == "response_error_mae"
    assert study["scientific_inputs"]["qualification_maximum_calls"] == 246
    assert study["scientific_inputs"]["full_cohort_maximum_calls"] == 9840
    assert study["scientific_inputs"]["scientific_execution_authorized"] is False
    env = (
        ROOT
        / "infrastructure/truba/paper120_b40_standardization_ablation_qualification_v1.env"
    ).read_text(encoding="utf-8")
    assert "MUTACO_V3_MAXIMUM_CALLS=82" in env
    assert "MUTACO_V3_WORKERS=3" in env
    assert "Prepared for validate-only" in env
    remaining_env = (
        ROOT
        / "infrastructure/truba/paper120_b40_standardization_ablation_remaining117_v1.env"
    ).read_text(encoding="utf-8")
    assert "MUTACO_V3_MAXIMUM_CALLS=246" in remaining_env
    assert "MUTACO_V3_WORKERS=39" in remaining_env
    assert "MUTACO_V3_SBATCH_NTASKS=40" in remaining_env
    assert "Approved for validation only" in remaining_env
