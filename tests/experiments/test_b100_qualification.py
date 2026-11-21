from __future__ import annotations

import json
from pathlib import Path

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
    shard_indices,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "experiments/graph_inversion/configs/paper"
PROFILE = ROOT / "experiments/graph_inversion/configs/profiles/graph_paper_joint95_mae_initial16_b100_v1.json"
DECLARATION = PAPER / "paper120_b100_three_method_qualification_v1.json"
METHOD_SPECS = PAPER / "paper120_b100_primary_method_specs_v1.json"
TARGET_LEDGER = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3"
FULL_B40_DECLARATION = PAPER / "paper120_b40_mutaco_npe_v3_v1.json"
REMAINING_DECLARATION = PAPER / "paper120_b100_mutaco_npe_snpe_remaining117_v1.json"


def test_b100_profile_and_method_configurations_are_locked_and_exact(tmp_path):
    profile = load_graph_policy_profile(PROFILE.stem)
    assert profile["simulation_budget"] == 100
    assert profile["initial_points"] == 16
    assert profile["sequential_points"] == 84

    specs = load_method_execution_specs(
        json.loads(METHOD_SPECS.read_text(encoding="utf-8")), source=METHOD_SPECS
    )
    declaration = load_population_declaration_v3(
        json.loads(DECLARATION.read_text(encoding="utf-8"))
    )
    paths = selected_method_configuration_paths(specs, declaration.methods, root=ROOT)
    assert set(paths) == {"mutaco", "sbi_npe", "sbi_snpe"}
    for spec in specs:
        assert spec.inference_budget == 100
        assert file_hash(paths[spec.method]) == spec.configuration_sha256
    for method in declaration.methods:
        assert method.configuration_identity == method_configuration_identity(
            method.method, paths[method.method]
        )

    mutaco = json.loads(paths["mutaco"].read_text(encoding="utf-8"))
    validate_graph_paper_configuration(mutaco)
    assert mutaco["call_accounting"]["initial_inference"] == 16
    assert mutaco["call_accounting"]["sequential_inference"] == 84
    assert json.loads(paths["sbi_npe"].read_text())["simulation_budget_per_target"] == 100
    snpe = json.loads(paths["sbi_snpe"].read_text())
    assert snpe["sbi"]["sequential_round_allocation"] == [34, 33, 33]

    GraphPopulationCallbacks(
        method_configurations=paths,
        output=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_b100_qualification_reuses_three_frozen_targets_at_exact_cap():
    document = json.loads(DECLARATION.read_text(encoding="utf-8"))
    declaration = load_population_declaration_v3(document)
    assert declaration.document() == document
    assert tuple(method.method for method in declaration.methods) == (
        "mutaco", "sbi_npe", "sbi_snpe"
    )
    assert population_call_preview(3, declaration.methods) == {
        "target_construction_calls": 0,
        "maximum_inference_calls": 900,
        "maximum_assessment_calls": 9,
        "maximum_total_calls": 909,
        "recommendation_calls": 0,
        "scientific_calls_executed": 0,
    }
    assert population_call_preview(1, declaration.methods)["maximum_total_calls"] == 303
    records = _target_records(TARGET_LEDGER, declaration, (0, 1, 2))
    assert tuple(record["target_index"] for record in records) == (0, 49, 99)
    assert tuple(record["target_record_id"] for record in records) == declaration.target_record_identities
    assert declaration.comparison.inference_campaign_seed == 92020401
    assert declaration.comparison.assessment_campaign_seed == 92020402


def test_b100_remaining117_is_exact_complement_at_exact_worker_and_global_caps():
    full = load_population_declaration_v3(
        json.loads(FULL_B40_DECLARATION.read_text(encoding="utf-8"))
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
    assert population_call_preview(117, remaining.methods)["maximum_total_calls"] == 35451
    shards = tuple(shard_indices(117, rank=rank, workers=39) for rank in range(39))
    assert all(len(shard) == 3 for shard in shards)
    assert sorted(index for shard in shards for index in shard) == list(range(117))
    assert all(
        population_call_preview(len(shard), remaining.methods)["maximum_total_calls"] == 909
        for shard in shards
    )
