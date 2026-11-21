from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.paper.common.method_specs import (
    DEFERRED_SBI_COMPARISON_GROUP,
    PRIMARY_COMPARISON_GROUP,
    load_method_execution_specs,
    selected_method_configuration_paths,
)
from experiments.paper.common.population_v3 import MethodDeclarationV3
from experiments.paper.common.protocol_v3 import PaperComparisonConfigV3
from experiments.paper.common.supplied_targets import load_supplied_target_set
from experiments.paper.common.validation import PaperContractError


ROOT = Path(__file__).resolve().parents[2]
METHOD_SPECS = ROOT / "experiments/graph_inversion/configs/paper/paper120_primary_method_specs_v1.json"


def _comparison():
    return PaperComparisonConfigV3(
        simulator_definition_identity="graph-definition",
        simulator_configuration_identity="graph-configuration",
        assessment_fidelity="m20",
        assessment_campaign_seed=22,
        inference_campaign_seed=11,
    )


def test_primary_methods_share_one_group_while_likelihood_ratio_sbi_is_separate():
    specs = load_method_execution_specs(
        json.loads(METHOD_SPECS.read_text()), source=METHOD_SPECS
    )
    assert {spec.comparison_group for spec in specs} == {PRIMARY_COMPARISON_GROUP}
    deferred = {
        "schema_version": "mutaco-paper-method-spec-set-v1",
        "state": "unexecuted",
        "execution_requires_approval": True,
        "methods": [{
            "method": "sbi_nle",
            "method_family": "likelihood",
            "comparison_group": DEFERRED_SBI_COMPARISON_GROUP,
            "adapter": "deferred",
            "configuration_path": "placeholder.json",
            "configuration_sha256": "0" * 64,
            "inference_budget": 40,
            "decision_policy_identifier": "not_approved",
        }],
    }
    loaded = load_method_execution_specs(deferred, source=Path("deferred.json"))
    with pytest.raises(PaperContractError, match="lacks an approved"):
        selected_method_configuration_paths(
            loaded,
            (MethodDeclarationV3("sbi_nle", "config", "not_approved", 40),),
            root=ROOT,
        )


def test_supplied_y_star_requires_frozen_identity_and_randomness_provenance():
    comparison = _comparison()
    document = {
        "schema_version": "mutaco-paper-supplied-target-set-v1",
        "state": "frozen",
        "simulator_definition_identity": comparison.simulator_definition_identity,
        "simulator_configuration_identity": comparison.simulator_configuration_identity,
        "records": [{
            "target_record_id": "supplied-target-a",
            "target_family_id": "supplied-family",
            "target_index": 0,
            "target_seed": 123,
            "target_disposition": "externally_supplied_frozen",
            "target_response": [1.0, 2.0],
            "theta_star": None,
            "target_randomness_provenance": {
                "source": "external_measurement",
                "disorder_coordinate": "declared-external-disorder",
                "thermal_coordinate": "declared-external-thermal",
            },
        }],
    }
    records = load_supplied_target_set(document, comparison=comparison)
    assert records[0]["target_response"] == [1.0, 2.0]
    assert records[0]["theta_star"] is None
    invalid = json.loads(json.dumps(document))
    invalid["records"][0].pop("target_randomness_provenance")
    with pytest.raises(PaperContractError, match="randomness provenance"):
        load_supplied_target_set(invalid, comparison=comparison)


def test_method_spec_budget_and_configuration_hash_are_cross_checked(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    document = {
        "schema_version": "mutaco-paper-method-spec-set-v1",
        "state": "unexecuted",
        "execution_requires_approval": True,
        "methods": [{
            "method": "sbi_npe",
            "method_family": "direct_posterior",
            "comparison_group": PRIMARY_COMPARISON_GROUP,
            "adapter": "direct_posterior",
            "configuration_path": "config.json",
            "configuration_sha256": "0" * 64,
            "inference_budget": 40,
            "decision_policy_identifier": "posterior_sample_map_physical_v1",
        }],
    }
    specs = load_method_execution_specs(document, source=tmp_path / "methods.json")
    declaration = MethodDeclarationV3(
        "sbi_npe", "identity", "posterior_sample_map_physical_v1", 41
    )
    with pytest.raises(PaperContractError, match="budgets differ"):
        selected_method_configuration_paths(specs, (declaration,), root=tmp_path)
