from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.cross_experiment_contract import (
    ERROR_QUANTITIES,
    EVIDENCE_STATUSES,
    PAPER120_COHORT_SHA256,
    RECOMMENDATION_PROVENANCE,
    contract_document,
    load_cross_experiment_contract,
    operation_owner,
    validate_primary_budget_pair,
    validate_primary_declaration_contract,
)
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark.scalar_protocol import (
    load_scalar_primary_population_declaration,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "scientific_contracts/cross_experiment_contract/contract.json"
COHORT = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
B40 = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_v3r2.json"
B100 = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_v3r2.json"
B40_CONFIGURATION = ROOT / "experiments/graph_inversion/configs/paper/policy_upgrade_100_targets_v1.json"
B100_CONFIGURATION = ROOT / "experiments/graph_inversion/configs/paper/paper120_b100_mutaco_v1.json"


def _declaration(path: Path):
    return load_scalar_primary_population_declaration(read_json(path))


def test_machine_contract_is_exactly_canonical_and_unambiguous() -> None:
    persisted = read_json(CONTRACT)
    assert load_cross_experiment_contract(persisted) == contract_document()
    assert persisted["canonical_cohort"]["manifest_sha256"] == file_hash(COHORT)
    assert set(ERROR_QUANTITIES) == {
        "observed_search_error",
        "raw_terminal_surrogate_error",
        "optimized_terminal_surrogate_error",
        "heldout_assessment_error",
    }
    assert tuple(persisted["recommendation_provenance"]) == RECOMMENDATION_PROVENANCE
    assert tuple(persisted["evidence_statuses"]) == EVIDENCE_STATUSES
    assert (
        ERROR_QUANTITIES["raw_terminal_surrogate_error"]["response_source"]
        != ERROR_QUANTITIES["heldout_assessment_error"]["response_source"]
    )


def test_contract_rejects_semantically_edited_persisted_document() -> None:
    changed = deepcopy(read_json(CONTRACT))
    changed["randomness_invariants"]["paired_assessment_omits_method"] = False
    with pytest.raises(PaperContractError, match="not canonical"):
        load_cross_experiment_contract(changed)


def test_randomness_owner_enforces_method_pairing_boundary() -> None:
    common = {
        "study_identity": "study-v1",
        "target_identity": "target-v1",
        "budget": 40,
        "index": 0,
    }
    assessment = operation_owner(
        **common,
        role="paired_heldout_assessment",
    )
    assert assessment["method"] is None
    assert assessment["pairing_scope"] == "target_budget_comparison"

    mutaco = operation_owner(**common, role="search_evaluation", method="mutaco")
    qlognei = operation_owner(
        **common, role="search_evaluation", method="scalar_qlognei"
    )
    assert mutaco["ownership_identity"] != qlognei["ownership_identity"]
    assert mutaco["pairing_scope"] == "method_specific"

    with pytest.raises(PaperContractError, match="must omit method"):
        operation_owner(
            **common,
            role="paired_heldout_assessment",
            method="mutaco",
        )
    with pytest.raises(PaperContractError, match="requires a method"):
        operation_owner(**common, role="terminal_recommendation")


def test_target_construction_has_no_budget_or_method_owner() -> None:
    owner = operation_owner(
        study_identity="paper120-v3-construction",
        target_identity="target-position-0",
        budget=None,
        role="target_construction",
        index=0,
    )
    assert owner["budget"] is None
    assert owner["method"] is None
    with pytest.raises(PaperContractError, match="must omit budget"):
        operation_owner(
            study_identity="paper120-v3-construction",
            target_identity="target-position-0",
            budget=40,
            role="target_construction",
            index=0,
        )


def test_frozen_scalar_b40_and_b100_satisfy_common_backbone() -> None:
    cohort = read_json(COHORT)
    b40 = _declaration(B40)
    b100 = _declaration(B100)
    result = validate_primary_budget_pair(
        (b40, b100),
        cohort_manifest=cohort,
        cohort_manifest_sha256=file_hash(COHORT),
        simulator_configurations={
            40: read_json(B40_CONFIGURATION),
            100: read_json(B100_CONFIGURATION),
        },
        simulator_configuration_sha256={
            40: file_hash(B40_CONFIGURATION),
            100: file_hash(B100_CONFIGURATION),
        },
    )
    assert result["status"] == "valid"
    assert result["budgets"] == [40, 100]
    assert result["campaigns_independent"] is True
    assert result["common_cohort"] == "paper120_v3"
    assert PAPER120_COHORT_SHA256 == file_hash(COHORT)


def test_primary_validation_rejects_cohort_order_drift() -> None:
    cohort = read_json(COHORT)
    records = cohort["records"]
    records["0"], records["1"] = records["1"], records["0"]
    with pytest.raises(PaperContractError, match="target order"):
        validate_primary_declaration_contract(
            _declaration(B40),
            cohort_manifest=cohort,
            cohort_manifest_sha256=PAPER120_COHORT_SHA256,
            simulator_configuration=read_json(B40_CONFIGURATION),
            simulator_configuration_sha256=file_hash(B40_CONFIGURATION),
        )


def test_contract_json_is_stable_pretty_json() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == contract_document()
