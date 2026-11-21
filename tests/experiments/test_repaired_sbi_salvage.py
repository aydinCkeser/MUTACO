from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.paper.common.repaired_sbi_salvage import (
    inventory_salvage,
    salvage_design,
    validate_salvage_design,
)
from experiments.paper.common.repaired_sbi_salvage_cli import build
from experiments.paper.common.validation import PaperContractError


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inventory():
    return inventory_salvage(ROOT, salvage_design())


def test_salvage_design_is_canonical_and_execution_locked() -> None:
    design = validate_salvage_design(salvage_design())
    assert design["call_formula"]["best_case"] == 720
    assert design["call_formula"]["worst_case"] == 17520
    assert len(set(design["assessment_policy"][key] for key in (
        "b40_assessment_campaign_seed", "b100_assessment_campaign_seed"
    ))) == 2
    assert not any(design["authorization"].values())


def test_b40_is_complete_and_sample_exact_state_replay_passes(inventory) -> None:
    b40 = inventory["b40"]
    assert b40["raw_trajectory_files_local_verified"] == 120
    assert b40["complete_observation_trajectories"] == 120
    assert b40["complete_fit_lineages"] == 120
    assert sum(b40["runtime_counts"].values()) == 120
    assert b40["exact_model_replays_completed"] == 120
    assert b40["current_classification"] == "population_exact_state_replay_passed_corrected_terminal_pending"
    assert b40["search_rerun_calls_required"] == 0


def test_b100_raw_recovery_is_complete_and_population_replay_is_pending(inventory) -> None:
    b100 = inventory["b100"]
    assert b100["raw_trajectory_files_local_verified"] == 120
    assert b100["remote_existence_verified_by_this_gate"]
    assert b100["qualification3_raw_recovery_path_registered"]
    assert b100["exact_model_replays_completed"] == 120
    assert b100["assessment_calls"] == 360
    assert b100["search_rerun_calls_required"] == 0


def test_inventory_freezes_causal_boundary_and_call_formula(inventory) -> None:
    assert inventory["causal_boundary"]["search_effect_of_terminal_defect"] == "none"
    assert inventory["combined_call_preview"] == {
        "formula": "720 + 40*n40_rerun + 100*n100_rerun",
        "best_case": 720,
        "worst_case": 17520,
        "resolved_search_rerun_calls": 0,
        "resolved_remaining_scientific_calls": 720,
    }
    assert not inventory["readiness"]["b40_scientific_execution_ready"]
    assert not inventory["readiness"]["b100_scientific_execution_ready"]


def test_design_rejects_policy_drift() -> None:
    corrupt = deepcopy(salvage_design())
    corrupt["assessment_policy"]["b100_assessment_campaign_seed"] = corrupt["assessment_policy"]["b40_assessment_campaign_seed"]
    with pytest.raises(PaperContractError, match="not canonical"):
        validate_salvage_design(corrupt)


def test_tracked_salvage_package_reproduces() -> None:
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["best_case_new_calls"] == 720
    assert result["scientific_calls"] == 0
