from __future__ import annotations

from pathlib import Path

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.exact_terminal_generation import _without_timings
from experiments.paper.common.repaired_sbi_terminal_preparation import (
    BOOTSTRAP_PATH,
    DECLARATION_PATH,
    GENERATOR_PATH,
    OUTPUT,
    build,
    corrected_terminal_declaration,
)


ROOT = Path(__file__).resolve().parents[2]


def test_corrected_terminal_declaration_freezes_only_permitted_delta() -> None:
    declaration = corrected_terminal_declaration()
    assert declaration["terminal_cases"] == 240
    assert declaration["source_policy"]["overlay_path"] == (
        "src/mutaco/recommendation/_continuous.py"
    )
    assert declaration["source_policy"]["all_other_files"] == "native_commit_unchanged"
    assert declaration["model_policy"]["optimizer_rerun"] is False
    assert declaration["terminal_policy"]["objective"] == (
        "mean_m_abs_mu_physical_m_theta_minus_target_physical_m"
    )
    assert declaration["terminal_policy"]["dtype"] == "float32"
    assert declaration["call_accounting"]["simulator_calls"] == 0
    assert not any(declaration["authorization"].values())


def test_preparation_inventory_covers_every_target_budget_once() -> None:
    inventory = read_json(ROOT / OUTPUT / "preparation_inventory.json")
    keys = [(case["budget"], case["target_position"]) for case in inventory["cases"]]
    assert keys == [(budget, position) for budget in (40, 100) for position in range(120)]
    assert inventory["case_count"] == 240
    assert inventory["exact_state_cases_passed"] == 240
    assert inventory["terminal_recommendations_generated"] == 0
    assert inventory["simulator_calls"] == 0
    assert inventory["generation_ready"] is True
    assert all(case["raw_sha256"] and case["exact_replay_identity"] for case in inventory["cases"])


def test_terminal_preparation_package_and_declaration_are_reproducible() -> None:
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["terminal_cases_ready"] == 240
    assert result["terminal_recommendations_generated"] == 0
    assert result["scientific_calls"] == 0
    manifest = read_json(ROOT / OUTPUT / "manifest.json")
    assert manifest["generator_sha256"] == file_hash(ROOT / GENERATOR_PATH)
    assert manifest["bootstrap_sha256"] == file_hash(ROOT / BOOTSTRAP_PATH)
    assert not manifest["generation_authorized"]
    assert read_json(ROOT / DECLARATION_PATH) == read_json(ROOT / OUTPUT / "declaration.json")


def test_recommendation_semantics_remove_timings_recursively() -> None:
    value = {
        "selected_theta": [0.1, 0.2],
        "elapsed_seconds": 4.0,
        "component_timings": {"raw_seconds": 1.0},
        "restart_histories": [{"best_score": 0.3, "elapsed_seconds": 2.0}],
    }
    assert _without_timings(value) == {
        "selected_theta": [0.1, 0.2],
        "restart_histories": [{"best_score": 0.3}],
    }


def test_generation_surface_contains_no_simulator_import() -> None:
    for relative in (GENERATOR_PATH, BOOTSTRAP_PATH):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "graph_simulator" not in source
        assert "analytic_simulator" not in source
