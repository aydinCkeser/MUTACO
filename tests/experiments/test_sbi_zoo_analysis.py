from __future__ import annotations

from pathlib import Path

from experiments.paper.studies.graph_paper120.analysis import sbi_zoo_comparison as analysis


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_evidence_forms_exact_paired_grid():
    specification = analysis._read_spec(ROOT)
    rows = analysis.load_rows(ROOT, specification)
    assert len(rows) == 720
    assert {
        (row["target_position"], row["budget"], row["method"]) for row in rows
    } == {
        (position, budget, method)
        for position in range(120)
        for budget in (40, 100)
        for method in ("mutaco", "nle", "nre")
    }
    for position in range(120):
        for budget in (40, 100):
            contexts = [
                row["assessment_context"]
                for row in rows
                if row["target_position"] == position and row["budget"] == budget
            ]
            assert contexts[0] == contexts[1] == contexts[2]


def test_analysis_direction_and_response_parameter_separation():
    specification = analysis._read_spec(ROOT)
    rows = analysis.load_rows(ROOT, specification)
    result = analysis.analyze(rows, specification)
    assert result["primary_metric"] == "response_error_mae"
    assert set(result["response_recovery"]) == {"40", "100"}
    assert set(result["parameter_recovery_separate"]) == {"40", "100"}
    assert result["paired_vs_mutaco"]["40"]["nle"]["difference"] == "competitor_minus_mutaco"
    assert result["cross_budget_change"]["nle"]["difference"] == "b100_minus_b40"
    assert result["target_winners"]["40"]["counts"]["mutaco"] == 112
    assert result["target_winners"]["100"]["counts"]["mutaco"] == 93
    assert result["scientific_calls_added_by_analysis"] == 0


def test_tracked_analysis_package_reproduces_without_scientific_execution():
    result = analysis.build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["scientific_calls"] == 0


def test_analysis_surface_has_no_simulator_or_training_path():
    source = (ROOT / analysis.SPEC_PATH.with_name("sbi_zoo_comparison.py")).read_text()
    assert "checked_evaluate" not in source
    assert "GraphScientificD4Adapter" not in source
    assert "train_single_round" not in source
    assert "recommend_sample_map" not in source
