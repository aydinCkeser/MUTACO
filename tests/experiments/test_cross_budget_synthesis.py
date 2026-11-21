from __future__ import annotations

from pathlib import Path

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.cross_budget_synthesis_cli import OUTPUT, build


ROOT = Path(__file__).resolve().parents[2]


def test_tracked_cross_budget_synthesis_reproduces() -> None:
    result = build(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["scientific_calls"] == 0
    assert result["truba_jobs_submitted"] == 0


def test_cross_budget_package_is_complete_and_hash_paired() -> None:
    output = ROOT / OUTPUT
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "complete_zero_call_deterministic_synthesis"
    assert manifest["row_count"] == 360
    assert manifest["target_count"] == 120
    assert manifest["campaign_relation"] == "independent_campaigns_same_ordered_target_cohort_not_nested"
    for path in output.glob("*.json*"):
        if path.suffix == ".sha256":
            continue
        assert (output / f"{path.name}.sha256").read_text(encoding="ascii").strip() == file_hash(path)


def test_cross_budget_scientific_claim_boundaries_and_results() -> None:
    report = read_json(ROOT / OUTPUT / "synthesis_report.json")
    endpoints = report["primary_endpoint_by_budget"]
    assert endpoints["40"]["heldout_component_mean_mae"]["mutaco"]["mean"] == 0.021502998957449875
    assert endpoints["100"]["heldout_component_mean_mae"]["mutaco"]["mean"] == 0.019611348367865662
    assert endpoints["40"]["target_winners"] == {"mutaco": 94, "scalar_qlognei": 20, "sobol": 6}
    assert endpoints["100"]["target_winners"] == {"mutaco": 80, "scalar_qlognei": 33, "sobol": 7}
    changes = report["descriptive_target_blocked_cross_budget_change"]
    assert changes["mutaco"]["summary_b100_minus_b40"]["mean"] < 0
    assert changes["scalar_qlognei"]["summary_b100_minus_b40"]["mean"] < 0
    assert changes["sobol"]["summary_b100_minus_b40"]["mean"] < 0
    prohibited = " ".join(report["claim_ledger"]["prohibited"])
    assert "continuation" in prohibited
    assert "mechanism causality" in prohibited
    terminal = report["terminal_refinement_effect_by_budget"]
    assert terminal["40"]["mutaco"]["available"] is True
    assert terminal["100"]["mutaco"]["available"] is True
    assert terminal["40"]["scalar_qlognei"]["available"] is False
    assert report["scientific_calls_added_by_synthesis"] == 0
