from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.evidence_analysis import (
    ERROR_KEYS,
    adapt_mechanism_ablation_records,
    adapt_paper_v3_bundle,
    adapt_repaired_sbi_assessment_export,
    adapt_scalar_primary_export,
    analysis_specification,
    deterministic_analysis,
    validate_analysis_rows,
)
from experiments.paper.common.evidence_analysis_cli import build
from experiments.paper.common.validation import PaperContractError


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
SCALAR_EXPORT = ROOT / "experiments/paper/exploration/outputs/scalar_primary_smoke/population_b40_v3r2_compact_fetch/population/collection/compact_export.json"
SCALAR_MANIFEST = ROOT / "results/paper/graph_scalar_primary_b40_v1/manifest.json"
HISTORICAL_MANIFEST = ROOT / "results/paper/graph_paper120/paper120_mutaco_npe_snpe_v1/manifest.json"
HISTORICAL_ROWS = ROOT / "results/paper/graph_paper120/paper120_mutaco_npe_snpe_v1/paired_records.jsonl"


@pytest.fixture(scope="module")
def scalar_rows(offline_analysis_root):
    manifest = read_json(SCALAR_MANIFEST)
    source = offline_analysis_root / SCALAR_EXPORT.relative_to(ROOT)
    return adapt_scalar_primary_export(
        read_json(source),
        source_identity=manifest["evidence_identity"],
        source_sha256=file_hash(source),
        cohort=read_json(COHORT),
        cohort_directory=COHORT.parent,
        cohort_sha256=file_hash(COHORT),
    )


def test_scalar_adapter_recomputes_complete_current_policy_grid(scalar_rows) -> None:
    assert len(scalar_rows) == 360
    assert tuple(scalar_rows[0]["errors"]) == ERROR_KEYS
    assert all(row["evidence_statuses"] == ["valid_current_policy_evidence"] for row in scalar_rows)
    mutaco = next(row for row in scalar_rows if row["method"] == "mutaco")
    sobol = next(row for row in scalar_rows if row["method"] == "sobol")
    assert mutaco["errors"]["raw_terminal_surrogate_error"]["available"] is True
    assert mutaco["errors"]["optimized_terminal_surrogate_error"]["available"] is True
    assert sobol["errors"]["optimized_terminal_surrogate_error"]["available"] is False
    assert sobol["recommendation_provenance"]["stage"] == "best_observed"


def test_scalar_analysis_reproduces_frozen_b40_claims(scalar_rows) -> None:
    report = deterministic_analysis(
        scalar_rows,
        analysis_specification(analysis_id="test-b40", bootstrap_seed=92024001, bootstrap_resamples=1000),
    )
    frozen = read_json(SCALAR_MANIFEST)
    for method in ("mutaco", "scalar_qlognei", "sobol"):
        actual = report["error_summaries_component_mean_mae"][method]["heldout_assessment_error"]
        expected = frozen["primary_endpoint_heldout_physical_response_mae"][method]
        for field in ("n", "mean", "standard_deviation", "median", "q1", "q3", "minimum", "maximum"):
            assert actual[field] == pytest.approx(expected[field], abs=1e-14)
    assert report["target_winners"] == frozen["paired_primary_endpoint"]["target_winners"]
    assert report["mean_rank"] == frozen["paired_primary_endpoint"]["mean_rank"]
    assert report["scientific_calls_added_by_analysis"] == 0
    assert report["call_accounting"]["mutaco"]["successful_calls"] == 120 * 41
    assert report["target_winner_ties"] == 0


def test_analysis_rejects_mixed_budget_and_policy_rows(scalar_rows) -> None:
    specification = analysis_specification(analysis_id="test-mix", bootstrap_seed=1, bootstrap_resamples=10)
    mixed_budget = deepcopy(scalar_rows)
    mixed_budget[0]["budget"] = 100
    body = dict(mixed_budget[0])
    body.pop("row_identity")
    from mutaco import stable_hash
    mixed_budget[0]["row_identity"] = stable_hash("cross-experiment-analysis-row-v1", body)
    with pytest.raises(PaperContractError, match="mix budgets"):
        deterministic_analysis(mixed_budget, specification)

    mixed_policy = deepcopy(scalar_rows)
    mixed_policy[0]["native_terminal_policy"] = "ambiguous_other_policy"
    body = dict(mixed_policy[0])
    body.pop("row_identity")
    mixed_policy[0]["row_identity"] = stable_hash("cross-experiment-analysis-row-v1", body)
    with pytest.raises(PaperContractError, match="mix policies"):
        deterministic_analysis(mixed_policy, specification)


def test_scalar_adapter_rejects_metric_and_pairing_corruption(scalar_rows) -> None:
    corrupt = deepcopy(scalar_rows)
    corrupt[0]["assessment"]["context"]["evaluation_seed"] += 1
    body = dict(corrupt[0])
    body.pop("row_identity")
    from mutaco import stable_hash
    corrupt[0]["row_identity"] = stable_hash("cross-experiment-analysis-row-v1", body)
    cohort_ids = tuple(read_json(COHORT)["records"][str(index)]["target_record_id"] for index in range(120))
    with pytest.raises(PaperContractError, match="paired assessment context"):
        validate_analysis_rows(corrupt, expected_methods=("mutaco", "scalar_qlognei", "sobol"), expected_budget=40, expected_targets=cohort_ids)


def test_historical_adapter_preserves_policy_and_partial_context() -> None:
    native = [json.loads(line) for line in HISTORICAL_ROWS.read_text(encoding="utf-8").splitlines() if line]
    manifest = read_json(HISTORICAL_MANIFEST)
    rows = adapt_paper_v3_bundle(
        native,
        source_identity=manifest["consolidation_identity"],
        source_sha256=file_hash(HISTORICAL_ROWS),
        cohort=read_json(COHORT),
        budget=40,
    )
    assert len(rows) == 360
    mutaco = next(row for row in rows if row["method"] == "mutaco")
    npe = next(row for row in rows if row["method"] == "sbi_npe")
    assert mutaco["recommendation_provenance"]["stage"] == "historical_one_step_terminal"
    assert "protocol_noncompliant_for_current_claim" in mutaco["evidence_statuses"]
    assert npe["recommendation_provenance"]["stage"] is None
    assert npe["assessment"]["pairing_status"] == "partial_legacy_context"


def test_analysis_spec_identity_and_mechanism_boundary() -> None:
    assert analysis_specification(analysis_id="x", bootstrap_seed=1, bootstrap_resamples=10) == analysis_specification(analysis_id="x", bootstrap_seed=1, bootstrap_resamples=10)
    assert adapt_mechanism_ablation_records(records=()) == ()
    with pytest.raises(PaperContractError, match="schema must be frozen"):
        adapt_mechanism_ablation_records(records=({"invented": True},))


def test_repaired_sbi_adapter_rejects_corrupt_metric_and_preserves_pairing() -> None:
    source = ROOT / "experiments/paper/exploration/outputs/repaired_mutaco_sbi_salvage/assessment_attempt_6251303/workers/collection/compact_export.json"
    if not source.is_file():
        pytest.skip("raw repaired SBI assessment archive is not present")
    native = read_json(source)
    rows = adapt_repaired_sbi_assessment_export(
        native,
        source_identity=native["collection_identity"],
        source_sha256=file_hash(source),
        cohort=read_json(COHORT),
        cohort_directory=COHORT.parent,
        cohort_sha256=file_hash(COHORT),
        budget=40,
    )
    assert len(rows) == 360
    assert all(row["assessment"]["pairing_status"] == "exact_context_available" for row in rows)
    assert all(row["errors"]["heldout_assessment_error"]["available"] for row in rows)
    assert all(not row["errors"]["observed_search_error"]["available"] for row in rows)
    corrupt = deepcopy(native)
    corrupt["records"][0]["response_error_mae"] += 1.0
    with pytest.raises(PaperContractError, match="metric recomputation mismatch"):
        adapt_repaired_sbi_assessment_export(
            corrupt,
            source_identity=native["collection_identity"],
            source_sha256=file_hash(source),
            cohort=read_json(COHORT),
            cohort_directory=COHORT.parent,
            cohort_sha256=file_hash(COHORT),
            budget=40,
        )


def test_tracked_b40_package_reproduces_without_writes(offline_analysis_root) -> None:
    result = build(repository_root=offline_analysis_root, check=True)
    assert result["status"] == "reproduced"
    assert result["discrepancy_count"] == 0
    assert result["scientific_calls"] == 0
