from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.benchmarking.artifacts import read_json
from experiments.paper.common.terminal_lineage_forensics import (
    B100_POPULATION_COMMIT,
    B100_QUALIFICATION_COMMIT,
    build_b100_rows,
    build_b40_rows,
    forensic_audit,
    forensic_report,
    forensic_specification,
    validate_forensic_rows,
    verify_exact_source_git,
)
from experiments.paper.common.terminal_lineage_forensics_cli import build as build_forensic_package
from experiments.paper.common.validation import PaperContractError


ROOT = Path(__file__).resolve().parents[2]
COHORT_PATH = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
B40_RAW = ROOT / "runs/paper/graph_oscillator/graph_paper120_b40_mutaco_npe_v3_v1"
B40_COMPACT = ROOT / "results/paper/graph_paper120/paper120/paired_records.jsonl"
B100_COMPACT = ROOT / "results/paper/graph_paper120/b100_v1/paired_records.jsonl"


@pytest.fixture(scope="module")
def evidence():
    cohort = read_json(COHORT_PATH)
    b40 = build_b40_rows(repository_root=ROOT, raw_root=B40_RAW, compact_path=B40_COMPACT, cohort=cohort)
    b100 = build_b100_rows(compact_path=B100_COMPACT, cohort=cohort)
    return cohort, b40, b100


def test_exact_source_blobs_are_identical_and_verified() -> None:
    result = verify_exact_source_git(ROOT, forensic_specification())
    assert result["status"] == "verified"
    assert set(result["commits"]) == {
        "9235d2fefefa6af0d35cc3af2dfaa01a2d56a3a9",
        B100_QUALIFICATION_COMMIT,
        B100_POPULATION_COMMIT,
    }


def test_b40_direct_lineage_and_all_error_quantities(evidence) -> None:
    _, b40, _ = evidence
    assert len(b40) == 120
    assert sum(len(row["restart_iterations"]) for row in b40) == 1200
    assert all(set(row["restart_iterations"]) == {1} for row in b40)
    assert all(row["posterior_calls"] == 21 for row in b40)
    assert all(all(value["available"] for value in row["errors"].values()) for row in b40)
    assert all(row["path_classification"] in {"historical_one_step_terminal", "raw_fallback_after_one_step_attempts"} for row in b40)


def test_b100_retains_endpoints_but_not_target_terminal_branches(evidence) -> None:
    _, _, b100 = evidence
    assert len(b100) == 120
    assert all(row["path_classification"] == "indeterminate_from_retained_evidence" for row in b100)
    assert all(row["errors"]["E_heldout"]["available"] for row in b100)
    assert all(not row["errors"][key]["available"] for row in b100 for key in ("E_obs", "E_raw_GP", "E_term_GP"))
    assert sum(row["source_commit"] == B100_QUALIFICATION_COMMIT for row in b100) == 3
    assert sum(row["source_commit"] == B100_POPULATION_COMMIT for row in b100) == 117


def test_report_and_audit_are_deterministic(evidence) -> None:
    cohort, b40, b100 = evidence
    rows = b40 + b100
    source = verify_exact_source_git(ROOT, forensic_specification())
    report = forensic_report(rows, forensic_specification(), cohort)
    audit = forensic_audit(rows, source)
    assert report == forensic_report(rows, forensic_specification(), cohort)
    assert report["b40"]["one_iteration_restart_histories"] == 1200
    assert report["b100"]["path_classifications"] == {"indeterminate_from_retained_evidence": 120}
    assert audit["mismatch_count"] == 0


def test_forensic_grid_rejects_identity_or_budget_corruption(evidence) -> None:
    cohort, b40, b100 = evidence
    corrupt = list(deepcopy(b40 + b100))
    corrupt[0]["budget"] = 100
    with pytest.raises(PaperContractError, match="identity mismatch"):
        validate_forensic_rows(corrupt, cohort)


def test_tracked_forensic_package_reproduces() -> None:
    result = build_forensic_package(repository_root=ROOT, check=True)
    assert result["status"] == "reproduced"
    assert result["mismatch_count"] == 0
    assert result["scientific_calls"] == 0
