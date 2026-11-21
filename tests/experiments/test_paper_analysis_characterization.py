"""Fixed pre-relocation contracts, plus independent archived report bytes."""
from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from experiments.paper.common import aggregate, timing_report
from experiments.paper.common import cross_budget_synthesis_cli as synthesis
from experiments.paper.common import evidence_analysis_cli as analysis
from experiments.paper.common.validation import PaperContractError


ROOT = Path(__file__).resolve().parents[2]


def test_fixed_descriptive_schema_and_rng_isolation() -> None:
    state = random.getstate()
    values = [3.0, 1.0, 2.0]
    assert aggregate._descriptive(values) == {
        "count": 3, "mean": 2.0, "median": 2.0, "sample_standard_deviation": 1.0,
        "q1": 1.5, "q3": 2.5, "iqr": 1.0, "p90": 2.8000000000000003, "p95": 2.9,
        "minimum": 1.0, "maximum": 3.0, "lower_whisker": 1.0,
        "upper_whisker": 3.0, "outliers": [],
    }
    assert aggregate._bootstrap_mean_interval([2.0] * 3, seed=17, resamples=11) == [2.0, 2.0]
    assert values == [3.0, 1.0, 2.0] and random.getstate() == state


def test_fixed_paired_summary_and_first_failure() -> None:
    schema = "mutaco-paper-target-method-record-v2"
    rows = tuple({"schema_version": schema, "target_record_identity": "A", "method": method,
                  "status": "completed", "target_error_l1": value,
                  "target_error_l2": value * 2, "design_error_l2": value * 3}
                 for method, value in (("mutaco", 1.0), ("sbi_npe", 3.0)))
    assert aggregate.summarize_paired(rows) == {
        "record_schema": schema, "join_key": "target_record_identity",
        "metrics": ("target_error_l1", "target_error_l2", "design_error_l2"),
        "methods": {
            "mutaco": {"completed": 1, "failed": 0, "means": {
                "target_error_l1": 1.0, "target_error_l2": 2.0, "design_error_l2": 3.0}},
            "sbi_npe": {"completed": 1, "failed": 0, "means": {
                "target_error_l1": 3.0, "target_error_l2": 6.0, "design_error_l2": 9.0}},
        },
        "paired": [{"left": "mutaco", "right": "sbi_npe", "target_count": 1,
                    "mean_left_minus_right_target_l2": -4.0}],
    }
    with pytest.raises(PaperContractError, match="duplicate target-method records"):
        aggregate.summarize_paired(rows + rows)
    with pytest.raises(PaperContractError, match="unsupported paper record cohort schema"):
        aggregate.summarize_paired([dict(rows[0], schema_version="unsupported")] * 2)


def test_fixed_timing_unknowns_and_format() -> None:
    report = timing_report.summarize_persisted_timings([
        {"method": "mutaco", "timings": {"total": 5.0}},
        {"method": "mutaco", "timings": None},
    ])
    assert report["schema_version"] == "mutaco-paper-persisted-timing-report-v2"
    assert report["source"] == "validated_persisted_records_only"
    assert report["unknown_duration_policy"] == "unavailable_is_null_never_zero"
    method = report["methods"]["mutaco"]
    assert method["timing_completeness"] == {"historical_total_only": 1, "timing_unavailable": 1}
    assert method["common_phase_times"]["method_total"] == {
        "wall_seconds": 5.0, "process_cpu_seconds": None, "wall_records": 1,
        "process_cpu_records": 0, "target_method_records": 2,
        "wall_complete": False, "process_cpu_complete": False,
    }
    assert method["common_phase_times"]["final_decision"]["wall_seconds"] is None


@pytest.mark.parametrize("module", [analysis, synthesis], ids=["B40-analysis", "cross-budget"])
def test_complete_offline_outputs_equal_independent_archived_files(module, offline_analysis_root) -> None:
    # These files predate the relocation. Expected rows, bootstrap statistics,
    # identities, schemas, key ordering and newline formatting come from them.
    contents = module._content(offline_analysis_root)
    assert len(contents) == 4
    for name, text in contents.items():
        assert text.encode() == (ROOT / module.OUTPUT / name).read_bytes()
        assert text.endswith("\n")
    result = module.build(repository_root=offline_analysis_root, check=True)
    assert result["status"] == "reproduced" and result["scientific_calls"] == 0


@pytest.mark.parametrize("module", [analysis, synthesis], ids=["B40-analysis", "cross-budget"])
def test_check_rejects_changed_output_and_sidecar(module, tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "fixture"
    destination.mkdir()
    monkeypatch.setattr(module, "OUTPUT", "fixture")
    monkeypatch.setattr(module, "_content", lambda root: {"manifest.json": '{"fixed":true}\n'})
    path = destination / "manifest.json"
    path.write_text('{"fixed":false}\n')
    with pytest.raises(PaperContractError, match="artifact differs"):
        module.build(repository_root=tmp_path, check=True)
    path.write_text('{"fixed":true}\n')
    path.with_name("manifest.json.sha256").write_text("wrong\n")
    with pytest.raises(PaperContractError, match="sidecar differs"):
        module.build(repository_root=tmp_path, check=True)


def test_b100_adapter_source_identity_remains_exact() -> None:
    import hashlib

    manifest = json.loads((ROOT / "results/paper/graph_scalar_primary_b100_population_preparation_v3r2/manifest.json").read_text())
    source = manifest["analysis"]
    assert source["adapter_source"] == "experiments/paper/common/evidence_analysis.py"
    assert hashlib.sha256((ROOT / source["adapter_source"]).read_bytes()).hexdigest() == source["adapter_source_sha256"]
