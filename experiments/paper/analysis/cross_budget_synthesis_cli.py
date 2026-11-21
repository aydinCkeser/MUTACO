"""Materialize the frozen scalar-primary B40/B100 synthesis package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from .cross_budget_synthesis import deterministic_cross_budget_synthesis
from ..common.evidence_analysis import ERROR_KEYS, canonical_json_text, deterministic_analysis
from ..common.validation import PaperContractError


SPEC = "experiments/paper/studies/graph_paper120/analysis/scalar_primary_b40_b100_synthesis_v1.json"
OUTPUT = "results/paper/scalar_primary_b40_b100_synthesis_v1"
INPUTS = {
    40: {
        "rows": "results/paper/cross_experiment_analysis/b40_v1/analysis_rows.jsonl",
        "report": "results/paper/cross_experiment_analysis/b40_v1/analysis_report.json",
        "analysis_spec": "experiments/paper/studies/graph_paper120/analysis/cross_experiment_b40_v1.json",
        "manifest": "results/paper/graph_scalar_primary_b40_v1/manifest.json",
        "package_manifest": "results/paper/cross_experiment_analysis/b40_v1/manifest.json",
        "sha256": {
            "rows": "4bd34f4ba8fdb01e3b9279270b1ed9c971da28970990bad52a2c7bbead801a6f",
            "report": "f46c453b38f3842d67c829a073e593a8e05534d7b4ab187d7a85a1fc62966157",
            "manifest": "ecfbf6621931a17e7bd62e1a8cda13137cbeb7949cc13f7aa4bcef7f60f985be",
            "package_manifest": "5c7f17404d762b12936392b756c113f6388b3c79dad521de7afde41b175e9005",
        },
    },
    100: {
        "rows": "results/paper/graph_scalar_primary_b100_v1/analysis_rows.jsonl",
        "report": "results/paper/graph_scalar_primary_b100_v1/analysis_report.json",
        "analysis_spec": "results/paper/graph_scalar_primary_b100_population_preparation_v3r2/analysis_spec.json",
        "manifest": "results/paper/graph_scalar_primary_b100_v1/manifest.json",
        "sha256": {
            "rows": "e03ddf7359724ab2d680e92b3996725e05f894d4c3228ce3c30bf78dac41e434",
            "report": "ff7f2fc48a5b23621d41764545b04058511a065af038eb6822ef378f38273305",
            "manifest": "7f1c79a1351457cd8f0ee918bcfc0077375635e334803f54beee6fd1428436e8",
        },
    },
}


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        errors = row.get("errors", {})
        if set(errors) != set(ERROR_KEYS):
            raise PaperContractError("persisted analysis row error vocabulary mismatch")
        # Canonical JSON sorts object keys. Restore the normative semantic order
        # expected by validate_analysis_rows without changing row identity.
        row["errors"] = {key: errors[key] for key in ERROR_KEYS}
        rows.append(row)
    return tuple(rows)


def _assert_report_identity(report: Mapping[str, Any]) -> None:
    body = dict(report)
    identity = body.pop("analysis_report_identity", None)
    if identity != stable_hash("cross-experiment-analysis-report-v1", body):
        raise PaperContractError("source analysis report identity mismatch")


def _load_and_audit(root: Path) -> tuple[dict[int, tuple[dict[str, Any], ...]], dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    rows_by_budget = {}
    reports_by_budget = {}
    manifests = {}
    audit_inputs = {}
    for budget, paths in INPUTS.items():
        for key, expected in paths["sha256"].items():
            actual = file_hash(root / paths[key])
            if actual != expected:
                raise PaperContractError(f"B{budget} {key} SHA-256 mismatch")
        rows = _read_jsonl(root / paths["rows"])
        report = read_json(root / paths["report"])
        analysis_spec = read_json(root / paths["analysis_spec"])
        regenerated = deterministic_analysis(rows, analysis_spec)
        if regenerated != report:
            raise PaperContractError(f"B{budget} analysis report does not reproduce from tracked rows")
        _assert_report_identity(report)
        manifest = read_json(root / paths["manifest"])
        rows_by_budget[budget] = rows
        reports_by_budget[budget] = report
        manifests[budget] = manifest
        audit_inputs[str(budget)] = {
            "rows_sha256": paths["sha256"]["rows"],
            "report_sha256": paths["sha256"]["report"],
            "manifest_sha256": paths["sha256"]["manifest"],
            "analysis_report_reproduced": True,
            "evidence_identity": manifest["evidence_identity"],
            "analysis_report_identity": report["analysis_report_identity"],
        }
    return rows_by_budget, reports_by_budget, manifests, audit_inputs


def _content(root: Path) -> dict[str, str]:
    specification = read_json(root / SPEC)
    rows_by_budget, reports, manifests, audit_inputs = _load_and_audit(root)
    rows, report = deterministic_cross_budget_synthesis(
        rows_by_budget,
        reports,
        manifests,
        specification,
    )
    rows_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    report_text = canonical_json_text(report)
    audit_body = {
        "schema_version": "mutaco-scalar-primary-cross-budget-integrity-audit-v1",
        "inputs": audit_inputs,
        "same_ordered_target_cohort": True,
        "complete_target_method_budget_grid": True,
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    audit = {**audit_body, "audit_identity": stable_hash("scalar-primary-cross-budget-integrity-audit-v1", audit_body)}
    audit_text = canonical_json_text(audit)
    manifest_body = {
        "schema_version": "mutaco-scalar-primary-cross-budget-synthesis-package-v1",
        "status": "complete_zero_call_deterministic_synthesis",
        "synthesis_spec_identity": specification["synthesis_spec_identity"],
        "synthesis_report_identity": report["synthesis_report_identity"],
        "integrity_audit_identity": audit["audit_identity"],
        "source_evidence_identities": report["source_evidence_identities"],
        "source_analysis_report_identities": report["source_analysis_report_identities"],
        "rows_content_identity": stable_hash("scalar-primary-cross-budget-rows-file-v1", rows_text),
        "row_count": len(rows),
        "target_count": 120,
        "budgets": [40, 100],
        "methods": ["mutaco", "scalar_qlognei", "sobol"],
        "campaign_relation": "independent_campaigns_same_ordered_target_cohort_not_nested",
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    package = {**manifest_body, "package_identity": stable_hash("scalar-primary-cross-budget-synthesis-package-v1", manifest_body)}
    return {
        "cross_budget_rows.jsonl": rows_text,
        "synthesis_report.json": report_text,
        "integrity_audit.json": audit_text,
        "manifest.json": canonical_json_text(package),
    }


def build(*, repository_root: Path, check: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    contents = _content(root)
    output = root / OUTPUT
    if check:
        for name, expected in contents.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise PaperContractError(f"tracked cross-budget artifact differs from deterministic materialization: {name}")
            sidecar = output / f"{name}.sha256"
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked cross-budget sidecar differs: {name}")
        status = "reproduced"
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            (output / f"{name}.sha256").write_text(file_hash(path) + "\n", encoding="ascii", newline="\n")
        status = "materialized"
    manifest = json.loads(contents["manifest.json"])
    return {
        "status": status,
        "output": OUTPUT,
        "package_identity": manifest["package_identity"],
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    print(canonical_json_text(build(repository_root=arguments.repository_root, check=arguments.check)), end="")


# Preserve historical pickle/global references while both imports share this module.
for _legacy_export in (
    _read_jsonl,
    _assert_report_identity,
    _load_and_audit,
    _content,
    build,
    main,
):
    _legacy_export.__module__ = "experiments.paper.common.cross_budget_synthesis_cli"
del _legacy_export


if __name__ == "__main__":
    main()
