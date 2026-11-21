"""Deterministic materialization for the tracked B40 analysis acceptance fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from ..common.evidence_analysis import (
    SPEC_IDENTITY_DOMAIN,
    adapt_scalar_primary_export,
    canonical_json_text,
    deterministic_analysis,
)
from ..common.validation import PaperContractError


SPEC = "experiments/paper/studies/graph_paper120/analysis/cross_experiment_b40_v1.json"
COHORT = "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
SOURCE_MANIFEST = "results/paper/graph_scalar_primary_b40_v1/manifest.json"
SOURCE_EXPORT = "experiments/paper/exploration/outputs/scalar_primary_smoke/population_b40_v3r2_compact_fetch/population/collection/compact_export.json"
OUTPUT = "results/paper/cross_experiment_analysis/b40_v1"


def _assert_spec(document: Mapping[str, Any]) -> None:
    body = dict(document)
    identity = body.pop("analysis_spec_identity", None)
    if identity != stable_hash(SPEC_IDENTITY_DOMAIN, body):
        raise PaperContractError("tracked analysis specification identity mismatch")


def _audit(report: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    discrepancies = []

    def compare(path: str, actual: Any, expected: Any) -> None:
        if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if abs(float(actual) - float(expected)) <= 1e-14:
                return
        elif actual == expected:
            return
        discrepancies.append({"path": path, "generated": actual, "historical_manifest": expected})

    for method in ("mutaco", "scalar_qlognei", "sobol"):
        current = report["error_summaries_component_mean_mae"][method]
        for field in ("n", "mean", "standard_deviation", "median", "q1", "q3", "minimum", "maximum"):
            compare(
                f"heldout.{method}.{field}",
                current["heldout_assessment_error"][field],
                frozen["primary_endpoint_heldout_physical_response_mae"][method][field],
            )
            compare(
                f"observed_search.{method}.{field}",
                current["observed_search_error"][field],
                frozen["secondary_noisy_best_observed_search_mae_at_call_40"][method].get(field, 120 if field == "n" else None),
            )
        compare(f"target_winners.{method}", report["target_winners"][method], frozen["paired_primary_endpoint"]["target_winners"][method])
        compare(f"mean_rank.{method}", report["mean_rank"][method], frozen["paired_primary_endpoint"]["mean_rank"][method])
    for pair, frozen_values in frozen["paired_primary_endpoint"].items():
        if pair in {"target_winners", "mean_rank"}:
            continue
        left, right = pair.split("_minus_", 1)
        current = report["paired_primary_endpoint"][f"{left}__{right}"]
        compare(f"paired.{pair}.mean", current["mean_left_minus_right"], frozen_values["mean"])
        compare(f"paired.{pair}.left_wins", current["left_wins"], frozen_values[f"{left}_better"])
        compare(f"paired.{pair}.left_losses", current["left_losses"], frozen_values[f"{right}_better"])
        compare(f"paired.{pair}.ties", current["ties"], frozen_values["ties"])
    body = {
        "schema_version": "mutaco-cross-experiment-b40-discrepancy-audit-v1",
        "comparison_scope": "generated canonical scalar-primary B40 statistics versus historical compact manifest claims",
        "numeric_tolerance": 1e-14,
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
        "historical_evidence_modified": False,
        "scientific_calls": 0,
    }
    return {**body, "audit_identity": stable_hash("cross-experiment-b40-discrepancy-audit-v1", body)}


def _content(root: Path) -> dict[str, str]:
    specification = read_json(root / SPEC)
    _assert_spec(specification)
    cohort_path = root / COHORT
    source_path = root / SOURCE_EXPORT
    source_manifest_path = root / SOURCE_MANIFEST
    frozen = read_json(source_manifest_path)
    actual_source_sha256 = file_hash(source_path)
    if frozen["integrity"]["compact_export_sha256"] != actual_source_sha256:
        raise PaperContractError("scalar-primary source export hash differs from archived manifest")
    rows = adapt_scalar_primary_export(
        read_json(source_path),
        source_identity=frozen["evidence_identity"],
        source_sha256=actual_source_sha256,
        cohort=read_json(cohort_path),
        cohort_directory=cohort_path.parent,
        cohort_sha256=file_hash(cohort_path),
    )
    report = deterministic_analysis(rows, specification)
    audit = _audit(report, frozen)
    rows_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    report_text = canonical_json_text(report)
    audit_text = canonical_json_text(audit)
    rows_sha = stable_hash("cross-experiment-analysis-rows-file-v1", rows_text)
    manifest_body = {
        "schema_version": "mutaco-cross-experiment-analysis-package-v1",
        "analysis_spec_identity": specification["analysis_spec_identity"],
        "analysis_report_identity": report["analysis_report_identity"],
        "source_evidence_identity": frozen["evidence_identity"],
        "source_export_sha256": actual_source_sha256,
        "cohort_sha256": file_hash(cohort_path),
        "row_count": len(rows),
        "target_count": report["target_count"],
        "methods": report["methods"],
        "budgets": report["budgets"],
        "rows_content_identity": rows_sha,
        "discrepancy_audit_identity": audit["audit_identity"],
        "discrepancy_count": audit["discrepancy_count"],
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    manifest = {**manifest_body, "package_identity": stable_hash("cross-experiment-analysis-package-v1", manifest_body)}
    return {
        "analysis_rows.jsonl": rows_text,
        "analysis_report.json": report_text,
        "discrepancy_audit.json": audit_text,
        "manifest.json": canonical_json_text(manifest),
    }


def build(*, repository_root: Path, check: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    output = root / OUTPUT
    contents = _content(root)
    if check:
        for name, expected in contents.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise PaperContractError(f"tracked analysis artifact differs from deterministic materialization: {name}")
            sidecar = output / f"{name}.sha256"
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked analysis sidecar differs from artifact: {name}")
        status = "reproduced"
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            (output / f"{name}.sha256").write_text(file_hash(path) + "\n", encoding="ascii", newline="\n")
        status = "materialized"
    manifest = json.loads(contents["manifest.json"])
    return {"status": status, "output": OUTPUT, "package_identity": manifest["package_identity"], "discrepancy_count": manifest["discrepancy_count"], "scientific_calls": 0, "truba_jobs_submitted": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    print(canonical_json_text(build(repository_root=arguments.repository_root, check=arguments.check)), end="")


# Preserve historical pickle/global references while both imports share this module.
for _legacy_export in (
    _assert_spec,
    _audit,
    _content,
    build,
    main,
):
    _legacy_export.__module__ = "experiments.paper.common.evidence_analysis_cli"
del _legacy_export


if __name__ == "__main__":
    main()
