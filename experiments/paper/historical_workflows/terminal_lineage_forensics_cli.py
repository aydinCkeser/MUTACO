"""Deterministic materialization of historical MUTACO terminal-lineage evidence."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import argparse
import json
from pathlib import Path
from typing import Any

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from experiments.paper.common.evidence_analysis import canonical_json_text
from .terminal_lineage_forensics import (
    build_b100_rows,
    build_b40_rows,
    forensic_audit,
    forensic_report,
    forensic_specification,
    validate_forensic_rows,
    validate_specification,
    verify_exact_source_git,
)
from experiments.paper.protocol.validation import PaperContractError


SPEC = "experiments/paper/studies/graph_paper120/analysis/historical_terminal_lineage_forensics_v1.json"
COHORT = "data/cohorts/paper120_v3/manifest.json"
B40_RAW = "runs/paper/graph_oscillator/graph_paper120_b40_mutaco_npe_v3_v1"
B40_COMPACT = "benchmarks/graph/paper120/paired_records.jsonl"
B100_COMPACT = "benchmarks/paper/graph_paper120/b100_v1/paired_records.jsonl"
OUTPUT = "results/paper/historical_terminal_lineage_forensics/v1"


def _content(root: Path) -> dict[str, str]:
    specification = validate_specification(read_json(root / SPEC))
    cohort = read_json(resolve_recorded_path(root, COHORT))
    source_verification = verify_exact_source_git(root, specification)
    b40 = build_b40_rows(
        repository_root=root,
        raw_root=root / B40_RAW,
        compact_path=resolve_recorded_path(root, B40_COMPACT),
        cohort=cohort,
    )
    b100 = build_b100_rows(compact_path=resolve_recorded_path(root, B100_COMPACT), cohort=cohort)
    rows = b40 + b100
    validate_forensic_rows(rows, cohort)
    report = forensic_report(rows, specification, cohort)
    audit = forensic_audit(rows, source_verification)
    rows_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    manifest_body = {
        "schema_version": "mutaco-historical-terminal-lineage-forensic-package-v1",
        "forensic_spec_identity": specification["forensic_spec_identity"],
        "forensic_report_identity": report["forensic_report_identity"],
        "forensic_audit_identity": audit["forensic_audit_identity"],
        "cohort_sha256": file_hash(resolve_recorded_path(root, COHORT)),
        "b40_compact_sha256": file_hash(resolve_recorded_path(root, B40_COMPACT)),
        "b100_compact_sha256": file_hash(resolve_recorded_path(root, B100_COMPACT)),
        "row_count": len(rows),
        "target_count_per_budget": 120,
        "budgets": [40, 100],
        "rows_content_identity": stable_hash("historical-terminal-lineage-forensic-rows-file-v1", rows_text),
        "b40_raw_files_verified": len(b40),
        "exact_source_verification": source_verification,
        "mismatch_count": audit["mismatch_count"],
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    manifest = {
        **manifest_body,
        "package_identity": stable_hash("historical-terminal-lineage-forensic-package-v1", manifest_body),
    }
    return {
        "forensic_rows.jsonl": rows_text,
        "forensic_report.json": canonical_json_text(report),
        "forensic_audit.json": canonical_json_text(audit),
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
                raise PaperContractError(
                    f"tracked forensic artifact differs from deterministic materialization: {name}"
                )
            sidecar = output / f"{name}.sha256"
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked forensic sidecar differs from artifact: {name}")
        status = "reproduced"
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            (output / f"{name}.sha256").write_text(
                file_hash(path) + "\n", encoding="ascii", newline="\n"
            )
        status = "materialized"
    manifest = json.loads(contents["manifest.json"])
    return {
        "status": status,
        "output": OUTPUT,
        "package_identity": manifest["package_identity"],
        "mismatch_count": manifest["mismatch_count"],
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-spec", action="store_true")
    arguments = parser.parse_args()
    if arguments.print_spec:
        print(canonical_json_text(forensic_specification()), end="")
        return
    print(
        canonical_json_text(build(repository_root=arguments.repository_root, check=arguments.check)),
        end="",
    )


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.terminal_lineage_forensics_cli")


if __name__ == "__main__":
    main()
