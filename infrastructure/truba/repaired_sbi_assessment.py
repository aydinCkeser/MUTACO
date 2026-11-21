"""Computation-free validation of repaired SBI assessment TRUBA geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.historical_workflows.repaired_sbi_assessment_execution import CALLS_PER_WORKER, WORKERS
from experiments.paper.historical_workflows.repaired_sbi_assessment_preparation import assessment_declaration
from experiments.paper.common.validation import PaperContractError


def validate(package: Path, *, workers: int, maximum_calls: int) -> dict:
    manifest_path = package / "manifest.json"
    if file_hash(manifest_path) != (package / "manifest.json.sha256").read_text(encoding="ascii").strip():
        raise PaperContractError("assessment preparation manifest hash mismatch")
    manifest = read_json(manifest_path)
    if workers != WORKERS or maximum_calls != CALLS_PER_WORKER or manifest.get("task_count") != 720:
        raise PaperContractError("assessment TRUBA geometry must be 120 workers and six calls each")
    if manifest.get("declaration_identity") != assessment_declaration()["declaration_identity"]:
        raise PaperContractError("assessment declaration identity mismatch")
    return {
        "schema_version": "mutaco-repaired-sbi-assessment-truba-preview-v1",
        "package_identity": manifest["package_identity"],
        "workers": workers,
        "target_budget_groups_per_worker": 2,
        "methods_per_group": 3,
        "maximum_calls_per_worker": maximum_calls,
        "global_assessment_call_cap": workers * maximum_calls,
        "nodes": 30,
        "workers_per_node": 4,
        "cpus_per_worker": 5,
        "automatic_retry": False,
        "assessment_authorized": False,
        "job_submitted": False,
        "scientific_calls_executed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--maximum-calls", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.package, workers=args.workers, maximum_calls=args.maximum_calls), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
