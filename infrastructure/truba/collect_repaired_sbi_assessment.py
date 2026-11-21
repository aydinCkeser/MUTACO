"""Collect repaired MUTACO--SBI assessment evidence without simulator calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mutaco import stable_hash
from experiments.benchmarking.artifacts import write_pretty_json, file_hash, read_json
from experiments.paper.historical_workflows.repaired_sbi_assessment_execution import load_hashed, shard_tasks
from experiments.paper.historical_workflows.repaired_sbi_assessment_preparation import assessment_declaration
from experiments.paper.common.validation import PaperContractError


def _write(path: Path, document: dict[str, Any]) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise PaperContractError("assessment collection is append-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pretty_json(path, document)
    path.with_suffix(".sha256").write_text(file_hash(path) + "\n", encoding="ascii", newline="\n")


def collect(*, tasks: list[dict], workers_root: Path, output: Path, workers: int, expected_commit: str, package_identity: str) -> dict[str, Any]:
    declaration_identity = assessment_declaration()["declaration_identity"]
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    ledger = {"maximum_calls": 0, "attempted_calls": 0, "successful_calls": 0, "failed_calls": 0, "retry_calls": 0}
    for rank in range(workers):
        root = workers_root / f"worker-{rank:03d}"
        try:
            start = load_hashed(root / "worker_start.json")
            terminal = load_hashed(root / "worker_terminal.json")
            expected_tasks = shard_tasks(tasks, rank=rank, workers=workers)
            if (
                start.get("source_commit") != expected_commit
                or terminal.get("source_commit") != expected_commit
                or start.get("declaration_identity") != declaration_identity
                or terminal.get("declaration_identity") != declaration_identity
                or start.get("package_identity") != package_identity
                or terminal.get("package_identity") != package_identity
            ):
                raise PaperContractError("assessment worker provenance mismatch")
            for key in ledger:
                ledger[key] += int(terminal["physical_calls"][key])
            for task in expected_tasks:
                path = root / "attempts" / f"b{task['budget']}" / f"target-{task['target_position']:03d}" / f"{task['method']}.json"
                record = load_hashed(path)
                if record.get("task_identity") != task["task_identity"]:
                    raise PaperContractError("assessment collected task mismatch")
                records.append(record)
            if terminal.get("status") != "completed":
                raise PaperContractError("assessment worker terminal reports failure")
        except Exception as exc:
            errors.append({"rank": rank, "failure_type": type(exc).__name__, "failure_explanation": str(exc)})
    completed = [record for record in records if record.get("status") == "completed"]
    if len(records) != 720 or len({record.get("task_identity") for record in records}) != 720:
        errors.append({"failure_type": "AssessmentCoverageError", "failure_explanation": "assessment task coverage is incomplete or duplicated"})
    if ledger != {"maximum_calls": 720, "attempted_calls": 720, "successful_calls": 720, "failed_calls": 0, "retry_calls": 0}:
        errors.append({"failure_type": "AssessmentCallLedgerError", "failure_explanation": "assessment global call ledger is not exactly 720 successful calls"})
    body = {
        "schema_version": "mutaco-repaired-sbi-assessment-collection-v1",
        "source_commit": expected_commit,
        "declaration_identity": declaration_identity,
        "package_identity": package_identity,
        "record_count": len(records),
        "completed_record_count": len(completed),
        "physical_calls": ledger,
        "errors": errors,
        "status": "failed" if errors else "completed",
        "scientific_calls_added_by_collection": 0,
    }
    collection = {**body, "collection_identity": stable_hash("repaired-sbi-assessment-collection-v1", body)}
    _write(output / "collection.json", collection)
    if not errors:
        compact = {
            "schema_version": "mutaco-repaired-sbi-assessment-compact-v1",
            "collection_identity": collection["collection_identity"],
            "records": [{
                key: record[key]
                for key in (
                    "result_identity", "task_identity", "budget", "target_position",
                    "target_record_id", "method", "recommendation_identity",
                    "recommended_theta", "assessment_context", "assessment_response",
                    "assessment_randomness_evidence", "response_error_l1_sum",
                    "response_error_mae", "response_error_l2", "parameter_error_l2",
                    "call_accounting", "timing",
                )
            } for record in records],
            "scientific_calls_added_by_export": 0,
        }
        _write(output / "compact_export.json", compact)
    if errors:
        raise PaperContractError("assessment collection failed; available evidence was preserved")
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--package-identity", required=True)
    args = parser.parse_args()
    sidecar = args.tasks.parent / f"{args.tasks.name}.sha256"
    if (
        not args.tasks.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip() != file_hash(args.tasks)
    ):
        raise PaperContractError("assessment task hash pair is invalid")
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line]
    result = collect(
        tasks=tasks, workers_root=args.workers_root, output=args.output,
        workers=args.workers, expected_commit=args.expected_commit,
        package_identity=args.package_identity,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
