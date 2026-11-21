"""Archive and deterministically analyze repaired MUTACO--SBI job 6251303."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import argparse
import json
from pathlib import Path
from typing import Any

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.evidence_analysis import (
    adapt_repaired_sbi_assessment_export,
    analysis_specification,
    canonical_json_text,
    deterministic_analysis,
)
from experiments.paper.common.validation import PaperContractError


RAW = "experiments/paper/exploration/outputs/repaired_mutaco_sbi_salvage/assessment_attempt_6251303"
OUTPUT = "results/paper/repaired_mutaco_sbi_assessment_6251303/v1"
COHORT = "data/cohorts/paper120_v3/manifest.json"
SOURCE_COMMIT = "61f26e368f426be95978991caf656609e954158f"
COLLECTION_IDENTITY = "repaired-sbi-assessment-collection-v1-507d012c72151554bea319511e8a04d699382407abe6c1ee97bb8c4c15c19ec6"
JOB_ID = 6251303


def _verify_raw(root: Path) -> dict[str, Any]:
    raw = root / RAW
    workers = raw / "workers"
    pairs = []
    for path in sorted(workers.rglob("*.json")):
        sidecar = path.with_suffix(".sha256")
        if not sidecar.is_file():
            raise PaperContractError(f"missing assessment sidecar: {path.relative_to(raw)}")
        actual = file_hash(path)
        if sidecar.read_text(encoding="ascii").strip() != actual:
            raise PaperContractError(f"assessment hash mismatch: {path.relative_to(raw)}")
        pairs.append({"path": path.relative_to(raw).as_posix(), "sha256": actual})
    if len(pairs) != 962:
        raise PaperContractError("repaired SBI raw archive does not contain exactly 962 hash pairs")
    starts = tuple(workers.glob("worker-*/worker_start.json"))
    terminals = tuple(workers.glob("worker-*/worker_terminal.json"))
    attempts = tuple(workers.glob("worker-*/attempts/b*/target-*/*.json"))
    if (len(starts), len(terminals), len(attempts)) != (120, 120, 720):
        raise PaperContractError("repaired SBI worker archive coverage is incomplete")
    collection_path = workers / "collection/collection.json"
    compact_path = workers / "collection/compact_export.json"
    collection = read_json(collection_path)
    if (
        collection.get("status") != "completed"
        or collection.get("source_commit") != SOURCE_COMMIT
        or collection.get("collection_identity") != COLLECTION_IDENTITY
        or collection.get("record_count") != 720
        or collection.get("completed_record_count") != 720
        or collection.get("errors") != []
        or collection.get("physical_calls") != {
            "maximum_calls": 720, "attempted_calls": 720,
            "successful_calls": 720, "failed_calls": 0, "retry_calls": 0,
        }
    ):
        raise PaperContractError("repaired SBI collection is not a complete 720-call result")
    stdout = raw / f"logs/repaired-sbi-assess-{JOB_ID}.out"
    stderr = raw / f"logs/repaired-sbi-assess-{JOB_ID}.err"
    if not stdout.is_file() or not stderr.is_file() or stderr.stat().st_size != 0:
        raise PaperContractError("repaired SBI scheduler logs are missing or stderr is nonempty")
    return {
        "json_hash_pairs_verified": len(pairs),
        "worker_starts": len(starts),
        "worker_terminals": len(terminals),
        "attempt_records": len(attempts),
        "archive_inventory_identity": stable_hash("repaired-sbi-assessment-raw-inventory-v1", pairs),
        "collection_sha256": file_hash(collection_path),
        "compact_export_sha256": file_hash(compact_path),
        "scheduler_stdout_sha256": file_hash(stdout),
        "scheduler_stderr_sha256": file_hash(stderr),
    }


def _content(root: Path) -> dict[str, str]:
    integrity = _verify_raw(root)
    compact_path = root / RAW / "workers/collection/compact_export.json"
    compact = read_json(compact_path)
    cohort_path = resolve_recorded_path(root, COHORT)
    cohort = read_json(cohort_path)
    reports = {}
    rows_by_budget = {}
    for budget in (40, 100):
        rows = adapt_repaired_sbi_assessment_export(
            compact,
            source_identity=COLLECTION_IDENTITY,
            source_sha256=integrity["compact_export_sha256"],
            cohort=cohort,
            cohort_directory=cohort_path.parent,
            cohort_sha256=file_hash(cohort_path),
            budget=budget,
        )
        specification = analysis_specification(
            analysis_id=f"repaired-mutaco-sbi-b{budget}-v1",
            bootstrap_seed=92024000 + budget,
            bootstrap_resamples=10_000,
        )
        reports[budget] = deterministic_analysis(rows, specification)
        rows_by_budget[budget] = rows
    result_files = {}
    for budget in (40, 100):
        result_files[f"analysis_rows_b{budget}.jsonl"] = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows_by_budget[budget]
        )
        result_files[f"analysis_report_b{budget}.json"] = canonical_json_text(reports[budget])
    body = {
        "schema_version": "mutaco-repaired-sbi-assessment-evidence-v1",
        "status": "complete_collected_archived_and_analyzed",
        "role": "repaired_primary_mutaco_npe_snpe_paired_assessment",
        "job": {
            "job_id": JOB_ID, "scheduler_state": "COMPLETED", "elapsed": "00:00:44",
            "exit_code": "0:0", "scientific_step_elapsed": "00:00:32",
            "source_commit": SOURCE_COMMIT, "nodes": 30, "tasks": 120,
            "cpus_per_task": 5, "automatic_retry": False,
        },
        "call_accounting": {
            "maximum_calls": 720, "attempted_calls": 720, "successful_calls": 720,
            "failed_calls": 0, "retry_calls": 0, "analysis_calls": 0,
        },
        "collection_identity": COLLECTION_IDENTITY,
        "integrity": integrity,
        "cohort_sha256": file_hash(cohort_path),
        "budgets": [40, 100],
        "methods": ["mutaco", "sbi_npe", "sbi_snpe"],
        "reports": {
            str(budget): {
                "analysis_report_identity": reports[budget]["analysis_report_identity"],
                "target_count": reports[budget]["target_count"],
                "row_count": reports[budget]["row_count"],
                "heldout_component_mean_mae": {
                    method: reports[budget]["error_summaries_component_mean_mae"][method]["heldout_assessment_error"]
                    for method in reports[budget]["methods"]
                },
                "paired_primary_endpoint": reports[budget]["paired_primary_endpoint"],
                "target_winners": reports[budget]["target_winners"],
                "target_winner_ties": reports[budget]["target_winner_ties"],
                "mean_rank": reports[budget]["mean_rank"],
            }
            for budget in (40, 100)
        },
        "raw_archive": RAW,
        "remote_output_root": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/repaired_sbi_assessment/v1-61f26e3/workers",
        "remote_log_root": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/repaired_sbi_assessment/v1-61f26e3/logs",
        "historical_evidence_modified": False,
        "scientific_calls_added_by_archival_and_analysis": 0,
        "truba_jobs_added_by_archival_and_analysis": 0,
    }
    manifest = {**body, "evidence_identity": stable_hash("repaired-sbi-assessment-evidence-v1", body)}
    result_files["manifest.json"] = canonical_json_text(manifest)
    return result_files


def build(*, repository_root: Path, check: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    contents = _content(root)
    output = root / OUTPUT
    if check:
        for name, content in contents.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise PaperContractError(f"tracked repaired SBI artifact differs: {name}")
            sidecar = output / f"{name}.sha256"
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked repaired SBI sidecar differs: {name}")
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
        "status": status, "output": OUTPUT,
        "evidence_identity": manifest["evidence_identity"],
        "scientific_calls": 0, "truba_jobs_submitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(canonical_json_text(build(repository_root=args.repository_root, check=args.check)), end="")


if __name__ == "__main__":
    main()
