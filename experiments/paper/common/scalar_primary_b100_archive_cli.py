"""Archive and deterministically analyze scalar-primary B100 jobs 6251851/6251852."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.evidence_analysis import (
    SPEC_IDENTITY_DOMAIN,
    adapt_scalar_primary_export,
    canonical_json_text,
    deterministic_analysis,
)
from experiments.paper.common.validation import PaperContractError
from experiments.paper.historical_workflows.scalar_primary_b100_preparation_audit_cli import (
    PROFILE, verified_historical_profile,
)


RAW = "experiments/paper/exploration/outputs/scalar_primary_smoke/population_b100_v3r2_compact_fetch"
OUTPUT = "results/paper/graph_scalar_primary_b100_v1"
PREPARATION = "benchmarks/paper/graph_scalar_primary_b100_population_preparation_v3r2"
COHORT = "data/cohorts/paper120_v3/manifest.json"
DECLARATION = "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_v3r2.json"
PLAN = "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_plan_v3r2.json"
SOURCE_COMMIT = "649235bcb7e8773f5f4b0425ad111314d7d3b1f9"
COLLECTION_IDENTITY = "scalar-primary-global-collection-v1-03cf838d3651496e4544a4627396589cc4071e0cb2a23a53288cf2057d478685"
SCHEDULER_INVENTORY_SHA256 = "84b0955e74a12538846c2d17dc521503045a56fd094dd2d6efdd4b2af29ad924"
REMOTE_WORKER_INVENTORY = {
    "schema_version": "mutaco-json-hash-pair-inventory-v1",
    "json_files": 842,
    "sha256_sidecars": 842,
    "hash_mismatches": [],
    "missing_sidecars": [],
    "orphan_sidecars": [],
    "json_bytes": 1_283_337_279,
    "categories": {
        "collection": 2,
        "decision_freeze": 120,
        "mutaco": 120,
        "scalar_qlognei": 120,
        "sobol": 120,
        "target_execution": 120,
        "worker_start": 120,
        "worker_terminal": 120,
    },
    "inventory_sha256": "27c65c91de858845d193d2c12ad4b779061ffc7a18a71edff6bdbc6a4e43d634",
    "scientific_calls": 0,
}


def _assert_hash_pair(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    actual = file_hash(path)
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != actual:
        raise PaperContractError(f"hash pair differs: {path}")
    return actual


def _log_inventory(logs: Path) -> dict[str, Any]:
    paths = tuple(sorted(path for path in logs.iterdir() if path.is_file()))
    if len(paths) != 122:
        raise PaperContractError("B100 scheduler log inventory is incomplete")
    entries = [
        {
            "path": path.name,
            "sha256": file_hash(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    failure_markers = (
        "Traceback (most recent call last):",
        "slurmstepd: error",
        "Out Of Memory",
        "CANCELLED AT",
        "Killed\n",
    )
    flagged = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in failure_markers:
            if marker in text:
                flagged.append({"path": path.name, "marker": marker})
    if flagged:
        raise PaperContractError(f"B100 scheduler logs contain failure markers: {flagged}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "files": len(paths),
        "nonempty_files": sum(item["size_bytes"] > 0 for item in entries),
        "bytes": sum(item["size_bytes"] for item in entries),
        "inventory_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "failure_markers": [],
    }


def _qlognei_diagnostics(compact: Mapping[str, Any]) -> dict[str, Any]:
    records = [row for row in compact["records"] if row["method"] == "scalar_qlognei"]
    trajectories = [item for row in records for item in row["decision"]["trajectory"]]
    adaptive = [item["acquisition_diagnostics"] for item in trajectories if item.get("acquisition_diagnostics") is not None]
    warnings = [warning for item in adaptive for warning in item.get("warnings", [])]
    warnings.extend(
        warning
        for row in records
        for warning in row["decision"]["terminal_metadata"].get("warnings", [])
    )
    direct_jitter_values = []
    retry_jitter_values = []
    for warning in warnings:
        values = [
            float(match)
            for match in re.findall(
                r"jitter of ([0-9.]+e[+-][0-9]+)",
                warning.get("message", ""),
                flags=re.IGNORECASE,
            )
        ]
        if warning.get("category") == "NumericalWarning":
            direct_jitter_values.extend(values)
        elif warning.get("category") == "RuntimeWarning":
            retry_jitter_values.extend(values)
    sources = Counter(item.get("candidate_source") for item in adaptive)
    duplicate_trajectories = sum(
        len({tuple(item["theta"]) for item in row["decision"]["trajectory"]}) != 100
        for row in records
    )
    statuses = Counter(
        item.get("gp_fit_status")
        for row in records
        for item in row["decision"]["trajectory"]
        if item.get("gp_fit_status") is not None
    )
    return {
        "targets": len(records),
        "adaptive_proposals": len(adaptive),
        "candidate_sources": dict(sorted((str(key), value) for key, value in sources.items())),
        "successful_gp_fits": statuses.get("success", 0),
        "precision_fallbacks": sum(bool(item.get("precision_fallback_used")) for item in adaptive),
        "terminal_precision_fallbacks": sum(
            bool(row["decision"]["terminal_metadata"].get("precision_fallback_used"))
            for row in records
        ),
        "warning_events": len(warnings),
        "warning_categories": dict(sorted(Counter(warning["category"] for warning in warnings).items())),
        "direct_numerical_jitter_events": sum(
            warning.get("category") == "NumericalWarning" for warning in warnings
        ),
        "maximum_direct_numerical_jitter": max(direct_jitter_values, default=0.0),
        "maximum_jitter_mentioned_in_runtime_retry_warnings": max(
            retry_jitter_values, default=0.0
        ),
        "duplicate_search_trajectories": duplicate_trajectories,
        "unrecovered_algorithm_failures": sum(row["decision"]["status"] != "recommended" for row in records),
    }


def _summary_projection(report: Mapping[str, Any], error: str) -> dict[str, Any]:
    return {
        method: report["error_summaries_component_mean_mae"][method][error]
        for method in report["methods"]
    }


def _archived_profile_sha256(root: Path, preparation: Mapping[str, Any]) -> str:
    """Tie the original jobs' profile bytes to both the source receipt and preparation."""
    original = verified_historical_profile(root, (root / PROFILE).read_bytes())
    digest = hashlib.sha256(original).hexdigest()
    if preparation["profile"] != PROFILE or preparation["profile_sha256"] != digest:
        raise PaperContractError("B100 archive scheduler profile differs from the preparation freeze")
    return digest


def _content(root: Path) -> dict[str, str]:
    raw = root / RAW
    collection_path = raw / "population/collection/collection.json"
    compact_path = raw / "population/collection/compact_export.json"
    export_manifest_path = raw / "export_manifest.json"
    archive_path = raw / "population.tar.gz"
    logs = raw / "logs"
    collection_sha = _assert_hash_pair(collection_path)
    compact_sha = _assert_hash_pair(compact_path)
    export_manifest = read_json(export_manifest_path)
    if (
        export_manifest.get("archive_sha256") != file_hash(archive_path)
        or export_manifest.get("scientific_calls") != 0
        or export_manifest.get("raw_posterior_artifacts_included") is not False
    ):
        raise PaperContractError("B100 compact transport package is invalid")
    exported = {item["path"]: item for item in export_manifest["files"]}
    if exported["collection/collection.json"]["sha256"] != collection_sha or exported["collection/compact_export.json"]["sha256"] != compact_sha:
        raise PaperContractError("B100 transport manifest differs from extracted collection")
    collection = read_json(collection_path)
    expected_calls = {
        "attempted_calls": 36_360,
        "failed_calls": 0,
        "maximum_calls": 36_360,
        "successful_calls": 36_360,
    }
    if (
        collection.get("status") != "completed"
        or collection.get("source_commit") != SOURCE_COMMIT
        or collection.get("collection_identity") != COLLECTION_IDENTITY
        or collection.get("record_count") != 360
        or collection.get("errors") != []
        or collection.get("physical_calls") != expected_calls
    ):
        raise PaperContractError("B100 collection is not a complete 36,360-call result")
    preparation = read_json(resolve_recorded_path(root, PREPARATION) / "manifest.json")
    profile_sha256 = _archived_profile_sha256(root, preparation)
    specification = read_json(resolve_recorded_path(root, PREPARATION) / "analysis_spec.json")
    spec_body = dict(specification)
    spec_identity = spec_body.pop("analysis_spec_identity", None)
    if (
        spec_identity != stable_hash(SPEC_IDENTITY_DOMAIN, spec_body)
        or spec_identity != preparation["analysis"]["analysis_spec_identity"]
        or specification["bootstrap"] != {
            "algorithm": "python_random_index_resampling_v1",
            "lower_percentile": 0.025,
            "resamples": 10_000,
            "seed": 92_024_100,
            "upper_percentile": 0.975,
        }
    ):
        raise PaperContractError("B100 analysis specification differs from the preparation freeze")
    cohort_path = resolve_recorded_path(root, COHORT)
    compact = read_json(compact_path)
    rows = adapt_scalar_primary_export(
        compact,
        source_identity=COLLECTION_IDENTITY,
        source_sha256=compact_sha,
        cohort=read_json(cohort_path),
        cohort_directory=cohort_path.parent,
        cohort_sha256=file_hash(cohort_path),
    )
    report = deterministic_analysis(rows, specification)
    if report["target_count"] != 120 or report["row_count"] != 360 or report["budgets"] != [100]:
        raise PaperContractError("B100 deterministic analysis grid is incomplete")
    log_inventory = _log_inventory(logs)
    integrity = {
        "remote_worker_hash_inventory": REMOTE_WORKER_INVENTORY,
        "scheduler_inventory_records": 61,
        "scheduler_inventory_sha256": SCHEDULER_INVENTORY_SHA256,
        "scheduler_logs": log_inventory,
        "collection_sha256": collection_sha,
        "compact_export_sha256": compact_sha,
        "compact_archive_sha256": file_hash(archive_path),
        "compact_archive_bytes": archive_path.stat().st_size,
        "compact_export_bytes": compact_path.stat().st_size,
        "compact_export_tracked_in_git": False,
        "remote_raw_evidence_mutated": False,
    }
    diagnostics = _qlognei_diagnostics(compact)
    primary = _summary_projection(report, "heldout_assessment_error")
    largest = max(
        (row for row in rows if row["method"] == "scalar_qlognei"),
        key=lambda row: row["errors"]["heldout_assessment_error"]["metric"]["component_mean_mae"],
    )
    body = {
        "schema_version": "mutaco-scalar-primary-population-evidence-v1",
        "status": "complete_collected_archived_and_analyzed",
        "study_id": "graph_paper120_mutaco_scalar_qlognei_sobol_primary_v1",
        "budget": 100,
        "targets": 120,
        "methods": ["mutaco", "scalar_qlognei", "sobol"],
        "scientific_source": {
            "commit": SOURCE_COMMIT,
            "declaration": DECLARATION,
            "declaration_sha256": file_hash(root / DECLARATION),
            "declaration_identity": preparation["declaration_identity"],
            "plan": PLAN,
            "plan_sha256": file_hash(root / PLAN),
            "plan_identity": preparation["plan_identity"],
            "profile": PROFILE,
            "profile_sha256": profile_sha256,
            "source_root": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b100-v3r2/source-exact",
            "output_root": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b100-v3r2/workers",
            "log_root": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b100-v3r2/logs",
        },
        "scheduler": {
            "array_job_id": 6_251_851,
            "collector_job_id": 6_251_852,
            "array_state": "COMPLETED",
            "collector_state": "COMPLETED",
            "physical_array_tasks": 60,
            "completed_physical_array_tasks": 60,
            "logical_workers": 120,
            "completed_logical_workers": 120,
            "failed_logical_workers": 0,
            "campaign_wall_interval": "00:44:05",
            "longest_worker_elapsed": "00:23:16",
            "collector_elapsed": "00:00:23",
            "automatic_retry": False,
            "slurm_requeue": False,
        },
        "call_accounting": {
            "attempted": 36_360,
            "successful": 36_360,
            "failed": 0,
            "maximum_authorized": 36_360,
            "calls_added_by_archival_and_analysis": 0,
        },
        "coverage": {
            "completed_targets": 120,
            "method_records": len(compact["records"]),
            "endpoint_results": len(compact["endpoint_results"]),
            "trajectory_results": len(compact["trajectory_results"]),
            "paired_endpoint_differences": len(compact["paired_endpoint_differences"]),
            "runtime_results": len(compact["runtime_results"]),
            "worker_start_records": REMOTE_WORKER_INVENTORY["categories"]["worker_start"],
            "worker_terminal_records": REMOTE_WORKER_INVENTORY["categories"]["worker_terminal"],
        },
        "integrity": integrity,
        "analysis_spec_identity": spec_identity,
        "analysis_report_identity": report["analysis_report_identity"],
        "primary_endpoint_heldout_physical_response_component_mean_mae": primary,
        "paired_primary_endpoint": report["paired_primary_endpoint"],
        "target_winners": report["target_winners"],
        "target_winner_ties": report["target_winner_ties"],
        "mean_rank": report["mean_rank"],
        "secondary_noisy_best_observed_search_component_mean_mae_at_call_100": _summary_projection(report, "observed_search_error"),
        "terminal_surrogate_error_component_mean_mae": {
            "raw": _summary_projection(report, "raw_terminal_surrogate_error"),
            "optimized": _summary_projection(report, "optimized_terminal_surrogate_error"),
        },
        "descriptive_decision_timing_seconds": {
            "claim_scope": "descriptive symmetric persisted component timings only; no production compute-efficiency claim",
            "summaries": report["descriptive_timing_seconds"],
        },
        "scalar_qlognei_diagnostics": diagnostics,
        "audit_note": {
            "largest_scalar_qlognei_heldout_target_identity": largest["target_id"],
            "largest_scalar_qlognei_heldout_component_mean_mae": largest["errors"]["heldout_assessment_error"]["metric"]["component_mean_mae"],
            "interpretation": "Retained without exclusion; medians, paired intervals, wins, and ranks accompany means.",
        },
        "historical_attempt_policy": {
            "previous_attempts_preserved": True,
            "previous_attempts_pooled_into_this_population": False,
            "this_population_complete_without_retry": True,
        },
        "historical_evidence_modified": False,
        "scientific_calls_added_by_archival_and_analysis": 0,
        "truba_jobs_added_by_archival_and_analysis": 0,
    }
    manifest = {**body, "evidence_identity": stable_hash("scalar-primary-population-evidence-v1", body)}
    rows_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    return {
        "analysis_rows.jsonl": rows_text,
        "analysis_report.json": canonical_json_text(report),
        "integrity_audit.json": canonical_json_text(integrity),
        "manifest.json": canonical_json_text(manifest),
    }


def build(*, repository_root: Path, check: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    contents = _content(root)
    output = root / OUTPUT
    if check:
        for name, content in contents.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise PaperContractError(f"tracked B100 artifact differs: {name}")
            sidecar = output / f"{name}.sha256"
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked B100 sidecar differs: {name}")
        status = "reproduced"
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in contents.items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            (output / f"{name}.sha256").write_text(file_hash(path) + "\n", encoding="ascii", newline="\n")
        status = "materialized"
    manifest = json.loads(contents["manifest.json"])
    return {
        "status": status,
        "output": OUTPUT,
        "evidence_identity": manifest["evidence_identity"],
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(canonical_json_text(build(repository_root=args.repository_root, check=args.check)), end="")


if __name__ == "__main__":
    main()
