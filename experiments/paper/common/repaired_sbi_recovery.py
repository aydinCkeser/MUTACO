"""Compact evidence for B100 raw recovery and exact-state replay qualification."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import json
from pathlib import Path
from typing import Any, Mapping

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from .cross_experiment_contract import PAPER120_COHORT_SHA256
from .evidence_analysis import canonical_json_text
from .repaired_sbi_salvage import (
    B40_COMMIT,
    B100_POPULATION_COMMIT,
    B100_QUALIFICATION_COMMIT,
)
from .validation import PaperContractError


RECOVERY_SCHEMA = "mutaco-repaired-sbi-b100-raw-recovery-v1"
QUALIFICATION_SCHEMA = "mutaco-repaired-sbi-exact-state-qualification-v1"
PACKAGE_SCHEMA = "mutaco-repaired-sbi-recovery-qualification-package-v1"
OUTPUT = "results/paper/repaired_mutaco_sbi_recovery_qualification/v1"
B100_RECOVERY = (
    "experiments/paper/exploration/outputs/repaired_mutaco_sbi_salvage/"
    "b100_raw_recovery"
)
REPLAY_ROOT = (
    "experiments/paper/exploration/outputs/repaired_mutaco_sbi_salvage/"
    "exact_replay_qualification"
)
B100_COMPACT = "benchmarks/paper/graph_paper120/b100_v1/paired_records.jsonl"
QUALIFICATION_POSITIONS = (0, 49, 99)


def _cohort_ids(root: Path) -> tuple[str, ...]:
    path = root / "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
    if file_hash(path) != PAPER120_COHORT_SHA256:
        raise PaperContractError("recovery cohort hash mismatch")
    records = read_json(path)["records"]
    return tuple(str(records[str(position)]["target_record_id"]) for position in range(120))


def _compact_b100(root: Path) -> dict[int, Mapping[str, Any]]:
    rows = [
        json.loads(line)
        for line in (resolve_recorded_path(root, B100_COMPACT)).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 120:
        raise PaperContractError("B100 compact source is incomplete")
    return {int(row["target_position"]): row for row in rows}


def recover_b100_inventory(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    cohort_ids = _cohort_ids(root)
    compact = _compact_b100(root)
    recovery_root = root / B100_RECOVERY
    groups = {
        "qualification3": {
            "directory": recovery_root / "qualification",
            "job_id": 6214971,
            "source_commit": B100_QUALIFICATION_COMMIT,
            "remote_root": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/b100_qualification/2b067c33/workers",
            "expected_count": 3,
        },
        "remaining117": {
            "directory": recovery_root / "remaining117",
            "job_id": 6215336,
            "source_commit": "e7baa235fd7282ea478acc83f9457cfd95b76185",
            "remote_root": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/b100_remaining117/e7baa235/workers",
            "expected_count": 117,
        },
    }
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    total_bytes = 0
    for group_name, group in groups.items():
        paths = sorted(group["directory"].glob("worker-*/raw/target-*/mutaco.json"))
        if len(paths) != group["expected_count"]:
            raise PaperContractError(f"B100 recovered {group_name} count mismatch")
        for path in paths:
            digest = file_hash(path)
            sidecar = path.with_suffix(".sha256")
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
                raise PaperContractError(f"B100 recovered raw hash mismatch: {path}")
            raw = read_json(path)
            position = raw.get("target_index")
            if not isinstance(position, int) or position not in range(120) or position in seen:
                raise PaperContractError("B100 recovered target coverage is invalid")
            seen.add(position)
            if raw.get("target_record_id") != cohort_ids[position] or raw.get("status") != "completed":
                raise PaperContractError("B100 recovered identity/status mismatch")
            projection = raw.get("scientific_projection", {})
            observations = projection.get("observations", [])
            fits = projection.get("fits", [])
            if raw.get("inference_calls") != 100 or len(observations) != 100 or len(fits) != 85:
                raise PaperContractError("B100 recovered trajectory is incomplete")
            if raw.get("recommended_theta") != compact[position]["methods"]["mutaco"]["recommended_theta"]:
                raise PaperContractError("B100 recovered and compact recommendations differ")
            final_fit = fits[-1]
            context = (
                raw.get("runner_provenance", {})
                .get("configuration", {})
                .get("execution", {})
                .get("terminal_recommendation_context", {})
            )
            if context.get("target_id") != cohort_ids[position] or context.get("target_index") != position:
                raise PaperContractError("B100 recovered terminal context mismatch")
            size = path.stat().st_size
            total_bytes += size
            entries.append(
                {
                    "target_position": position,
                    "target_record_id": cohort_ids[position],
                    "archive_group": group_name,
                    "source_job_id": group["job_id"],
                    "source_commit": group["source_commit"],
                    "raw_sha256": digest,
                    "raw_bytes": size,
                    "observation_count": 100,
                    "fit_count": 85,
                    "final_training_data_id": final_fit["training_data_id"],
                    "final_model_id": final_fit["model_id"],
                }
            )
    entries.sort(key=lambda item: item["target_position"])
    if seen != set(range(120)):
        raise PaperContractError("B100 recovered target matrix is incomplete")
    if {item["target_position"] for item in entries if item["archive_group"] == "qualification3"} != set(QUALIFICATION_POSITIONS):
        raise PaperContractError("B100 qualification archive positions are unexpected")
    body = {
        "schema_version": RECOVERY_SCHEMA,
        "cohort_sha256": PAPER120_COHORT_SHA256,
        "coverage": {
            "targets": 120,
            "qualification3": 3,
            "remaining117": 117,
            "missing": 0,
            "raw_bytes": total_bytes,
            "sha256_mismatches": 0,
        },
        "sources": {
            name: {
                key: value
                for key, value in group.items()
                if key in {"job_id", "source_commit", "remote_root", "expected_count"}
            }
            for name, group in groups.items()
        },
        "entries": entries,
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    return {**body, "recovery_identity": stable_hash("repaired-sbi-b100-raw-recovery-v1", body)}


def exact_state_qualification(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    cohort_ids = _cohort_ids(root)
    rows: list[dict[str, Any]] = []
    for budget in (40, 100):
        for position in range(120):
            if budget == 40:
                commit = B40_COMMIT
                prefix = "qualified"
            elif position in QUALIFICATION_POSITIONS:
                commit = B100_QUALIFICATION_COMMIT
                prefix = "qualified"
            else:
                commit = B100_POPULATION_COMMIT
                prefix = "native-qualified"
            path = root / REPLAY_ROOT / f"{prefix}-b{budget}-target-{position:03d}.json"
            report = read_json(path)
            identity_body = dict(report)
            identity = identity_body.pop("replay_identity")
            identity_body.pop("elapsed_seconds")
            if stable_hash("exact-source-final-gp-replay-v1", identity_body) != identity:
                raise PaperContractError("exact-state replay identity mismatch")
            required = (
                "transform_identity_equal",
                "standardized_observations_bitwise_equal",
                "final_fit_configuration_equal",
                "final_training_data_identity_equal",
                "final_model_identity_equal",
                "final_parameter_state_loaded_from_archived_provenance",
            )
            if (
                report.get("source_commit") != commit
                or report.get("target_position") != position
                or report.get("target_record_id") != cohort_ids[position]
                or report.get("budget") != budget
                or not report.get("qualification_passed")
                or not all(report.get(key) is True for key in required)
                or report.get("optimizer_rerun") is not False
                or report.get("simulator_calls") != 0
                or report.get("terminal_recommendation_executed") is not False
            ):
                raise PaperContractError("exact-state replay qualification failed")
            rows.append(
                {
                    "budget": budget,
                    "target_position": position,
                    "target_record_id": cohort_ids[position],
                    "source_commit": commit,
                    "observation_count": report["observation_count"],
                    "fit_count": report["fit_count"],
                    "final_training_data_id": report["materialized_final_training_data_id"],
                    "final_model_id": report["materialized_final_model_id"],
                    "replay_identity": identity,
                    "required_equalities_passed": True,
                }
            )
    body = {
        "schema_version": QUALIFICATION_SCHEMA,
        "positions": list(range(120)),
        "qualification_positions": list(QUALIFICATION_POSITIONS),
        "budgets": [40, 100],
        "cases": rows,
        "cases_passed": 240,
        "cases_failed": 0,
        "method": "exact_source_model_construction_plus_archived_selected_parameter_state",
        "optimizer_rerun_is_not_an_identity_test": True,
        "terminal_recommendation_executed": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    return {**body, "qualification_identity": stable_hash("repaired-sbi-exact-state-qualification-v1", body)}


def _contents(root: Path) -> dict[str, str]:
    recovery = recover_b100_inventory(root)
    qualification = exact_state_qualification(root)
    recovery_text = canonical_json_text(recovery)
    qualification_text = canonical_json_text(qualification)
    body = {
        "schema_version": PACKAGE_SCHEMA,
        "recovery_identity": recovery["recovery_identity"],
        "qualification_identity": qualification["qualification_identity"],
        "recovery_content_identity": stable_hash("repaired-sbi-recovery-file-v1", recovery_text),
        "qualification_content_identity": stable_hash("repaired-sbi-qualification-file-v1", qualification_text),
        "b100_raw_trajectories_verified": 120,
        "exact_state_cases_passed": 240,
        "population_exact_state_replay_complete": True,
        "search_reruns_required": 0,
        "terminal_generation_authorized": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    manifest = {**body, "package_identity": stable_hash("repaired-sbi-recovery-qualification-package-v1", body)}
    return {
        "b100_raw_recovery.json": recovery_text,
        "exact_state_qualification.json": qualification_text,
        "manifest.json": canonical_json_text(manifest),
    }


def build(*, repository_root: Path, check: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    output = root / OUTPUT
    contents = _contents(root)
    if check:
        for name, expected in contents.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise PaperContractError(f"tracked recovery qualification differs: {name}")
            if (output / f"{name}.sha256").read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked recovery qualification sidecar differs: {name}")
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
