"""Target-only preview, capped construction, recovery, and collection.

This module intentionally has no dependency on paper method registries or
method runners.  Callers provide a target evaluator explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mutaco import freeze_strict_metadata, stable_hash, strict_metadata_document

from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json

from .cohort_v2 import validate_cohort_declaration
from .validation import PaperContractError


TARGET_RECORD_SCHEMA = "mutaco-frozen-target-record-v2"
TARGET_ATTEMPT_SCHEMA = "mutaco-target-freeze-attempt-v1"


def _record_path(root: Path, index: int) -> Path:
    return root / "target_ledger" / f"target_{index:03d}.json"


def _write_hashed(path: Path, document: Mapping[str, Any]) -> None:
    write_pretty_json(path, document)
    path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


def _load_hashed(path: Path) -> dict[str, Any] | None:
    sidecar = path.with_suffix(".sha256")
    if not path.exists() and not sidecar.exists():
        return None
    if not path.is_file() or not sidecar.is_file():
        raise PaperContractError(f"incomplete frozen target record: {path}")
    if file_hash(path) != sidecar.read_text(encoding="ascii").strip():
        raise PaperContractError(f"frozen target hash mismatch: {path}")
    record = read_json(path)
    content = {key: value for key, value in record.items() if key not in {"content_hash", "target_record_id"}}
    content_hash = stable_hash("frozen-target-content-v2", content)
    record_id = stable_hash("frozen-target-record-v2", {**content, "content_hash": content_hash})
    if record.get("content_hash") != content_hash or record.get("target_record_id") != record_id:
        raise PaperContractError(f"frozen target content identity mismatch: {path}")
    return record


def _selection(document: Mapping[str, Any], indices: Sequence[int] | None) -> tuple[int, ...]:
    count = int(document["target_count"])
    resolved = tuple(range(count)) if indices is None else tuple(indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in resolved):
        raise PaperContractError("target indices must be integers")
    if len(set(resolved)) != len(resolved) or any(index < 0 or index >= count for index in resolved):
        raise PaperContractError("target indices must be unique and in range")
    return resolved


@dataclass(frozen=True)
class TargetFreezePreview:
    target_indices: tuple[int, ...]
    reusable: tuple[int, ...]
    missing: tuple[int, ...]
    maximum_calls: int

    @property
    def planned_calls(self) -> int:
        return len(self.missing)

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": "mutaco-target-freeze-preview-v1",
            "scientific_calls_executed": 0,
            "target_indices": list(self.target_indices),
            "reusable_target_indices": list(self.reusable),
            "missing_target_indices": list(self.missing),
            "planned_target_construction_calls": self.planned_calls,
            "maximum_target_construction_calls": self.maximum_calls,
            "within_cap": self.planned_calls <= self.maximum_calls,
        }


def preview_target_freeze(
    declaration: Mapping[str, Any], output: Path, *, maximum_calls: int,
    target_indices: Sequence[int] | None = None,
) -> TargetFreezePreview:
    validated = validate_cohort_declaration(declaration)
    if isinstance(maximum_calls, bool) or not isinstance(maximum_calls, int) or maximum_calls < 0:
        raise PaperContractError("maximum_calls must be a nonnegative integer")
    selected = _selection(validated, target_indices)
    reusable, missing = [], []
    by_index = {record["target_index"]: record for record in validated["records"]}
    for index in selected:
        record = _load_hashed(_record_path(output, index))
        if record is None:
            missing.append(index)
            continue
        expected = by_index[index]
        if record.get("declaration_hash") != validated["declaration_hash"] or record.get("target_identity") != expected["target_identity"]:
            raise PaperContractError(f"frozen target declaration identity mismatch: {index}")
        for field in (
            "target_index", "sobol_index", "theta_star", "target_seed_coordinate",
            "target_generation_seed",
        ):
            if record.get(field) != expected.get(field):
                raise PaperContractError(f"frozen target theta/seed mismatch: {index}")
        reusable.append(index)
    return TargetFreezePreview(selected, tuple(reusable), tuple(missing), maximum_calls)


def execute_target_freeze(
    declaration: Mapping[str, Any], output: Path, *, maximum_calls: int,
    evaluate_target: Callable[[Mapping[str, Any]], tuple[Sequence[float], Mapping[str, Any]]],
    execution_provenance: Mapping[str, Any],
    target_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Create missing target responses under a fail-before-first-call cap."""

    validated = validate_cohort_declaration(declaration)
    preview = preview_target_freeze(
        validated, output, maximum_calls=maximum_calls, target_indices=target_indices
    )
    if preview.planned_calls > maximum_calls:
        raise PaperContractError(
            f"target construction requires {preview.planned_calls} calls, cap is {maximum_calls}"
        )
    if not execution_provenance:
        raise PaperContractError("target execution provenance must be nonempty")
    attempts = output / "target_attempts"
    attempt_index = len(tuple(attempts.glob("attempt_*.json"))) if attempts.exists() else 0
    attempt_path = attempts / f"attempt_{attempt_index:04d}.json"
    if attempt_path.exists():
        raise PaperContractError("append-only target attempt identity collision")
    attempt: dict[str, Any] = {
        "schema_version": TARGET_ATTEMPT_SCHEMA,
        "attempt_index": attempt_index,
        "declaration_hash": validated["declaration_hash"],
        "selected_indices": list(preview.target_indices),
        "planned_calls": preview.planned_calls,
        "maximum_calls": maximum_calls,
        "execution_provenance": dict(execution_provenance),
        "status": "running",
        "calls_attempted": 0,
        "calls_successful": 0,
        "calls_failed": 0,
        "completed_indices": [],
        "failed_indices": [],
    }
    _write_hashed(attempt_path, attempt)
    by_index = {record["target_index"]: record for record in validated["records"]}
    try:
        for index in preview.missing:
            attempt["calls_attempted"] += 1
            theta_record = by_index[index]
            failure_phase = "evaluation"
            try:
                response, metadata = evaluate_target(theta_record)
                values = [float(value) for value in response]
                if not values or not all(__import__("math").isfinite(value) for value in values):
                    raise PaperContractError("target response must contain finite values")
                failure_phase = "persistence"
                metadata_document = strict_metadata_document(
                    freeze_strict_metadata(
                        metadata, label="simulator_result_metadata"
                    )
                )
                content = {
                    "schema_version": TARGET_RECORD_SCHEMA,
                    "declaration_hash": validated["declaration_hash"],
                    "target_family_id": validated["target_family_id"],
                    "simulator_adapter": validated["simulator_adapter"],
                    "simulator_definition_id": validated["simulator_definition_id"],
                    "simulator_configuration_identity": validated["simulator_configuration_identity"],
                    "simulator_configuration_sha256": validated["simulator_configuration_sha256"],
                    "parameter_profile": validated["parameter_profile"],
                    "target_index": index,
                    "sobol_index": theta_record["sobol_index"],
                    "target_identity": theta_record["target_identity"],
                    "theta_star": theta_record["theta_star"],
                    "target_seed_coordinate": theta_record["target_seed_coordinate"],
                    "target_generation_seed": theta_record["target_generation_seed"],
                    "target_response": values,
                    "simulator_result_metadata": metadata_document,
                    "execution_provenance": dict(execution_provenance),
                }
                content_hash = stable_hash("frozen-target-content-v2", content)
                record = {
                    **content,
                    "content_hash": content_hash,
                    "target_record_id": stable_hash(
                        "frozen-target-record-v2", {**content, "content_hash": content_hash}
                    ),
                }
                record_path = _record_path(output, index)
                _write_hashed(record_path, record)
                reloaded = _load_hashed(record_path)
                if reloaded != record:
                    raise PaperContractError(
                        f"frozen target record failed exact persisted reload: {index}"
                    )
            except Exception as exc:
                attempt["calls_failed"] += 1
                attempt["failed_indices"].append({
                    "target_index": index,
                    "failure_phase": failure_phase,
                    "failure_type": type(exc).__name__,
                    "failure_explanation": str(exc),
                })
                raise
            attempt["calls_successful"] += 1
            attempt["completed_indices"].append(index)
    except Exception:
        attempt["status"] = "failed_preserved_no_automatic_retry"
        _write_hashed(attempt_path, attempt)
        raise
    attempt["status"] = "completed"
    _write_hashed(attempt_path, attempt)
    manifest = collect_target_freeze(validated, output, selected=preview.target_indices)
    return {"preview": preview.document(), "attempt": attempt, "manifest": manifest}


def collect_target_freeze(
    declaration: Mapping[str, Any], output: Path, *, selected: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate and summarize a target ledger without executing science."""

    validated = validate_cohort_declaration(declaration)
    indices = _selection(validated, selected)
    completed, missing = {}, []
    for index in indices:
        record = _load_hashed(_record_path(output, index))
        if record is None:
            missing.append(index)
        else:
            completed[str(index)] = {
                "target_record_id": record["target_record_id"],
                "content_hash": record["content_hash"],
            }
    manifest = {
        "schema_version": "mutaco-target-freeze-manifest-v2",
        "declaration_hash": validated["declaration_hash"],
        "selected_indices": list(indices),
        "completed_count": len(completed),
        "missing_count": len(missing),
        "missing_indices": missing,
        "records": completed,
        "scientific_calls_executed_by_collection": 0,
    }
    _write_hashed(output / "target_ledger" / "manifest.json", manifest)
    return manifest

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.target_freeze_v2")
