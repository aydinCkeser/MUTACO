"""Validation and compact summaries for observational SBI timings."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from typing import Any, Mapping, Sequence


TIMING_SCHEMA = "mutaco-paper-sbi-timing-v1"
TIMING_SCOPE = "in_process_target_method"
ABSOLUTE_CLOCK_TOLERANCE_SECONDS = 1e-6
RELATIVE_CLOCK_TOLERANCE = 1e-9

_TASK_TO_PHASE = {
    "simulation": "inference_simulation",
    "inference_simulation": "inference_simulation",
    "neural_training": "neural_training",
    "posterior_construction": "posterior_construction",
    "posterior_sampling": "posterior_sampling",
    "posterior_diagnostics_recommendation": "posterior_diagnostics_recommendation",
    "posterior_sample_map_scoring_selection": "posterior_diagnostics_recommendation",
    "held_out_assessment": "held_out_assessment",
}


class TimingValidationError(ValueError):
    """Raised when an observational timing ledger is contradictory."""


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise TimingValidationError(f"{label} must be finite and nonnegative")
    return result


def _tolerance(*values: float) -> float:
    scale = max((abs(value) for value in values), default=0.0)
    return ABSOLUTE_CLOCK_TOLERANCE_SECONDS + RELATIVE_CLOCK_TOLERANCE * scale


def _duration(wall: float, cpu: float) -> dict[str, float]:
    return {"wall_seconds": wall, "process_cpu_seconds": cpu}


def summarize_sbi_timing(
    events: Sequence[Mapping[str, Any]],
    *,
    method_total_wall_seconds: float,
    method_total_process_cpu_seconds: float,
    status: str,
    failure_phase: str | None,
) -> dict[str, Any]:
    """Validate a ledger and summarize non-overlapping phase durations."""

    total_wall = _finite_nonnegative(
        method_total_wall_seconds, "method_total_wall_seconds"
    )
    total_cpu = _finite_nonnegative(
        method_total_process_cpu_seconds, "method_total_process_cpu_seconds"
    )
    rows = [dict(event) for event in events]
    by_id: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise TimingValidationError(f"timing event {position} lacks event_id")
        if event_id in by_id:
            raise TimingValidationError(f"duplicate timing event_id: {event_id}")
        by_id[event_id] = row
        task = row.get("task")
        if task not in _TASK_TO_PHASE:
            raise TimingValidationError(f"unknown timing task: {task!r}")
        if row.get("status") not in {"completed", "failed"}:
            raise TimingValidationError(f"timing event {event_id} has invalid status")
        _finite_nonnegative(row.get("inclusive_seconds"), f"{event_id}.inclusive")
        _finite_nonnegative(
            row.get("process_cpu_inclusive_seconds", row.get("process_cpu_seconds")),
            f"{event_id}.process_cpu_inclusive",
        )

    for event_id, row in by_id.items():
        parent = row.get("parent_event_id")
        if parent is not None and parent not in by_id:
            raise TimingValidationError(
                f"timing event {event_id} references missing parent {parent}"
            )
        seen = {event_id}
        cursor = parent
        while cursor is not None:
            if cursor in seen:
                raise TimingValidationError("timing event parent cycle")
            seen.add(cursor)
            cursor = by_id[cursor].get("parent_event_id")

    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("parent_event_id") is not None:
            children[str(row["parent_event_id"])].append(row)

    phase_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"wall_seconds": 0.0, "process_cpu_seconds": 0.0}
    )
    round_totals: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"wall_seconds": 0.0, "process_cpu_seconds": 0.0}
    )
    for row in rows:
        event_id = str(row["event_id"])
        inclusive_wall = float(row["inclusive_seconds"])
        inclusive_cpu = float(
            row.get("process_cpu_inclusive_seconds", row["process_cpu_seconds"])
        )
        child_wall = sum(float(child["inclusive_seconds"]) for child in children[event_id])
        child_cpu = sum(
            float(
                child.get(
                    "process_cpu_inclusive_seconds", child["process_cpu_seconds"]
                )
            )
            for child in children[event_id]
        )
        tolerance = _tolerance(inclusive_wall, inclusive_cpu, child_wall, child_cpu)
        if child_wall - inclusive_wall > tolerance:
            raise TimingValidationError(f"children exceed {event_id} wall duration")
        if child_cpu - inclusive_cpu > tolerance:
            raise TimingValidationError(f"children exceed {event_id} CPU duration")
        exclusive_wall = (
            0.0
            if abs(inclusive_wall - child_wall) <= tolerance
            else inclusive_wall - child_wall
        )
        exclusive_cpu = (
            0.0
            if abs(inclusive_cpu - child_cpu) <= tolerance
            else inclusive_cpu - child_cpu
        )
        phase = _TASK_TO_PHASE[str(row["task"])]
        phase_totals[phase]["wall_seconds"] += exclusive_wall
        phase_totals[phase]["process_cpu_seconds"] += exclusive_cpu
        round_index = row.get("round_index")
        if round_index is not None:
            if isinstance(round_index, bool) or not isinstance(round_index, int):
                raise TimingValidationError(f"{event_id}.round_index must be integer")
            round_totals[(phase, round_index)]["wall_seconds"] += exclusive_wall
            round_totals[(phase, round_index)]["process_cpu_seconds"] += exclusive_cpu

    measured_wall = sum(value["wall_seconds"] for value in phase_totals.values())
    measured_cpu = sum(value["process_cpu_seconds"] for value in phase_totals.values())
    tolerance = _tolerance(total_wall, total_cpu, measured_wall, measured_cpu)
    if measured_wall - total_wall > tolerance:
        raise TimingValidationError("measured phases exceed method wall total")
    if measured_cpu - total_cpu > tolerance:
        raise TimingValidationError("measured phases exceed method CPU total")
    residual_wall = total_wall - measured_wall
    residual_cpu = total_cpu - measured_cpu
    if abs(residual_wall) <= tolerance:
        residual_wall = 0.0
    if abs(residual_cpu) <= tolerance:
        residual_cpu = 0.0

    completeness = "exact_complete" if status == "completed" else "exact_partial_failure"
    return {
        "schema_version": TIMING_SCHEMA,
        "scope": TIMING_SCOPE,
        "clock_tolerance": {
            "absolute_seconds": ABSOLUTE_CLOCK_TOLERANCE_SECONDS,
            "relative": RELATIVE_CLOCK_TOLERANCE,
            "applied_seconds": tolerance,
        },
        "completeness": completeness,
        "termination_status": status,
        "failure_phase": failure_phase,
        "raw_events": deepcopy(rows),
        "phases": dict(sorted(phase_totals.items())),
        "rounds": [
            {"phase": phase, "round_index": round_index, **durations}
            for (phase, round_index), durations in sorted(round_totals.items())
        ],
        "within_method_unattributed_overhead": _duration(
            residual_wall, residual_cpu
        ),
        "method_total": _duration(total_wall, total_cpu),
        "reconciliation": {
            "measured_phase_wall_seconds": measured_wall,
            "measured_phase_process_cpu_seconds": measured_cpu,
            "wall_residual_seconds": residual_wall,
            "process_cpu_residual_seconds": residual_cpu,
        },
    }


def attach_paper_dispatch_timing(
    summary: Mapping[str, Any],
    *,
    paper_dispatch_wall_seconds: float,
    paper_dispatch_process_cpu_seconds: float,
) -> dict[str, Any]:
    """Add the outer in-process paper-dispatch scope to an SBI summary."""

    result = deepcopy(dict(summary))
    method = result.get("method_total")
    if not isinstance(method, Mapping):
        raise TimingValidationError("SBI timing summary lacks method_total")
    dispatch_wall = _finite_nonnegative(
        paper_dispatch_wall_seconds, "paper_dispatch_wall_seconds"
    )
    dispatch_cpu = _finite_nonnegative(
        paper_dispatch_process_cpu_seconds, "paper_dispatch_process_cpu_seconds"
    )
    method_wall = _finite_nonnegative(method.get("wall_seconds"), "method wall total")
    method_cpu = _finite_nonnegative(
        method.get("process_cpu_seconds"), "method CPU total"
    )
    tolerance = _tolerance(dispatch_wall, dispatch_cpu, method_wall, method_cpu)
    if method_wall - dispatch_wall > tolerance or method_cpu - dispatch_cpu > tolerance:
        raise TimingValidationError("method timing exceeds paper-dispatch timing")
    overhead_wall = dispatch_wall - method_wall
    overhead_cpu = dispatch_cpu - method_cpu
    if abs(overhead_wall) <= tolerance:
        overhead_wall = 0.0
    if abs(overhead_cpu) <= tolerance:
        overhead_cpu = 0.0
    result["paper_dispatch_overhead"] = _duration(overhead_wall, overhead_cpu)
    result["paper_dispatch_total"] = _duration(dispatch_wall, dispatch_cpu)
    result["paper_dispatch_scope"] = (
        "paper_target_method_dispatch_through_record_materialization_excluding_record_file_io"
    )
    return result
