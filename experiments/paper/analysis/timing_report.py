"""Computation-free timing reports from validated paper records."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping


REPORT_SCHEMA = "mutaco-paper-persisted-timing-report-v2"
MAP_TIMING_TASK = "posterior_sample_map_scoring_selection"

_SBI_PHASES = (
    "inference_simulation",
    "neural_training",
    "posterior_construction",
    "posterior_sampling",
    "posterior_diagnostics_recommendation",
    "held_out_assessment",
)
_COMMON_PHASES = (
    "inference_simulation",
    "model_fitting_or_neural_training",
    "final_decision",
    "held_out_assessment",
    "method_total",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def _new_bucket() -> dict[str, float | int]:
    return {
        "wall_seconds": 0.0,
        "process_cpu_seconds": 0.0,
        "wall_records": 0,
        "process_cpu_records": 0,
    }


def _add(
    bucket: dict[str, float | int],
    *,
    wall: Any = None,
    cpu: Any = None,
) -> None:
    wall_value = _number(wall)
    cpu_value = _number(cpu)
    if wall_value is not None:
        bucket["wall_seconds"] = float(bucket["wall_seconds"]) + wall_value
        bucket["wall_records"] = int(bucket["wall_records"]) + 1
    if cpu_value is not None:
        bucket["process_cpu_seconds"] = (
            float(bucket["process_cpu_seconds"]) + cpu_value
        )
        bucket["process_cpu_records"] = int(bucket["process_cpu_records"]) + 1


def _bucket_document(
    bucket: Mapping[str, float | int], *, total_records: int
) -> dict[str, Any]:
    wall_records = int(bucket["wall_records"])
    cpu_records = int(bucket["process_cpu_records"])
    return {
        "wall_seconds": float(bucket["wall_seconds"]) if wall_records else None,
        "process_cpu_seconds": (
            float(bucket["process_cpu_seconds"]) if cpu_records else None
        ),
        "wall_records": wall_records,
        "process_cpu_records": cpu_records,
        "target_method_records": total_records,
        "wall_complete": wall_records == total_records,
        "process_cpu_complete": cpu_records == total_records,
    }


def _duration(value: Any) -> tuple[Any, Any]:
    if not isinstance(value, Mapping):
        return None, None
    return value.get("wall_seconds"), value.get("process_cpu_seconds")


def _map_event_duration(sbi: Mapping[str, Any]) -> tuple[float, float] | None:
    events = sbi.get("raw_events")
    if not isinstance(events, list):
        return None
    selected = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("task") == MAP_TIMING_TASK
    ]
    if not selected:
        return None
    wall_values = [_number(event.get("inclusive_seconds")) for event in selected]
    cpu_values = [
        _number(
            event.get(
                "process_cpu_inclusive_seconds", event.get("process_cpu_seconds")
            )
        )
        for event in selected
    ]
    if any(value is None for value in (*wall_values, *cpu_values)):
        return None
    return sum(value for value in wall_values if value is not None), sum(
        value for value in cpu_values if value is not None
    )


def summarize_persisted_timings(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize observational timings without importing scientific methods.

    Common phases intentionally stop at concepts that both methods own.
    Acquisition/proposal and posterior-specific work remain method-specific;
    unavailable clocks remain ``None`` rather than being invented as zero.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)
    methods: dict[str, Any] = {}
    for method, rows in sorted(grouped.items()):
        phase_wall = {phase: 0.0 for phase in _SBI_PHASES}
        method_wall = 0.0
        method_cpu = 0.0
        unattributed_wall = 0.0
        dispatch_overhead_wall = 0.0
        detailed = 0
        phase_detailed = 0
        completeness: dict[str, int] = defaultdict(int)
        common = {phase: _new_bucket() for phase in _COMMON_PHASES}
        specific: dict[str, dict[str, float | int]] = defaultdict(_new_bucket)

        for row in rows:
            timings = row.get("timings")
            if not isinstance(timings, Mapping):
                completeness["timing_unavailable"] += 1
                continue
            sbi = timings.get("sbi")
            if isinstance(sbi, Mapping):
                detailed += 1
                phase_detailed += 1
                completeness[str(sbi.get("completeness", "partial_unknown"))] += 1
                phases = sbi.get("phases", {})
                if isinstance(phases, Mapping):
                    for phase in _SBI_PHASES:
                        value = phases.get(phase)
                        wall, cpu = _duration(value)
                        wall_number = _number(wall)
                        if wall_number is not None:
                            phase_wall[phase] += wall_number
                        _add(specific[f"sbi_{phase}"], wall=wall, cpu=cpu)
                    for common_name, source in (
                        ("inference_simulation", "inference_simulation"),
                        ("model_fitting_or_neural_training", "neural_training"),
                        ("held_out_assessment", "held_out_assessment"),
                    ):
                        wall, cpu = _duration(phases.get(source))
                        _add(common[common_name], wall=wall, cpu=cpu)
                    map_duration = _map_event_duration(sbi)
                    if map_duration is not None:
                        map_wall, map_cpu = map_duration
                        _add(common["final_decision"], wall=map_wall, cpu=map_cpu)
                        _add(
                            specific["sbi_posterior_sample_map_scoring_selection"],
                            wall=map_wall,
                            cpu=map_cpu,
                        )
                    else:
                        wall, cpu = _duration(
                            phases.get("posterior_diagnostics_recommendation")
                        )
                        _add(common["final_decision"], wall=wall, cpu=cpu)
                method_total = sbi.get("method_total", {})
                wall, cpu = _duration(method_total)
                _add(common["method_total"], wall=wall, cpu=cpu)
                wall_number, cpu_number = _number(wall), _number(cpu)
                if wall_number is not None:
                    method_wall += wall_number
                if cpu_number is not None:
                    method_cpu += cpu_number
                unattributed = sbi.get("within_method_unattributed_overhead", {})
                unattributed_value = _number(
                    unattributed.get("wall_seconds")
                    if isinstance(unattributed, Mapping)
                    else None
                )
                if unattributed_value is not None:
                    unattributed_wall += unattributed_value
                dispatch = sbi.get("paper_dispatch_overhead", {})
                dispatch_value = _number(
                    dispatch.get("wall_seconds")
                    if isinstance(dispatch, Mapping)
                    else None
                )
                if dispatch_value is not None:
                    dispatch_overhead_wall += dispatch_value
                continue

            total = _number(timings.get("total"))
            flat_values = {
                "inference_simulation": timings.get("simulator"),
                "model_fitting_or_neural_training": timings.get("training_or_fitting"),
                "final_decision": timings.get("recommendation"),
                "held_out_assessment": timings.get("assessment"),
                "method_total": total,
            }
            has_flat_detail = any(
                _number(value) is not None
                for name, value in flat_values.items()
                if name != "method_total"
            )
            if has_flat_detail:
                phase_detailed += 1
                completeness["flat_phase_wall_only"] += 1
                for phase, value in flat_values.items():
                    _add(common[phase], wall=value)
                for name, key in (
                    ("mutaco_simulator", "simulator"),
                    ("mutaco_gp_fitting", "training_or_fitting"),
                    ("mutaco_proposal_acquisition", "proposal"),
                    ("mutaco_terminal_recommendation", "recommendation"),
                    ("mutaco_held_out_assessment", "assessment"),
                ):
                    _add(specific[name], wall=timings.get(key))
            elif total is not None:
                completeness["historical_total_only"] += 1
                _add(common["method_total"], wall=total)
            else:
                completeness["timing_unavailable"] += 1
            if total is not None:
                method_wall += total

        percentages = {
            phase: (100.0 * value / method_wall if method_wall > 0.0 else None)
            for phase, value in phase_wall.items()
        }
        methods[method] = {
            "target_method_records": len(rows),
            "detailed_timing_records": phase_detailed,
            "detailed_sbi_timing_records": detailed,
            "timing_completeness": dict(sorted(completeness.items())),
            "common_phase_times": {
                phase: _bucket_document(bucket, total_records=len(rows))
                for phase, bucket in common.items()
            },
            "method_specific_phase_times": {
                phase: _bucket_document(bucket, total_records=len(rows))
                for phase, bucket in sorted(specific.items())
            },
            # Backward-compatible projections retained under a new report schema.
            "phase_wall_seconds": phase_wall,
            "within_method_unattributed_wall_seconds": unattributed_wall,
            "paper_dispatch_overhead_wall_seconds": dispatch_overhead_wall,
            "method_total_wall_seconds": method_wall,
            "method_total_process_cpu_seconds": method_cpu if detailed else None,
            "phase_percent_of_method_total": percentages,
            "phase_decomposition_available": phase_detailed == len(rows),
        }
    return {
        "schema_version": REPORT_SCHEMA,
        "source": "validated_persisted_records_only",
        "common_phase_policy": (
            "shared_concepts_only; acquisition and posterior-specific work retained "
            "under method_specific_phase_times"
        ),
        "unknown_duration_policy": "unavailable_is_null_never_zero",
        "methods": methods,
    }

# Preserve historical pickle/global references while both imports share this module.
for _legacy_export in (
    _number,
    _new_bucket,
    _add,
    _bucket_document,
    _duration,
    _map_event_duration,
    summarize_persisted_timings,
):
    _legacy_export.__module__ = "experiments.paper.common.timing_report"
del _legacy_export
