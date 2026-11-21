"""Capped append-only execution for repaired MUTACO--SBI assessment tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Callable, Mapping, Sequence

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json

from experiments.paper.protocol.protocol_v3 import AssessmentEvaluationV3
from experiments.paper.protocol.validation import PaperContractError


WORKER_START_SCHEMA = "mutaco-repaired-sbi-assessment-worker-start-v1"
ATTEMPT_SCHEMA = "mutaco-repaired-sbi-assessment-attempt-v1"
WORKER_TERMINAL_SCHEMA = "mutaco-repaired-sbi-assessment-worker-terminal-v1"
RESULT_DOMAIN = "repaired-sbi-assessment-result-v1"
WORKERS = 120
CALLS_PER_WORKER = 6


def _append_hashed(path: Path, document: Mapping[str, Any]) -> None:
    sidecar = path.with_suffix(".sha256")
    if path.exists() or sidecar.exists():
        raise PaperContractError(f"append-only assessment evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pretty_json(path, dict(document))
    sidecar.write_text(file_hash(path) + "\n", encoding="ascii", newline="\n")


def load_hashed(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise PaperContractError(f"assessment hash pair is missing: {path}")
    if sidecar.read_text(encoding="ascii").strip() != file_hash(path):
        raise PaperContractError(f"assessment hash mismatch: {path}")
    return read_json(path)


def shard_tasks(
    tasks: Sequence[Mapping[str, Any]], *, rank: int, workers: int
) -> tuple[Mapping[str, Any], ...]:
    if workers != WORKERS or not 0 <= rank < workers:
        raise PaperContractError("repaired SBI assessment requires exactly 120 workers")
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for task in tasks:
        key = (int(task["budget"]), int(task["target_position"]))
        groups.setdefault(key, []).append(task)
    ordered = [(budget, position) for budget in (40, 100) for position in range(120)]
    if set(groups) != set(ordered) or any(
        tuple(item["method"] for item in groups[key]) != ("mutaco", "sbi_npe", "sbi_snpe")
        for key in ordered
    ):
        raise PaperContractError("assessment tasks do not form complete paired groups")
    selected = [key for index, key in enumerate(ordered) if index % workers == rank]
    return tuple(task for key in selected for task in groups[key])


def worker_preview(
    tasks: Sequence[Mapping[str, Any]], *, rank: int, workers: int, maximum_calls: int
) -> dict[str, Any]:
    selected = shard_tasks(tasks, rank=rank, workers=workers)
    if maximum_calls != CALLS_PER_WORKER or len(selected) != maximum_calls:
        raise PaperContractError("assessment worker cap must exactly equal its six tasks")
    groups = [(int(selected[index]["budget"]), int(selected[index]["target_position"])) for index in (0, 3)]
    return {
        "schema_version": "mutaco-repaired-sbi-assessment-worker-preview-v1",
        "rank": rank,
        "workers": workers,
        "target_budget_groups": groups,
        "methods_per_group": 3,
        "task_count": len(selected),
        "maximum_calls": maximum_calls,
        "call_preview": {
            "maximum_assessment_calls": maximum_calls,
            "scientific_calls_executed": 0,
        },
        "job_submitted": False,
        "scientific_calls_executed": 0,
    }


@dataclass
class _CallLedger:
    maximum_calls: int
    attempted_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

    def begin(self) -> None:
        if self.attempted_calls >= self.maximum_calls:
            raise PaperContractError("assessment worker physical-call cap exceeded")
        self.attempted_calls += 1

    def success(self) -> None:
        self.successful_calls += 1

    def failure(self) -> None:
        self.failed_calls += 1

    def document(self) -> dict[str, int]:
        return {
            "maximum_calls": self.maximum_calls,
            "attempted_calls": self.attempted_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "retry_calls": 0,
        }


def _finite_vector(value: Sequence[float], *, length: int, field: str) -> list[float]:
    result = [float(item) for item in value]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise PaperContractError(f"{field} must contain {length} finite values")
    return result


def _attempt_path(output: Path, task: Mapping[str, Any]) -> Path:
    return (
        output
        / "attempts"
        / f"b{int(task['budget'])}"
        / f"target-{int(task['target_position']):03d}"
        / f"{task['method']}.json"
    )


def execute_worker(
    *,
    tasks: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
    output: Path,
    rank: int,
    workers: int,
    maximum_calls: int,
    source_commit: str,
    declaration_identity: str,
    package_identity: str,
    evaluate: Callable[[Sequence[float], Mapping[str, Any], Mapping[str, Any]], AssessmentEvaluationV3 | Sequence[float]],
) -> dict[str, Any]:
    preview = worker_preview(tasks, rank=rank, workers=workers, maximum_calls=maximum_calls)
    if output.exists():
        raise PaperContractError("assessment worker output must be append-only")
    _append_hashed(output / "worker_start.json", {
        "schema_version": WORKER_START_SCHEMA,
        "source_commit": source_commit,
        "declaration_identity": declaration_identity,
        "package_identity": package_identity,
        "preview": preview,
        "automatic_retry": False,
    })
    ledger = _CallLedger(maximum_calls)
    failures: list[dict[str, Any]] = []
    for task in shard_tasks(tasks, rank=rank, workers=workers):
        target_id = str(task["target_record_id"])
        target = targets.get(target_id)
        if target is None:
            raise PaperContractError("assessment target is missing from frozen cohort")
        wall_start, cpu_start = perf_counter(), process_time()
        ledger.begin()
        try:
            evaluated = evaluate(task["recommended_theta"], task["assessment_context"], target)
            if isinstance(evaluated, AssessmentEvaluationV3):
                response = _finite_vector(evaluated.response, length=20, field="assessment response")
                randomness = dict(evaluated.randomness_evidence)
                raw_randomness = None if evaluated.raw_randomness_evidence is None else dict(evaluated.raw_randomness_evidence)
            else:
                response = _finite_vector(evaluated, length=20, field="assessment response")
                randomness = None
                raw_randomness = None
            goal = _finite_vector(target["target_response"], length=20, field="target response")
            theta = _finite_vector(task["recommended_theta"], length=4, field="recommended theta")
            differences = [value - expected for value, expected in zip(response, goal)]
            truth = target.get("theta_star")
            parameter_error = None
            if truth is not None:
                truth_values = _finite_vector(truth, length=4, field="theta star")
                parameter_error = math.sqrt(sum((value - expected) ** 2 for value, expected in zip(theta, truth_values)))
            body = {
                "schema_version": ATTEMPT_SCHEMA,
                "status": "completed",
                "source_commit": source_commit,
                "declaration_identity": declaration_identity,
                "package_identity": package_identity,
                "task_identity": task["task_identity"],
                "budget": task["budget"],
                "target_position": task["target_position"],
                "target_record_id": target_id,
                "method": task["method"],
                "recommendation_identity": task["recommendation_identity"],
                "recommended_theta": theta,
                "assessment_context": task["assessment_context"],
                "assessment_response": response,
                "assessment_randomness_evidence": randomness,
                "assessment_randomness_raw": raw_randomness,
                "response_error_l1_sum": sum(abs(value) for value in differences),
                "response_error_mae": sum(abs(value) for value in differences) / 20,
                "response_error_l2": math.sqrt(sum(value * value for value in differences)),
                "parameter_error_l2": parameter_error,
                "call_accounting": {
                    "assessment_attempted_calls": 1,
                    "assessment_successful_calls": 1,
                    "assessment_failed_calls": 0,
                    "retry_calls": 0,
                    "search_calls": 0,
                    "gp_fit_attempts": 0,
                    "terminal_optimizations": 0,
                },
            }
            result = {**body, "result_identity": stable_hash(RESULT_DOMAIN, body)}
            ledger.success()
        except Exception as exc:
            ledger.failure()
            result = {
                "schema_version": ATTEMPT_SCHEMA,
                "status": "failed",
                "source_commit": source_commit,
                "declaration_identity": declaration_identity,
                "package_identity": package_identity,
                "task_identity": task["task_identity"],
                "budget": task["budget"],
                "target_position": task["target_position"],
                "target_record_id": target_id,
                "method": task["method"],
                "recommendation_identity": task["recommendation_identity"],
                "recommended_theta": task["recommended_theta"],
                "assessment_context": task["assessment_context"],
                "failure_stage": "paired_heldout_assessment",
                "failure_type": type(exc).__name__,
                "failure_explanation": str(exc),
                "call_accounting": {
                    "assessment_attempted_calls": 1,
                    "assessment_successful_calls": 0,
                    "assessment_failed_calls": 1,
                    "retry_calls": 0,
                    "search_calls": 0,
                    "gp_fit_attempts": 0,
                    "terminal_optimizations": 0,
                },
            }
            failures.append({
                "task_identity": task["task_identity"],
                "failure_type": type(exc).__name__,
                "failure_explanation": str(exc),
            })
        result["timing"] = {
            "wall_seconds": perf_counter() - wall_start,
            "process_cpu_seconds": process_time() - cpu_start,
        }
        _append_hashed(_attempt_path(output, task), result)
    terminal = {
        "schema_version": WORKER_TERMINAL_SCHEMA,
        "status": "failed" if failures else "completed",
        "source_commit": source_commit,
        "declaration_identity": declaration_identity,
        "package_identity": package_identity,
        "rank": rank,
        "workers": workers,
        "physical_calls": ledger.document(),
        "attempt_records": maximum_calls,
        "failures": failures,
        "automatic_retry": False,
    }
    _append_hashed(output / "worker_terminal.json", terminal)
    if failures:
        raise PaperContractError("assessment worker failed; append-only evidence was preserved")
    return terminal

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.repaired_sbi_assessment_execution")
