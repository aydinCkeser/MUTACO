"""Append-only target-centric primary worker orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.paper.protocol.population_v3 import shard_indices
from experiments.paper.protocol.validation import PaperContractError
from .design import PrimarySobolDesign
from .execution import PrimaryTargetExecution, execute_primary_target
from .protocol import primary_population_call_preview
from .qualification_execution import append_hashed_json


def primary_worker_preview(
    declaration: Any,
    *,
    rank: int,
    workers: int,
    maximum_calls: int,
) -> dict[str, Any]:
    if workers < 1 or rank not in range(workers):
        raise PaperContractError("primary worker geometry is invalid")
    positions = shard_indices(len(declaration.target_record_identities), rank=rank, workers=workers)
    planned = primary_population_call_preview(len(positions), declaration.budget)[
        "maximum_total_calls"
    ]
    if planned > maximum_calls:
        raise PaperContractError("primary worker call cap rejects declared shard")
    return {
        "schema_version": "mutaco-primary-worker-preview-v1",
        "declaration_identity": declaration.document()["declaration_identity"],
        "rank": rank,
        "workers": workers,
        "target_positions": list(positions),
        "planned_calls": planned,
        "maximum_calls_argument": maximum_calls,
        "scientific_calls_executed": 0,
    }


def execute_primary_worker(
    *,
    declaration: Any,
    output: Path,
    rank: int,
    workers: int,
    maximum_calls: int,
    source_commit: str,
    load_target: Callable[[str], Mapping[str, Any]],
    build_design: Callable[[str], PrimarySobolDesign],
    run_method: Callable[..., Any],
    evaluate_search: Callable[[str, Sequence[float], Mapping[str, Any]], Sequence[float]],
    evaluate_assessment: Callable[[Sequence[float], Mapping[str, Any]], Sequence[float]],
) -> dict[str, Any]:
    """Run a Q2 or population shard through the common capped target executor."""

    preview = primary_worker_preview(
        declaration, rank=rank, workers=workers, maximum_calls=maximum_calls
    )
    if output.exists():
        raise PaperContractError("primary worker output must be a new append-only path")
    identity = declaration.document()["declaration_identity"]
    append_hashed_json(output / "worker_start.json", {
        "schema_version": "mutaco-primary-worker-start-v1",
        "source_commit": source_commit,
        "declaration_identity": identity,
        "preview": preview,
    })
    executions: list[PrimaryTargetExecution] = []
    attempted = successful = failed = 0
    status = "completed"
    error = None
    try:
        for position in preview["target_positions"]:
            target_id = declaration.target_record_identities[int(position)]
            target = load_target(target_id)
            if target.get("target_record_id") != target_id:
                raise PaperContractError("loaded primary target identity mismatch")
            design = build_design(target_id)
            target_dir = output / "targets" / f"target-{int(position):03d}"
            def tracked_search(
                method: str, theta: Sequence[float], context: Mapping[str, Any]
            ) -> Sequence[float]:
                nonlocal attempted, successful, failed
                if attempted >= maximum_calls:
                    raise PaperContractError("primary worker physical-call cap exhausted")
                attempted += 1
                try:
                    value = evaluate_search(method, theta, context)
                except Exception:
                    failed += 1
                    raise
                successful += 1
                return value

            def tracked_assessment(
                theta: Sequence[float], context: Mapping[str, Any]
            ) -> Sequence[float]:
                nonlocal attempted, successful, failed
                if attempted >= maximum_calls:
                    raise PaperContractError("primary worker physical-call cap exhausted")
                attempted += 1
                try:
                    value = evaluate_assessment(theta, context)
                except Exception:
                    failed += 1
                    raise
                successful += 1
                return value

            execution = execute_primary_target(
                declaration=declaration,
                target_id=target_id,
                target_response=target["target_response"],
                design=design,
                run_method=run_method,
                evaluate_search=tracked_search,
                persist_freeze=lambda document, root=target_dir: append_hashed_json(
                    root / "decision_freeze.json", document
                ),
                evaluate_assessment=tracked_assessment,
            )
            append_hashed_json(target_dir / "target_execution.json", execution.document())
            for record in execution.records:
                append_hashed_json(target_dir / f"{record['method']}.json", record)
            executions.append(execution)
    except Exception as exc:
        status = "failed"
        error = {"failure_type": type(exc).__name__, "explanation": str(exc)}
    calls = {
        "maximum_calls": maximum_calls,
        "attempted_calls": attempted,
        "successful_calls": successful,
        "failed_calls": failed,
    }
    terminal = {
        "schema_version": "mutaco-primary-worker-terminal-v1",
        "source_commit": source_commit,
        "declaration_identity": identity,
        "status": status,
        "completed_targets": len(executions),
        "physical_calls": calls,
        "failure": error,
    }
    append_hashed_json(output / "worker_terminal.json", terminal)
    if status != "completed":
        raise PaperContractError("primary worker failed; terminal evidence was preserved")
    if calls["attempted_calls"] != preview["planned_calls"]:
        raise PaperContractError("primary completed worker call ledger differs from preview")
    return terminal
