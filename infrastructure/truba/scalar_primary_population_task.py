"""Run two scalar-primary logical workers sequentially in one Slurm array task."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


Runner = Callable[..., subprocess.CompletedProcess[object]]


def logical_ranks(
    task_id: int, *, physical_tasks: int, logical_workers: int
) -> tuple[int, ...]:
    if physical_tasks != 60 or logical_workers != 120:
        raise ValueError("scalar-primary v3 requires 60 physical and 120 logical workers")
    if task_id < 0 or task_id >= physical_tasks:
        raise ValueError("scalar-primary physical task id is outside [0, 59]")
    ranks = (task_id, task_id + physical_tasks)
    if any(rank < 0 or rank >= logical_workers for rank in ranks):
        raise ValueError("scalar-primary logical rank is outside [0, 119]")
    return ranks


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if not value:
        raise ValueError(f"required environment variable is missing: {name}")
    return value


def worker_command(environ: Mapping[str, str], rank: int) -> tuple[str, ...]:
    output_root = Path(_required(environ, "MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT"))
    worker_name = f"worker-{rank:02d}"
    return (
        _required(environ, "MUTACO_V3_PYTHON"),
        "-u",
        "-m",
        "experiments.graph_inversion.run_primary_integration_smoke_worker",
        "--declaration",
        _required(environ, "MUTACO_SCALAR_PRIMARY_DECLARATION"),
        "--plan",
        _required(environ, "MUTACO_SCALAR_PRIMARY_PLAN"),
        "--targets-root",
        _required(environ, "MUTACO_SCALAR_PRIMARY_TARGET_LEDGER"),
        "--config",
        _required(environ, "MUTACO_SCALAR_PRIMARY_CONFIG"),
        "--qlognei-settings",
        _required(environ, "MUTACO_SCALAR_PRIMARY_QLOGNEI_SETTINGS"),
        "--output",
        str(output_root / worker_name),
        "--rank",
        str(rank),
        "--workers",
        _required(environ, "MUTACO_SCALAR_PRIMARY_WORKERS"),
        "--maximum-calls",
        _required(environ, "MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER"),
        "--expected-commit",
        _required(environ, "MUTACO_EXPECTED_COMMIT"),
        "--torch-intraop-threads",
        _required(environ, "MUTACO_SCALAR_PRIMARY_CPUS_PER_TASK"),
        "--torch-interop-threads",
        "1",
        "--execute",
        "--confirm-approved",
    )


def execute_task(
    environ: Mapping[str, str], *, runner: Runner = subprocess.run
) -> tuple[int, dict[str, object]]:
    task_id = int(_required(environ, "SLURM_ARRAY_TASK_ID"))
    physical_tasks = int(_required(environ, "MUTACO_SCALAR_PRIMARY_PHYSICAL_TASKS"))
    logical_workers = int(_required(environ, "MUTACO_SCALAR_PRIMARY_WORKERS"))
    workers_per_task = int(
        _required(environ, "MUTACO_SCALAR_PRIMARY_LOGICAL_WORKERS_PER_TASK")
    )
    ranks = logical_ranks(
        task_id, physical_tasks=physical_tasks, logical_workers=logical_workers
    )
    if workers_per_task != len(ranks):
        raise ValueError("logical-workers-per-task does not match the frozen rank mapping")
    results: list[dict[str, int]] = []
    for rank in ranks:
        completed = runner(worker_command(environ, rank), check=False)
        results.append({"rank": rank, "returncode": int(completed.returncode)})
    document: dict[str, object] = {
        "schema_version": "mutaco-scalar-primary-physical-task-terminal-v1",
        "physical_task_id": task_id,
        "logical_ranks": list(ranks),
        "logical_worker_results": results,
        "all_logical_workers_succeeded": all(
            result["returncode"] == 0 for result in results
        ),
        "automatic_retry": False,
    }
    return (0 if document["all_logical_workers_succeeded"] else 1), document


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("scalar-primary physical task accepts no command arguments")
    returncode, document = execute_task(os.environ)
    print(json.dumps(document, sort_keys=True), flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
