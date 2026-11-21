"""External fresh-process execution and cold end-to-end timing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from time import perf_counter, process_time
from typing import Any, Mapping

from ..artifacts import read_json, write_pretty_json
from .direct_posterior_diagnostics import (
    SBI_TENSORBOARD_AUDIT_SCHEMA,
    safe_audit_training_artifacts,
)


def run_sbi_target_process(
    *,
    method: str,
    simulator_adapter: str,
    target_index: int,
    target_response: list[float],
    target_record_id: str | None = None,
    config: Mapping[str, Any],
    device: str,
    deadline_seconds: float | None,
    hard_timeout_seconds: float | None = None,
    output: Path,
    repository_root: Path,
    fixed_inference_evaluation_seed: int | None = None,
) -> dict[str, Any]:
    process_root = output / method / f"target_{target_index:03d}"
    process_root.mkdir(parents=True, exist_ok=True)
    request_path = process_root / "request.json"
    result_path = process_root / "result.json"
    stdout_path = process_root / "stdout.log"
    stderr_path = process_root / "stderr.log"
    request = {
        "schema_version": "mutaco-sbi-worker-request-v1",
        "method": method,
        "simulator_adapter": simulator_adapter,
        "target_index": target_index,
        "target_response": target_response,
        "target_record_id": target_record_id,
        "configuration": dict(config),
        "device": device,
        "deadline_seconds": deadline_seconds,
        "artifact_output": str(process_root / "training"),
        "fixed_inference_evaluation_seed": fixed_inference_evaluation_seed,
    }
    write_pretty_json(request_path, request)
    command = [
        sys.executable,
        "-m",
        "experiments.benchmarking.methods.worker",
        "--input",
        str(request_path),
        "--output",
        str(result_path),
    ]
    started_utc = datetime.now(timezone.utc).isoformat()
    wall_started, cpu_started = perf_counter(), process_time()
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=hard_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        wall_seconds = perf_counter() - wall_started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return {
            "schema_version": "mutaco-sbi-target-v1",
            "target_index": target_index,
            "method": method,
            "status": "failed",
            "failure_explanation": (
                f"WorkerProcessTimeout: hard_timeout_seconds={hard_timeout_seconds}"
            ),
            "failure_type": "timeout",
            "failure_stage": "worker_process",
            "inference_calls": 0,
            "assessment_calls": 0,
            "gp_fits": 0,
            "runtime_seconds": wall_seconds,
            "cold_process_timing": {
                "started_utc": started_utc,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": wall_seconds,
                "parent_process_cpu_seconds": process_time() - cpu_started,
                "command": command,
                "exit_code": None,
                "termination_reason": "deadline_exceeded",
            },
        }
    wall_seconds = perf_counter() - wall_started
    parent_cpu_seconds = process_time() - cpu_started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        return {
            "schema_version": "mutaco-sbi-target-v1",
            "target_index": target_index,
            "method": method,
            "status": "failed",
            "failure_explanation": (
                f"WorkerProcessError: exit={completed.returncode}; "
                f"stderr={completed.stderr[-1000:]}"
            ),
            "inference_calls": 0,
            "assessment_calls": 0,
            "gp_fits": 0,
            "runtime_seconds": wall_seconds,
            "cold_process_timing": {
                "started_utc": started_utc,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": wall_seconds,
                "parent_process_cpu_seconds": parent_cpu_seconds,
                "command": command,
                "exit_code": completed.returncode,
            },
        }
    result = read_json(result_path)
    sbi_config = config.get("sbi", {})
    audit_policy = sbi_config.get("training_diagnostics_audit")
    if audit_policy is not None:
        if audit_policy != SBI_TENSORBOARD_AUDIT_SCHEMA:
            result["training_diagnostics_audit"] = {
                "diagnostic_schema": audit_policy,
                "diagnostic_status": "training_diagnostics_audit_failed",
                "diagnostic_acceptance": "failed",
                "diagnostic_failure": (
                    f"UnknownTrainingDiagnosticsAuditPolicy: {audit_policy}"
                ),
                "diagnostic_simulator_calls": 0,
                "diagnostic_training_calls": 0,
            }
        elif result.get("status") == "completed":
            result["training_diagnostics_audit"] = (
                safe_audit_training_artifacts(
                    process_root / "training",
                    result.get("training_diagnostics", ()),
                    max_num_epochs=int(sbi_config["max_num_epochs"]),
                    stop_after_epochs=int(sbi_config["stop_after_epochs"]),
                )
            )
    result["cold_process_timing"] = {
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "parent_process_cpu_seconds": parent_cpu_seconds,
        "command": command,
        "exit_code": completed.returncode,
    }
    result["cold_end_to_end_seconds"] = wall_seconds
    return result
