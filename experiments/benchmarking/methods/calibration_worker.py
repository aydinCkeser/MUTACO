"""Fresh calibration worker for one MUTACO or SBI method-target pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter, process_time
import traceback


def execute(input_path: Path, output_path: Path) -> dict:
    worker_started = perf_counter()
    worker_cpu_started = process_time()
    worker_started_utc = datetime.now(timezone.utc).isoformat()
    request_started = perf_counter()
    request = json.loads(input_path.read_text(encoding="utf-8"))
    request_seconds = perf_counter() - request_started

    import_started = perf_counter()
    import torch

    from mutaco import RuntimeRequest, ScientificIdentity, resolve_runtime

    from ..artifacts import write_pretty_json
    from ..methods import run_mutaco_target
    from ..simulators import get_simulator_adapter
    from .sbi_suite import run_sbi_target

    import_seconds = perf_counter() - import_started
    method = str(request["method"])
    target_index = int(request["target_index"])
    config = request["configuration"]
    phase = "device_resolution"
    try:
        device_started = perf_counter()
        runtime = resolve_runtime(
            RuntimeRequest(device=request["device"], dtype=config["dtype"])
        )
        device_seconds = perf_counter() - device_started

        preparation_started = perf_counter()
        adapter = get_simulator_adapter(request["simulator_adapter"])
        bank = adapter.target_bank(
            config, device=runtime.torch_device, dtype=runtime.torch_dtype
        )
        theta_star = bank.theta[target_index]
        campaign_science = (
            ScientificIdentity.create(request["campaign_science_document"])
            if request.get("campaign_science_document") is not None
            else bank.science
        )
        preparation_seconds = perf_counter() - preparation_started

        phase = "method_execution"
        method_started = perf_counter()
        progress_path = Path(request["progress_output"])

        def persist_progress(document: dict) -> None:
            write_pretty_json(
                progress_path,
                {
                    "schema_version": "mutaco-calibration-progress-v1",
                    "method": method,
                    "target_index": target_index,
                    **document,
                },
            )

        if method == "mutaco":
            supplied_target = (
                torch.tensor(
                    request["target_response"],
                    device=runtime.torch_device,
                    dtype=runtime.torch_dtype,
                ).contiguous()
                if request.get("target_response") is not None
                else None
            )
            result = run_mutaco_target(
                adapter=adapter,
                index=target_index,
                theta_star=theta_star,
                config=config,
                campaign_science=campaign_science,
                device=runtime.torch_device,
                dtype=runtime.torch_dtype,
                accounting_callback=persist_progress,
                supplied_target_response=supplied_target,
                supplied_target_seed=request.get("target_seed"),
            )
        else:
            target_response = torch.tensor(
                request["target_response"],
                device=runtime.torch_device,
                dtype=runtime.torch_dtype,
            ).contiguous()
            result = run_sbi_target(
                method=method,
                adapter=adapter,
                index=target_index,
                theta_star=theta_star,
                target_response=target_response,
                config=config,
                campaign_science=campaign_science,
                device=runtime.torch_device,
                dtype=runtime.torch_dtype,
                deadline_seconds=None,
                artifact_output=Path(request["artifact_output"]),
                progress_callback=persist_progress,
            )
        method_seconds = perf_counter() - method_started
    except Exception as exc:
        result = {
            "schema_version": "mutaco-calibration-worker-result-v1",
            "campaign_id": request.get("campaign_id"),
            "target_index": target_index,
            "method": method,
            "status": "failed",
            "failure_type": f"{phase}_failure",
            "failure_explanation": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "target_construction_calls": 0,
            "inference_calls": 0,
            "assessment_calls": 0,
            "gp_fits": 0,
        }
        device_seconds = locals().get("device_seconds", 0.0)
        preparation_seconds = locals().get("preparation_seconds", 0.0)
        method_seconds = locals().get("method_seconds", 0.0)
        runtime = None
    result["worker"] = {
        "schema_version": "mutaco-calibration-worker-v1",
        "pid": os.getpid(),
        "started_utc": worker_started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "request_read_seconds": request_seconds,
        "scientific_import_seconds": import_seconds,
        "device_resolution_seconds": device_seconds,
        "target_preparation_seconds": preparation_seconds,
        "method_execution_seconds": method_seconds,
        "total_wall_seconds": perf_counter() - worker_started,
        "process_cpu_seconds": process_time() - worker_cpu_started,
        "runtime": runtime.provenance() if runtime is not None else None,
    }
    result["effective_method_configuration"] = {
        "method": method,
        "dtype": config["dtype"],
        "requested_device": request["device"],
        "simulation_budget_per_target": config["simulation_budget_per_target"],
        "runner": config["runner"] if method == "mutaco" else None,
        "sbi": config["sbi"] if method != "mutaco" else None,
        "fallback_policy": "none",
    }
    write_pretty_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.input, args.output)


if __name__ == "__main__":
    main()
