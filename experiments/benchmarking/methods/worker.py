"""Fresh-process worker for exactly one target-specific SBI method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mutaco import resolve_runtime
from experiments.benchmarking.runtime import (
    configure_graph_determinism,
    runtime_request_from_resolved_device,
)

from ..artifacts import write_pretty_json
from ..simulators import get_simulator_adapter
from .sbi_suite import run_sbi_target


def execute(input_path: Path, output_path: Path) -> dict:
    request = json.loads(input_path.read_text(encoding="utf-8"))
    config = request["configuration"]
    runtime = resolve_runtime(
        runtime_request_from_resolved_device(
            request["device"], dtype=config["dtype"]
        )
    )
    configure_graph_determinism(config, runtime.torch_device)
    adapter = get_simulator_adapter(request["simulator_adapter"])
    bank = adapter.target_bank(
        config, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    index = int(request["target_index"])
    target_response = torch.tensor(
        request["target_response"],
        device=runtime.torch_device,
        dtype=runtime.torch_dtype,
    ).contiguous()
    result = run_sbi_target(
        method=request["method"],
        adapter=adapter,
        index=index,
        theta_star=bank.theta[index],
        target_response=target_response,
        config=config,
        campaign_science=bank.science,
        device=runtime.torch_device,
        dtype=runtime.torch_dtype,
        deadline_seconds=request.get("deadline_seconds"),
        artifact_output=Path(request["artifact_output"]),
        target_record_id=request.get("target_record_id"),
        fixed_inference_evaluation_seed=request.get(
            "fixed_inference_evaluation_seed"
        ),
    )
    result["worker_runtime"] = runtime.provenance()
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
