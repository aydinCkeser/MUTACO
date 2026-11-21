"""Fresh worker for the Prompt 5D shared-bank NPE stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mutaco import RuntimeRequest, ScientificIdentity, resolve_runtime

from ..artifacts import write_pretty_json
from ..simulators import get_simulator_adapter
from .sbi_amortized import run_amortized_bank


def execute(input_path: Path, output_path: Path) -> list[dict]:
    request = json.loads(input_path.read_text(encoding="utf-8"))
    config = request["configuration"]
    runtime = resolve_runtime(RuntimeRequest(
        device=request["device"], dtype=config["dtype"]
    ))
    adapter = get_simulator_adapter(request["simulator_adapter"])
    bank = adapter.target_bank(
        config, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    records = request["target_records"]
    indices = tuple(int(record["target_index"]) for record in records)
    responses = {
        int(record["target_index"]): torch.tensor(
            record["target_response"], device=runtime.torch_device,
            dtype=runtime.torch_dtype,
        ).contiguous()
        for record in records
    }
    rows = run_amortized_bank(
        method="sbi_npe", adapter=adapter, target_indices=indices,
        theta_bank=bank.theta, target_responses=responses, config=config,
        campaign_science=ScientificIdentity.create(request["campaign_science_document"]),
        device=runtime.torch_device, dtype=runtime.torch_dtype,
        artifact_output=Path(request["artifact_output"]),
    )
    write_pretty_json(output_path, {"rows": rows, "runtime": runtime.provenance()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.input, args.output)


if __name__ == "__main__":
    main()
