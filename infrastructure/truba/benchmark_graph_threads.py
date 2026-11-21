"""Bounded graph-forward CPU thread benchmark for TRUBA qualification."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from mutaco import SimulationContext, checked_evaluate
from experiments.benchmarking.simulators import get_simulator_adapter
from experiments.paper.common.validation import read_json_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 2:
        raise ValueError("threads must be positive and repetitions must be at least two")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[2]
    config = read_json_object(root / "experiments/graph_inversion/configs/d4_100_targets.json")
    target = read_json_object(
        Path("/arf/scratch") / getpass.getuser()
        / "mutaco/prompt16/source-target/target_000.json"
    )
    adapter = get_simulator_adapter(config["simulator"]["adapter"])
    simulator = adapter.build_simulator(config, device=torch.device("cpu"), dtype=torch.float32)
    theta = torch.tensor(target["theta_star"], dtype=torch.float32)
    context = SimulationContext(**target["target_context"])

    durations: list[float] = []
    responses: list[list[float]] = []
    for _ in range(args.repetitions):
        started = perf_counter()
        result = checked_evaluate(
            simulator, theta, device=torch.device("cpu"), dtype=torch.float32,
            context=context,
        )
        durations.append(perf_counter() - started)
        responses.append(result.values.detach().cpu().tolist())
    encoded = json.dumps(responses[0], separators=(",", ":")).encode("utf-8")
    document = {
        "schema_version": "mutaco-truba-thread-benchmark-v1",
        "classification": "engineering_no_claim",
        "threads_requested": args.threads,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "repetitions": args.repetitions,
        "durations_seconds": durations,
        "median_seconds": median(durations),
        "within_process_exact_replay": all(item == responses[0] for item in responses[1:]),
        "response": responses[0],
        "response_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_target_content_hash": target["content_hash"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
