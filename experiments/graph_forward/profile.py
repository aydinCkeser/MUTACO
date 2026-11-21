"""Profile Prompt 7 graph-forward generation and evolution on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import torch

from mutaco.simulator import SimulationContext
from mutaco_graph import (
    GraphSimulationConfig,
    generate_scientific_realization,
    parameter_profile,
    run_scientific_torch,
)

from .artifacts import write_json


ROOT = Path(__file__).resolve().parents[2]


def _profile_case(label: str, config: GraphSimulationConfig) -> dict[str, Any]:
    theta = torch.tensor((1.0, 1.25, 1.0, 0.30), dtype=torch.float32)
    context = SimulationContext("prompt7-profile-v1", "profiling", 0, 0, 73007)
    profile = parameter_profile(config.parameter_profile_id)
    started = perf_counter()
    realization = generate_scientific_realization(
        profile.expand_theta(theta),
        config,
        context,
        device=torch.device("cpu"),
    )
    generation = perf_counter() - started
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in realization.tensors())
    run = run_scientific_torch(realization, config)
    return {
        "label": label,
        "configuration": config.identity_document(),
        "generation_seconds": generation,
        "evolution_and_reduction_seconds": run.timings["evolution_seconds"],
        "realized_input_bytes": tensor_bytes,
        "response_components": run.response.numel(),
    }


def execute(output: Path) -> dict[str, Any]:
    cases = (
        (
            "representative_small",
            GraphSimulationConfig(
                parameter_profile_id="p4-paper-baseline",
                node_count=6,
                batch_size=2,
                repeat_size=2,
            ),
        ),
        (
            "projected_scientific",
            GraphSimulationConfig(
                parameter_profile_id="p4-paper-baseline",
                node_count=20,
                batch_size=8,
                repeat_size=4,
            ),
        ),
    )
    results = []
    for label, config in cases:
        print(f"[graph-profile] {label} N={config.node_count} B={config.batch_size} R={config.repeat_size}")
        results.append(_profile_case(label, config))
    document = {
        "schema_version": "prompt7-graph-cpu-profile-v1",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "cases": results,
        "code_path_findings": {
            "python_time_loop": True,
            "time_loop_iterations": 1000,
            "repeated_phase_difference_allocation": "[V,B,R,N,N] each step",
            "repeated_transcendental_temporaries": ["sin(restoring)", "sin(pair differences)", "exp(i*phase)"],
            "host_to_device_transfers_inside_loop": 0,
            "device_to_host_transfers_inside_loop": 0,
            "synchronization_points_inside_loop": 0,
            "generation": "device-native named Torch generators",
            "priority_bottleneck": "O(V*B*R*N^2) pairwise phase-difference tensor and Python time loop",
        },
        "cuda_parity_claimed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cpu_profile.json", document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "graph_forward" / "prompt7_profile",
    )
    args = parser.parse_args()
    execute(args.output)


if __name__ == "__main__":
    main()
