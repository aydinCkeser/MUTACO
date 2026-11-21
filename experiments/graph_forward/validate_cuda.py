"""Bounded real-CUDA validation using identical captured realized inputs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform

import torch

from mutaco.simulator import SimulationContext
from mutaco_graph import (
    FINAL_RESPONSE_ATOL,
    FINAL_RESPONSE_RELATIVE_L2,
    FINAL_RESPONSE_RTOL,
    FIXED_INTERMEDIATE_ATOL,
    FIXED_INTERMEDIATE_RTOL,
    SCIENTIFIC_FIXED_THETA_P4,
    GraphSimulationConfig,
    ScientificRealization,
    difference_metrics,
    generate_scientific_realization,
    parameter_profile,
    run_scientific_torch,
)

from .artifacts import write_json


def _to_cuda(realization: ScientificRealization) -> ScientificRealization:
    return replace(
        realization,
        weight=realization.weight.to("cuda"),
        inverse_viscosity=realization.inverse_viscosity.to("cuda"),
        bias_scale=realization.bias_scale.to("cuda"),
        coupling=realization.coupling.to("cuda"),
        initial_phase=realization.initial_phase.to("cuda"),
        thermal_noise=realization.thermal_noise.to("cuda"),
    )


def execute(output: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("real Torch CUDA is unavailable; CUDA parity was not executed")
    torch.use_deterministic_algorithms(True)
    cpu_config = GraphSimulationConfig(
        parameter_profile_id="p4-paper-baseline",
        node_count=4,
        batch_size=2,
        repeat_size=2,
        burn_in_steps=10,
        sample_steps=20,
        device="cpu",
    )
    theta = torch.tensor(SCIENTIFIC_FIXED_THETA_P4[0], dtype=torch.float32)
    context = SimulationContext("prompt7-cuda-validation-v1", "cuda_parity", 0, 0, 73007)
    realization_cpu = generate_scientific_realization(
        parameter_profile(cpu_config.parameter_profile_id).expand_theta(theta),
        cpu_config,
        context,
        device=torch.device("cpu"),
    )
    cpu = run_scientific_torch(realization_cpu, cpu_config)
    cuda_config = replace(cpu_config, device="cuda")
    cuda = run_scientific_torch(_to_cuda(realization_cpu), cuda_config)
    intermediates = {}
    passed = True
    for name in cpu.first_step:
        metric = difference_metrics(
            cpu.first_step[name],
            cuda.first_step[name],
            rtol=FIXED_INTERMEDIATE_RTOL,
            atol=FIXED_INTERMEDIATE_ATOL,
        )
        intermediates[name] = metric.__dict__
        passed = passed and metric.allclose
    final = difference_metrics(
        cpu.response,
        cuda.response,
        rtol=FINAL_RESPONSE_RTOL,
        atol=FINAL_RESPONSE_ATOL,
    )
    passed = passed and final.allclose and final.relative_l2 <= FINAL_RESPONSE_RELATIVE_L2
    document = {
        "schema_version": "prompt7-graph-cuda-parity-v1",
        "passed": passed,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "same_captured_realized_inputs": True,
        "intermediates": intermediates,
        "final_response": final.__dict__,
        "thresholds": {
            "intermediate_rtol": FIXED_INTERMEDIATE_RTOL,
            "intermediate_atol": FIXED_INTERMEDIATE_ATOL,
            "final_rtol": FINAL_RESPONSE_RTOL,
            "final_atol": FINAL_RESPONSE_ATOL,
            "final_relative_l2": FINAL_RESPONSE_RELATIVE_L2,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cuda_parity.json", document)
    if not passed:
        raise RuntimeError("real CUDA parity gate failed; see cuda_parity.json")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    result = execute(args.output)
    print(json.dumps({"passed": result["passed"], "device": result["device"]}, indent=2))


if __name__ == "__main__":
    main()
