"""Run the bounded forward-only approved graph-oscillator experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import torch

from mutaco.simulator import SimulationContext
from mutaco_graph import (
    SCIENTIFIC_IDENTITY,
    SCIENTIFIC_M20,
    GraphSimulationConfig,
    OverdampedScientificSimulator,
    generate_scientific_realization,
    parameter_profile,
    run_scientific_torch,
)

if __package__:
    from .artifacts import file_hash, write_csv, write_json
    from .plotting import render_forward_plots
else:
    from artifacts import file_hash, write_csv, write_json
    from plotting import render_forward_plots


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "graph_forward" / "prompt7_cpu"


def _relative_l2(reference: torch.Tensor, value: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(reference)), 1e-30)
    return float(torch.linalg.vector_norm(value - reference)) / denominator


def execute(*, output: Path, device: str = "auto", seed: int = 73007) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    print("[graph-forward 1/5] resolving approved CPU/CUDA-neutral configuration")
    base = GraphSimulationConfig(
        parameter_profile_id="p4-paper-baseline",
        node_count=6,
        batch_size=2,
        repeat_size=2,
        device=device,
    )
    simulator = OverdampedScientificSimulator(base)
    effective = simulator.device_resolution.device
    theta = torch.tensor((1.0, 1.25, 1.0, 0.30), dtype=torch.float32, device=effective)
    context = SimulationContext("prompt7-forward-v1", "forward_only", 0, 0, seed)
    profile = parameter_profile(base.parameter_profile_id)

    print(
        "[graph-forward 2/5] generating named streams "
        f"P=4 N={base.node_count} B={base.batch_size} R={base.repeat_size} "
        f"V=10 device={effective} fallback={simulator.device_resolution.used_fallback}"
    )
    generation_started = perf_counter()
    realization = generate_scientific_realization(
        profile.expand_theta(theta),
        base,
        context,
        device=effective,
    )
    generation_seconds = perf_counter() - generation_started

    print("[graph-forward 3/5] integrating 1000 steps and evaluating fixed-length sensitivity")
    schedules = (
        (0.0, GraphSimulationConfig(**{**base.__dict__, "burn_in_steps": 0, "sample_steps": 1000})),
        (0.1, base),
        (0.2, GraphSimulationConfig(**{**base.__dict__, "burn_in_steps": 200, "sample_steps": 800})),
    )
    runs = [(fraction, config, run_scientific_torch(realization, config)) for fraction, config in schedules]
    approved = runs[1][2]
    sensitivity = [
        {
            "burn_fraction": fraction,
            "burn_in_steps": config.burn_in_steps,
            "sample_steps": config.sample_steps,
            "relative_l2_from_10_percent": _relative_l2(approved.response, run.response),
            "response": run.response.detach().cpu().tolist(),
        }
        for fraction, config, run in runs
    ]
    material_threshold = 0.05
    residual_transient_limitation = sensitivity[2]["relative_l2_from_10_percent"] > material_threshold

    result = {
        "schema_version": "prompt7-graph-forward-result-v1",
        "model_identity": SCIENTIFIC_IDENTITY,
        "simulator_identity": simulator.capabilities.simulator_identity,
        "theta": theta.detach().cpu().tolist(),
        "response": approved.response.detach().cpu().tolist(),
        "response_names": list(SCIENTIFIC_M20.names),
        "configuration": {
            **base.identity_document(),
            "forces": list(SCIENTIFIC_M20.forces),
            "gauges": list(SCIENTIFIC_M20.gauges),
        },
        "context": {
            "scientific_run_id": context.scientific_run_id,
            "purpose": context.purpose,
            "logical_index": context.logical_index,
            "repeat_index": context.repeat_index,
            "evaluation_seed": context.evaluation_seed,
        },
        "seed_manifest": dict(realization.seed_manifest),
        "burn_in_sensitivity": sensitivity,
        "burn_in_material_threshold_relative_l2": material_threshold,
        "residual_transient_limitation": residual_transient_limitation,
        "timing": {
            "generation_seconds": generation_seconds,
            "approved_evolution_seconds": approved.timings["evolution_seconds"],
            "elapsed_before_artifacts_seconds": perf_counter() - started,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device_requested": device,
            "device_effective": str(effective),
            "device_fallback": simulator.device_resolution.used_fallback,
            "dtype": "float32",
            "cuda_parity_claimed": False,
        },
    }
    print("[graph-forward 4/5] writing public JSON/CSV and PDF/SVG artifacts")
    write_json(output / "result.json", result)
    response_rows = []
    for quantity_index, quantity in enumerate(("omega", "Z_magnitude")):
        for force_index, force in enumerate(SCIENTIFIC_M20.forces):
            for gauge_index, gauge in enumerate(("0", "pi")):
                component = (
                    quantity_index * 10 + force_index * 2 + gauge_index
                )
                response_rows.append(
                    {
                        "component": component,
                        "quantity": quantity,
                        "force": force,
                        "gauge": gauge,
                        "value": result["response"][component],
                    }
                )
    write_csv(output / "response.csv", response_rows)
    write_csv(
        output / "burn_in_sensitivity.csv",
        (
            {
                "burn_fraction": item["burn_fraction"],
                "burn_in_steps": item["burn_in_steps"],
                "sample_steps": item["sample_steps"],
                "relative_l2_from_10_percent": item["relative_l2_from_10_percent"],
            }
            for item in sensitivity
        ),
    )
    plot_paths = render_forward_plots(result, output)
    hashed = ("result.json", "response.csv", "burn_in_sensitivity.csv") + tuple(
        path.name for path in plot_paths
    )
    manifest = {
        "schema_version": "prompt7-graph-forward-manifest-v1",
        "model_identity": SCIENTIFIC_IDENTITY,
        "sha256": {name: file_hash(output / name) for name in hashed},
        "cuda_validation_command": (
            "python -m experiments.graph_forward.validate_cuda "
            "--output <output> --device cuda"
        ),
        "cuda_parity_claimed": False,
        "total_elapsed_seconds": perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    print(
        "[graph-forward 5/5] complete "
        f"elapsed={manifest['total_elapsed_seconds']:.3f}s output={output}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=73007)
    arguments = parser.parse_args()
    execute(output=arguments.output, device=arguments.device, seed=arguments.seed)


if __name__ == "__main__":
    main()
