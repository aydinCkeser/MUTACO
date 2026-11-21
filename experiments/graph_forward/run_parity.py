"""Execute the bounded predeclared Prompt 7 CPU parity cohorts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from mutaco.simulator import SimulationContext
from mutaco_graph import (
    FINAL_RESPONSE_ATOL,
    FINAL_RESPONSE_RELATIVE_L2,
    FINAL_RESPONSE_RTOL,
    FIXED_INTERMEDIATE_ATOL,
    FIXED_INTERMEDIATE_RTOL,
    FIXED_SEEDS,
    RETAINED_FIXED_THETA_P2,
    SCIENTIFIC_FIXED_THETA_P4,
    STATISTICAL_SEEDS,
    STATISTICAL_THETA_P4,
    GraphSimulationConfig,
    RetainedGraphFixture,
    RetainedOracleConfig,
    difference_metrics,
    generate_retained_realization,
    generate_scientific_realization,
    parameter_profile,
    retained_realization_from_fixture,
    run_retained_numpy_reference,
    run_retained_torch,
    run_scientific_numpy_reference,
    run_scientific_torch,
    statistical_gate,
)

from .artifacts import write_json


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "scientific_contracts"
    / "graph_oscillator"
    / "reference_fixtures"
    / "retained_cpu_small_v1.json"
)


def _context(seed: int, logical_index: int) -> SimulationContext:
    return SimulationContext("prompt7-parity-v1", "reference_parity", logical_index, 0, seed)


def _compare_runs(reference: Any, candidate: Any) -> dict[str, Any]:
    intermediates = {}
    passed = True
    for name in reference.first_step:
        metric = difference_metrics(
            reference.first_step[name],
            candidate.first_step[name],
            rtol=FIXED_INTERMEDIATE_RTOL,
            atol=FIXED_INTERMEDIATE_ATOL,
        )
        intermediates[name] = asdict(metric)
        passed = passed and metric.allclose
    final = difference_metrics(
        reference.response,
        candidate.response,
        rtol=FINAL_RESPONSE_RTOL,
        atol=FINAL_RESPONSE_ATOL,
    )
    passed = passed and final.allclose and final.relative_l2 <= FINAL_RESPONSE_RELATIVE_L2
    return {"passed": passed, "intermediates": intermediates, "final_response": asdict(final)}


def _fixed_scientific() -> dict[str, Any]:
    config = GraphSimulationConfig(
        parameter_profile_id="p4-paper-baseline",
        node_count=3,
        batch_size=1,
        repeat_size=1,
        burn_in_steps=2,
        sample_steps=4,
    )
    profile = parameter_profile(config.parameter_profile_id)
    cases = []
    for index, (theta_value, seed) in enumerate(zip(SCIENTIFIC_FIXED_THETA_P4, FIXED_SEEDS)):
        theta = torch.tensor(theta_value, dtype=torch.float32)
        realization = generate_scientific_realization(
            profile.expand_theta(theta),
            config,
            _context(seed, index),
            device=torch.device("cpu"),
        )
        reference = run_scientific_numpy_reference(realization, config)
        candidate = run_scientific_torch(realization, config)
        cases.append(
            {
                "case": index,
                "theta": list(theta_value),
                "seed": seed,
                **_compare_runs(reference, candidate),
            }
        )
    return {"passed": all(item["passed"] for item in cases), "cases": cases}


def _fixed_retained() -> dict[str, Any]:
    config = RetainedOracleConfig(node_count=3, batch_size=1, repeat_size=1, steps=4)
    cases = []
    for index, (theta_value, seed) in enumerate(zip(RETAINED_FIXED_THETA_P2, FIXED_SEEDS)):
        realization = generate_retained_realization(
            torch.tensor(theta_value, dtype=torch.float64),
            config,
            seed,
        )
        reference = run_retained_numpy_reference(
            realization,
            dt=config.dt,
            temperature=config.temperature,
        )
        candidate = run_retained_torch(
            realization,
            dt=config.dt,
            temperature=config.temperature,
        )
        cases.append(
            {
                "case": index,
                "theta": list(theta_value),
                "seed": seed,
                **_compare_runs(reference, candidate),
            }
        )
    fixture = RetainedGraphFixture.load(FIXTURE)
    realization = retained_realization_from_fixture(fixture)
    frozen = run_retained_torch(realization, dt=0.1, temperature=0.15)
    frozen_metric = difference_metrics(
        fixture.tensor("final_response")[0],
        frozen.response,
        rtol=FINAL_RESPONSE_RTOL,
        atol=FINAL_RESPONSE_ATOL,
    )
    frozen_direct = {
        "passed": frozen_metric.allclose
        and frozen_metric.relative_l2 <= FINAL_RESPONSE_RELATIVE_L2,
        "fixture_identity": fixture.fixture_identity,
        "metrics": asdict(frozen_metric),
    }
    return {
        "passed": all(item["passed"] for item in cases) and frozen_direct["passed"],
        "cases": cases,
        "direct_frozen_fixture": frozen_direct,
    }


def _statistical_scientific() -> dict[str, Any]:
    config = GraphSimulationConfig(
        parameter_profile_id="p4-paper-baseline",
        node_count=3,
        batch_size=1,
        repeat_size=1,
    )
    profile = parameter_profile(config.parameter_profile_id)
    theta_results = []
    for theta_index, theta_value in enumerate(STATISTICAL_THETA_P4):
        reference_values = []
        candidate_values = []
        theta = torch.tensor(theta_value, dtype=torch.float32)
        full = profile.expand_theta(theta)
        for seed_index, seed in enumerate(STATISTICAL_SEEDS):
            realization = generate_scientific_realization(
                full,
                config,
                _context(seed, theta_index * len(STATISTICAL_SEEDS) + seed_index),
                device=torch.device("cpu"),
            )
            reference_values.append(run_scientific_numpy_reference(realization, config).response.numpy())
            candidate_values.append(run_scientific_torch(realization, config).response.numpy())
        gate = statistical_gate(np.stack(reference_values), np.stack(candidate_values))
        theta_results.append({"theta": list(theta_value), **gate})
    return {"passed": all(item["passed"] for item in theta_results), "theta_results": theta_results}


def _statistical_retained() -> dict[str, Any]:
    config = RetainedOracleConfig(node_count=3, batch_size=1, repeat_size=1, steps=20)
    theta_results = []
    for theta_index, theta_value in enumerate(RETAINED_FIXED_THETA_P2[:8]):
        reference_values = []
        candidate_values = []
        theta = torch.tensor(theta_value, dtype=torch.float64)
        for seed in STATISTICAL_SEEDS:
            realization = generate_retained_realization(theta, config, seed)
            reference_values.append(
                run_retained_numpy_reference(
                    realization,
                    dt=config.dt,
                    temperature=config.temperature,
                ).response.numpy()
            )
            candidate_values.append(
                run_retained_torch(
                    realization,
                    dt=config.dt,
                    temperature=config.temperature,
                ).response.numpy()
            )
        gate = statistical_gate(np.stack(reference_values), np.stack(candidate_values))
        theta_results.append({"theta": list(theta_value), **gate})
    return {"passed": all(item["passed"] for item in theta_results), "theta_results": theta_results}


def execute(output: Path) -> dict[str, Any]:
    started = perf_counter()
    print("[graph-parity 1/4] retained and scientific fixed-input cohorts")
    fixed_retained = _fixed_retained()
    fixed_scientific = _fixed_scientific()
    print("[graph-parity 2/4] scientific 8 theta x 16 seed statistical cohort")
    statistical_scientific = _statistical_scientific()
    print("[graph-parity 3/4] retained 8 theta x 16 seed statistical cohort")
    statistical_retained = _statistical_retained()
    passed = all(
        (
            fixed_retained["passed"],
            fixed_scientific["passed"],
            statistical_retained["passed"],
            statistical_scientific["passed"],
        )
    )
    document = {
        "schema_version": "prompt7-graph-cpu-parity-v1",
        "passed": passed,
        "fixed_retained": fixed_retained,
        "fixed_scientific": fixed_scientific,
        "statistical_retained": statistical_retained,
        "statistical_scientific": statistical_scientific,
        "thresholds": {
            "intermediate_rtol": FIXED_INTERMEDIATE_RTOL,
            "intermediate_atol": FIXED_INTERMEDIATE_ATOL,
            "final_rtol": FINAL_RESPONSE_RTOL,
            "final_atol": FINAL_RESPONSE_ATOL,
            "final_relative_l2": FINAL_RESPONSE_RELATIVE_L2,
            "statistical_mean_margin": "max(0.01,0.10*pooled_SD)",
            "statistical_covariance_relative_frobenius": 0.15,
        },
        "elapsed_seconds": perf_counter() - started,
        "cuda_parity_claimed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cpu_parity.json", document)
    print(f"[graph-parity 4/4] passed={passed} output={output / 'cpu_parity.json'}")
    if not passed:
        raise RuntimeError("Prompt 7 CPU parity gate failed")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "graph_forward" / "prompt7_parity",
    )
    args = parser.parse_args()
    execute(args.output)


if __name__ == "__main__":
    main()
