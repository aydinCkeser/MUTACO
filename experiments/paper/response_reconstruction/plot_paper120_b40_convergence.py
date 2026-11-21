r"""Replay and plot Paper-120 B40 MUTACO convergence diagnostics.

This utility is deliberately written as a readable PyCharm ``#%%`` workflow.
Ordinary execution only loads already-produced diagnostic JSON records and
plots them.  Scientific replay is disabled unless both execution controls
below are edited deliberately.  A replay uses the canonical frozen target,
executed B40 configuration, named randomness, and 16+24 call allocation.

The replay artifact retains the ordered physical observations needed for a
Figure-5-style best-so-far convergence plot.  As in Bandits-v4 Figure 5, the
plot excludes the initial design and numbers only the sequential BO rounds.
It also queries the live terminal GP at MUTACO's final recommendation and
records latent posterior covariances and standard deviations in standardized
and physical units.
"""

from __future__ import annotations

#%%
# ---------------------------------------------------------------------------
# Imports and easy-to-edit choices.
# ---------------------------------------------------------------------------

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mutaco import RunnerStatus, ScientificIdentity, SequentialRunResult
from experiments.benchmarking.artifacts import write_pretty_json
from experiments.benchmarking.methods.mutaco_reference import run_mutaco_target
from experiments.benchmarking.simulator_adapters.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)

# Start with one representative target.  Use tuple(range(120)) only for an
# explicitly authorized full-population replay.
TARGET_INDICES = (98,)

# Two deliberate edits are required before any scientific simulator call.
EXECUTE_SCIENTIFIC_REPLAY = False
SCIENTIFIC_REPLAY_CONFIRMATION = ""
REQUIRED_CONFIRMATION = "I_AUTHORIZE_PAPER120_B40_DIAGNOSTIC_REPLAY"

PAPER120_INTRAOP_THREADS = 8
PAPER120_INTEROP_THREADS = 1
INITIAL_POINT_COUNT = 16
TOTAL_INFERENCE_CALLS = 40
BO_ITERATION_COUNT = TOTAL_INFERENCE_CALLS - INITIAL_POINT_COUNT
RECOMMENDATION_TOLERANCE = 1.0e-5

CONFIGURATION_FILE = REPO / "results/paper/graph_paper120/paper120/executed_mutaco_configuration.json"
COHORT = REPO / "experiments/graph_inversion/frozen_targets/paper120_v3"
CANONICAL_RECORDS = REPO / "results/paper/graph_paper120/paper120/paired_records.jsonl"
OUTPUT_DIRECTORY = REPO / "experiments/paper/exploration/outputs/paper120_b40_convergence"
FORMATS = ("pdf", "svg")


#%%
# ---------------------------------------------------------------------------
# Small loading and validation helpers.  These perform no simulator calls.
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_canonical_records(path: Path = CANONICAL_RECORDS) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index = int(row["target_position"])
            if index in records:
                raise ValueError(f"duplicate canonical target position {index}")
            records[index] = row
    if set(records) != set(range(120)):
        raise ValueError("canonical Paper-120 records must contain target positions 0..119")
    return records


def load_target(target_index: int) -> dict[str, Any]:
    if not 0 <= target_index < 120:
        raise ValueError("target index must be between 0 and 119")
    target = _read_json(COHORT / f"target_{target_index:03d}.json")
    if int(target["target_index"]) != target_index:
        raise ValueError("frozen target index differs from its filename")
    return target


def require_scientific_replay_authorization(
    execute: bool = EXECUTE_SCIENTIFIC_REPLAY,
    confirmation: str = SCIENTIFIC_REPLAY_CONFIRMATION,
) -> None:
    """Reject scientific execution unless both explicit local gates are open."""

    if not execute or confirmation != REQUIRED_CONFIRMATION:
        raise RuntimeError(
            "Paper-120 B40 scientific replay is disabled. Set "
            "EXECUTE_SCIENTIFIC_REPLAY=True and copy REQUIRED_CONFIRMATION "
            "into SCIENTIFIC_REPLAY_CONFIRMATION only after authorization."
        )


def _configure_historical_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(PAPER120_INTRAOP_THREADS)
    torch.set_num_threads(PAPER120_INTRAOP_THREADS)
    try:
        torch.set_num_interop_threads(PAPER120_INTEROP_THREADS)
    except RuntimeError:
        # PyCharm may already have initialized the inter-op pool.  Accept only
        # the historical value; any other layout is scientifically different.
        if torch.get_num_interop_threads() != PAPER120_INTEROP_THREADS:
            raise


#%%
# ---------------------------------------------------------------------------
# Convert one completed live run into a compact diagnostic document.
# This is model-side post-processing and makes no simulator calls.
# ---------------------------------------------------------------------------

def diagnostic_document(
    *,
    target: Mapping[str, Any],
    canonical_record: Mapping[str, Any],
    run_result: SequentialRunResult,
) -> dict[str, Any]:
    if run_result.status is not RunnerStatus.COMPLETED:
        raise ValueError(f"MUTACO replay did not complete: {run_result.status.value}")
    if run_result.terminal_surrogate is None or run_result.transform is None:
        raise ValueError("completed MUTACO replay lacks its terminal GP or transform")
    if run_result.recommendation is None:
        raise ValueError("completed MUTACO replay lacks a terminal recommendation")
    if len(run_result.observations) != TOTAL_INFERENCE_CALLS:
        raise ValueError("Paper-120 B40 replay must contain exactly 40 observations")

    target_response = torch.tensor(
        target["target_response"],
        dtype=run_result.transform.dtype,
        device=run_result.transform.device,
    ).contiguous()
    ordered_errors: list[float] = []
    best_so_far: list[float] = []
    observations: list[dict[str, Any]] = []
    current_best = float("inf")
    for evaluation_index, observation in enumerate(run_result.observations, start=1):
        error = float((observation.response_physical - target_response).abs().sum().detach())
        current_best = min(current_best, error)
        ordered_errors.append(error)
        best_so_far.append(current_best)
        observations.append(
            {
                "evaluation_index": evaluation_index,
                "phase": observation.phase.value,
                "logical_index": observation.logical_index,
                "observation_id": observation.observation_id,
                "theta": observation.theta.detach().cpu().tolist(),
                "response_physical": observation.response_physical.detach().cpu().tolist(),
                "target_error_l1": error,
                "best_so_far_target_error_l1": current_best,
            }
        )

    recommended_theta = run_result.recommendation.selected_theta.detach().contiguous()
    posterior = run_result.terminal_surrogate.posterior(recommended_theta.unsqueeze(0))
    covariance_standardized = posterior.covariance[0].detach().contiguous()
    covariance_physical = run_result.transform.covariance_to_physical(
        posterior.covariance,
        response_names=run_result.transform.metadata.response_names,
    )[0].detach().contiguous()
    standardized_variance = torch.diagonal(covariance_standardized)
    physical_variance = torch.diagonal(covariance_physical)
    if bool((standardized_variance < 0).any() or (physical_variance < 0).any()):
        raise ValueError("terminal GP covariance has a negative marginal variance")
    gp_std_standardized = torch.sqrt(standardized_variance)
    gp_std_physical = torch.sqrt(physical_variance)
    if gp_std_physical.numel() != 20:
        raise ValueError("Paper-120 terminal GP must have 20 response tasks")

    gp_std_omega_gauge0 = gp_std_physical[:10].reshape(5, 2)[:, 0]
    gp_std_M_gauge0 = gp_std_physical[10:].reshape(5, 2)[:, 0]
    canonical_theta = torch.tensor(
        canonical_record["methods"]["mutaco"]["recommended_theta"],
        dtype=recommended_theta.dtype,
        device=recommended_theta.device,
    )
    maximum_theta_drift = float((recommended_theta - canonical_theta).abs().max().detach())

    return {
        "schema_version": "paper120-b40-mutaco-convergence-diagnostic-v1",
        "scientific_calls_added_by_postprocessing": 0,
        "target_index": int(target["target_index"]),
        "target_record_id": str(target["target_record_id"]),
        "method": "mutaco",
        "inference_budget": TOTAL_INFERENCE_CALLS,
        "initial_point_count": INITIAL_POINT_COUNT,
        "sequential_point_count": TOTAL_INFERENCE_CALLS - INITIAL_POINT_COUNT,
        "target_error_definition": "sum_m abs(response_physical_m-target_physical_m)",
        "ordered_target_error_l1": ordered_errors,
        "best_so_far_target_error_l1": best_so_far,
        "observations": observations,
        "recommended_theta": recommended_theta.cpu().tolist(),
        "canonical_recommended_theta": canonical_theta.cpu().tolist(),
        "canonical_recommendation_max_abs_drift": maximum_theta_drift,
        "canonical_recommendation_tolerance": RECOMMENDATION_TOLERANCE,
        "canonical_recommendation_within_tolerance": (
            maximum_theta_drift <= RECOMMENDATION_TOLERANCE
        ),
        "gp_posterior_kind": "latent_observation_noise_false",
        "gp_response_order": "omega[5 forces,2 gauges],M[5 forces,2 gauges]",
        "gp_mean_standardized": posterior.mean[0].detach().cpu().tolist(),
        "gp_covariance_standardized": covariance_standardized.cpu().tolist(),
        "gp_covariance_physical": covariance_physical.cpu().tolist(),
        "gp_std_standardized": gp_std_standardized.cpu().tolist(),
        "gp_std_physical": gp_std_physical.cpu().tolist(),
        "gp_std_omega_gauge0": gp_std_omega_gauge0.cpu().tolist(),
        "gp_std_M_gauge0": gp_std_M_gauge0.cpu().tolist(),
        "mean_gp_std_omega_gauge0": float(gp_std_omega_gauge0.mean().cpu()),
        "mean_gp_std_M_gauge0": float(gp_std_M_gauge0.mean().cpu()),
        "logical_accounting": dict(run_result.logical_accounting),
        "physical_accounting": dict(run_result.physical_accounting),
    }


#%%
# ---------------------------------------------------------------------------
# Explicitly gated canonical replay of one frozen target.
# ---------------------------------------------------------------------------

def replay_target(
    target_index: int,
    *,
    execute: bool = EXECUTE_SCIENTIFIC_REPLAY,
    confirmation: str = SCIENTIFIC_REPLAY_CONFIRMATION,
) -> dict[str, Any]:
    require_scientific_replay_authorization(execute, confirmation)
    _configure_historical_threads()

    configuration = _read_json(CONFIGURATION_FILE)
    if (
        int(configuration["simulation_budget_per_target"]) != TOTAL_INFERENCE_CALLS
        or int(configuration["runner"]["initial_points"]) != INITIAL_POINT_COUNT
        or int(configuration["runner"]["sequential_rounds"])
        != TOTAL_INFERENCE_CALLS - INITIAL_POINT_COUNT
    ):
        raise ValueError("executed MUTACO configuration is not canonical Paper-120 B40")

    target = load_target(target_index)
    canonical_record = load_canonical_records()[target_index]
    if canonical_record["target_record_identity"] != target["target_record_id"]:
        raise ValueError("canonical result and frozen target identities differ")

    captured: list[SequentialRunResult] = []
    row = run_mutaco_target(
        adapter=GraphScientificD4Adapter(),
        index=target_index,
        theta_star=torch.tensor(target["theta_star"], dtype=torch.float32),
        config=configuration,
        campaign_science=ScientificIdentity(str(target["target_family_id"])),
        device=torch.device("cpu"),
        dtype=torch.float32,
        supplied_target_response=torch.tensor(target["target_response"], dtype=torch.float32),
        supplied_target_seed=int(target["target_generation_seed"]),
        supplied_target_identity=str(target["target_record_id"]),
        perform_held_out_assessment=False,
        diagnostic_result_callback=captured.append,
    )
    if len(captured) != 1:
        raise RuntimeError("canonical replay did not expose exactly one live run result")
    document = diagnostic_document(
        target=target,
        canonical_record=canonical_record,
        run_result=captured[0],
    )
    document["serialized_method_status"] = row["status"]
    document["serialized_inference_calls"] = row["inference_calls"]
    return document


def write_diagnostic(document: Mapping[str, Any], output: Path = OUTPUT_DIRECTORY) -> Path:
    if document.get("canonical_recommendation_within_tolerance") is not True:
        raise ValueError(
            "refusing to write a canonical diagnostic because the replay "
            "recommendation differs from the accepted Paper-120 record"
        )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"target_{int(document['target_index']):03d}.json"
    if path.exists():
        raise FileExistsError(f"diagnostic record is append-only and already exists: {path}")
    write_pretty_json(path, document)
    return path


#%%
# ---------------------------------------------------------------------------
# Load completed diagnostics and build the Figure-5-style convergence view.
# No simulator or GP calls occur below this line.
# ---------------------------------------------------------------------------

def load_diagnostics(output: Path = OUTPUT_DIRECTORY) -> list[dict[str, Any]]:
    if not output.is_dir():
        return []
    records = [_read_json(path) for path in sorted(output.glob("target_*.json"))]
    indices = [int(record["target_index"]) for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("diagnostic directory contains duplicate target indices")
    for record in records:
        if (
            record.get("schema_version")
            != "paper120-b40-mutaco-convergence-diagnostic-v1"
            or len(record.get("ordered_target_error_l1", ())) != TOTAL_INFERENCE_CALLS
            or record.get("canonical_recommendation_within_tolerance") is not True
        ):
            raise ValueError("invalid Paper-120 B40 convergence diagnostic record")
    return records


def bo_convergence_trajectory(record: Mapping[str, Any]) -> torch.Tensor:
    """Return Figure-5-style cumulative best error over BO iterations only."""

    ordered_errors = record.get("ordered_target_error_l1", ())
    if len(ordered_errors) != TOTAL_INFERENCE_CALLS:
        raise ValueError("Paper-120 B40 diagnostics require 40 ordered errors")
    bo_errors = torch.tensor(
        ordered_errors[INITIAL_POINT_COUNT:],
        dtype=torch.float64,
    )
    if bo_errors.numel() != BO_ITERATION_COUNT or not bool(
        torch.isfinite(bo_errors).all()
    ):
        raise ValueError("Paper-120 B40 diagnostics require 24 finite BO errors")
    return torch.cummin(bo_errors, dim=0).values


def convergence_statistics(
    records: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("at least one diagnostic record is required")
    trajectories = torch.stack([bo_convergence_trajectory(record) for record in records])
    mean = trajectories.mean(dim=0)
    std = (
        trajectories.std(dim=0, unbiased=True)
        if trajectories.shape[0] > 1
        else torch.zeros_like(mean)
    )
    return trajectories, mean, std


def plot_convergence(
    records: Sequence[Mapping[str, Any]],
    output: Path = OUTPUT_DIRECTORY,
) -> plt.Figure:
    trajectories, mean, std = convergence_statistics(records)
    bo_iterations = torch.arange(1, BO_ITERATION_COUNT + 1)
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    if len(records) == 1:
        target_index = int(records[0]["target_index"])
        axis.plot(
            bo_iterations,
            trajectories[0],
            color="#0072B2",
            linewidth=2.2,
            label=f"Target {target_index}",
        )
    else:
        for trajectory in trajectories:
            axis.plot(
                bo_iterations,
                trajectory,
                color="gray",
                alpha=0.45,
                linewidth=0.7,
            )
        lower = torch.maximum(mean - std, trajectories.min(dim=0).values)
        axis.fill_between(
            bo_iterations,
            lower,
            mean + std,
            color="#0072B2",
            alpha=0.20,
            label=r"Mean $\pm 1\sigma$",
        )
        axis.plot(
            bo_iterations,
            mean,
            color="#0072B2",
            linewidth=2.2,
            label="Mean",
        )
    axis.set(
        xlabel=r"BO iteration $t$",
        ylabel=r"Best $E_t=\|\mathbf{y}(\boldsymbol{\theta}_t)-\mathbf{y}^\star\|_1$",
        xlim=(1, BO_ITERATION_COUNT),
        yscale="log",
    )
    axis.grid(True, which="major", linestyle="--", alpha=0.3, linewidth=0.5)
    axis.legend(frameon=False)

    output.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        figure.savefig(output / f"paper120_b40_convergence.{extension}")
    return figure


def print_gp_summary(record: Mapping[str, Any]) -> None:
    gp_std_omega = torch.tensor(record["gp_std_omega_gauge0"])
    gp_std_M = torch.tensor(record["gp_std_M_gauge0"])
    print(f"target {int(record['target_index']):03d}")
    print("mutaco_gp_std_omega:", gp_std_omega)
    print("mutaco_gp_std_M:", gp_std_M)
    print(
        "mean_gp_std_omega:",
        record.get("mean_gp_std_omega_gauge0", float(gp_std_omega.mean())),
    )
    print(
        "mean_gp_std_M:",
        record.get("mean_gp_std_M_gauge0", float(gp_std_M.mean())),
    )
    print(
        "canonical recommendation max drift:",
        record["canonical_recommendation_max_abs_drift"],
    )


#%%
# ---------------------------------------------------------------------------
# Run this file.  With the default controls, this section only loads/plots.
# ---------------------------------------------------------------------------

def main() -> None:
    if EXECUTE_SCIENTIFIC_REPLAY:
        for target_index in TARGET_INDICES:
            document = replay_target(target_index)
            written = write_diagnostic(document)
            print("Wrote diagnostic:", written)

    records = load_diagnostics()
    if not records:
        print("No completed Paper-120 B40 convergence diagnostics found.")
        print("Scientific replay remains disabled; no simulator calls were made.")
        return
    for record in records:
        print_gp_summary(record)
    plot_convergence(records)
    plt.show()


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.exploration.plot_paper120_b40_convergence")


if __name__ == "__main__":
    main()
