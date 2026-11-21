"""Reproduce the target-98 forward and MUTACO-only review figures.

This is a visualization replay of the existing bandits_v5 construction.  It
does not create benchmark evidence, recommendations, or population results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from experiments.paper.common.population_v3 import load_population_declaration_v3
from experiments.paper.common.protocol_v3 import paired_assessment_context
from mutaco import SimulationContext
from mutaco_graph import (
    GraphSimulationConfig,
    OverdampedScientificSimulator,
    ScientificConditionGrid,
)
from mutaco_graph.scientific import CANONICAL_CONDITION_GRID

from experiments.paper.exploration.target098_response_archive import (
    ARCHIVE_NAME,
    MANIFEST_NAME,
    write_archive,
)


TARGET_POSITION = 98
TARGET_RECORD_ID = (
    "frozen-target-record-v2-"
    "c1f2caab22dc5c7f72782e379c38ce9ee06297491b6174c6306303cb9f35a7c5"
)
SIMULATOR_CONFIGURATION_ID = (
    "graph-target-simulator-configuration-v1-"
    "ff70b08584d86189e7464b20602142d9faeca01f588c0d691de6cdf2c7568a08"
)
EXPECTED_MUTACO_THETA = (
    1.9519236087799072,
    1.1798672676086426,
    1.1263821125030518,
    0.409557968378067,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs(repo: Path) -> tuple[dict, dict, object]:
    target = _load_json(repo / "experiments/graph_inversion/frozen_targets/paper120_v3/target_098.json")
    if target["target_index"] != TARGET_POSITION:
        raise RuntimeError("Target file does not contain Paper-120 position 98")
    if target["target_record_id"] != TARGET_RECORD_ID:
        raise RuntimeError("Target-98 record identity mismatch")
    if target["simulator_configuration_identity"] != SIMULATOR_CONFIGURATION_ID:
        raise RuntimeError("Target-98 simulator configuration identity mismatch")

    result = None
    records_path = repo / "results/paper/graph_paper120/paper120/paired_records.jsonl"
    with records_path.open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate["target_position"] == TARGET_POSITION:
                result = candidate
                break
    if result is None:
        raise RuntimeError("No historical B40 record found for target 98")
    if result["target_record_identity"] != TARGET_RECORD_ID:
        raise RuntimeError("Target-98 B40 result identity mismatch")
    mutaco = result["methods"]["mutaco"]
    if mutaco["status"] != "completed":
        raise RuntimeError("Historical target-98 MUTACO result is not complete")
    if tuple(mutaco["recommended_theta"]) != EXPECTED_MUTACO_THETA:
        raise RuntimeError("Historical target-98 MUTACO recommendation mismatch")

    declaration = load_population_declaration_v3(
        _load_json(repo / "results/paper/graph_paper120/paper120/executed_declaration.json")
    )
    return target, mutaco, declaration


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage[T1]{fontenc}",
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def main(*, archive_directory: Path | None = None, figure_directory: Path | None = None) -> None:
    if archive_directory is not None and figure_directory is None:
        raise ValueError("--figure-directory is required with --archive-directory")
    repo = Path(__file__).resolve().parents[3]
    if archive_directory is not None:
        for destination in (archive_directory, figure_directory):
            resolved = destination.resolve()
            if resolved == repo or repo in resolved.parents:
                raise ValueError("archival outputs must be outside the repository")
        archival_outputs = (
            archive_directory / ARCHIVE_NAME,
            archive_directory / MANIFEST_NAME,
            figure_directory / "paper120_target_098_forward.pdf",
            figure_directory / "paper120_target_098_forward.png",
            figure_directory / "paper120_target_098_mutaco_reconstruction.pdf",
            figure_directory / "paper120_target_098_mutaco_reconstruction.png",
        )
        if any(path.exists() for path in archival_outputs):
            raise FileExistsError("archival output already exists; refusing to overwrite")
    target, mutaco_result, declaration = _load_inputs(repo)

    theta_star = torch.tensor(target["theta_star"], dtype=torch.float32)
    target_response = torch.tensor(target["target_response"], dtype=torch.float32)
    theta_mutaco = torch.tensor(
        mutaco_result["recommended_theta"], dtype=torch.float32
    )

    target_config = _load_json(
        repo
        / "experiments/graph_inversion/configs/paper"
        / "paper120_v3_target_config.json"
    )
    simulator_settings = target_config["simulator"]
    config = GraphSimulationConfig(
        model_identity=simulator_settings["identifier"],
        parameter_profile_id=simulator_settings["parameter_profile"],
        response_profile_id=simulator_settings["response_profile"],
        node_count=simulator_settings["node_count"],
        batch_size=simulator_settings["batch_size"],
        repeat_size=simulator_settings["repeat_size"],
        temperature=simulator_settings["temperature"],
        dt=simulator_settings["dt"],
        burn_in_steps=simulator_settings["burn_in_steps"],
        sample_steps=simulator_settings["sample_steps"],
        sample_stride=simulator_settings["sample_stride"],
        device="cpu",
        dtype=torch.float32,
    )

    number_of_forces = 200
    dense_grid = ScientificConditionGrid(
        force_ticks=tuple(range(number_of_forces)),
        force_step=5.0 / (number_of_forces - 1),
        gauges=(0.0,),
        identifier="paper120_dense_force_gauge_zero",
    )
    forces = torch.tensor(dense_grid.forces)
    sparse_forces = torch.tensor(
        [0.0, 1.25, 2.5, 3.75, 5.0], dtype=torch.float32
    )

    target_context = SimulationContext(
        target["target_family_id"],
        "target",
        TARGET_POSITION,
        0,
        target["target_generation_seed"],
    )
    assessment = paired_assessment_context(
        declaration.comparison,
        target["target_record_id"],
    )
    assessment_context = SimulationContext(
        assessment["family_identity"],
        "paired_held_out_assessment",
        TARGET_POSITION,
        assessment["replicate"],
        assessment["evaluation_seed"],
    )

    print("Visualization/replay only; not new scientific benchmark evidence.")
    print("Target position:", TARGET_POSITION)
    print("Target record identity:", target["target_record_id"])
    print("theta_star:", target["theta_star"])
    print("MUTACO recommended theta:", mutaco_result["recommended_theta"])
    print("Target context:", target_context)
    print("MUTACO held-out context:", assessment_context)
    print("Simulator configuration identity:", SIMULATOR_CONFIGURATION_ID)

    simulator = OverdampedScientificSimulator(config)

    # Visualization replay 1/3: dense target response at Phi=0.
    dense_target_run, _ = simulator.evaluate_condition_grid(
        theta_star,
        target_context,
        dense_grid,
        max_conditions_per_chunk=10,
    )
    target_omega_graphs = dense_target_run.raw_omega.mean(dim=-1)
    target_M_graphs = dense_target_run.raw_synchronization.mean(dim=-1)

    # Visualization replay 2/3: dense historical MUTACO reconstruction.
    dense_mutaco_run, _ = simulator.evaluate_condition_grid(
        theta_mutaco,
        assessment_context,
        dense_grid,
        max_conditions_per_chunk=10,
    )
    mutaco_omega = dense_mutaco_run.raw_omega.mean(dim=(1, 2))
    mutaco_M = dense_mutaco_run.raw_synchronization.mean(dim=(1, 2))

    # Visualization replay 3/3: MUTACO at the canonical sparse grid.
    sparse_mutaco_run, _ = simulator.evaluate_condition_grid(
        theta_mutaco,
        assessment_context,
        CANONICAL_CONDITION_GRID,
        max_conditions_per_chunk=10,
    )
    mutaco_sparse_omega = sparse_mutaco_run.raw_omega[0::2].mean(dim=(1, 2))
    mutaco_sparse_M = sparse_mutaco_run.raw_synchronization[0::2].mean(dim=(1, 2))

    target_omega_mean = target_omega_graphs.mean(dim=1)
    target_omega_std = target_omega_graphs.std(dim=1, unbiased=True)
    target_M_mean = target_M_graphs.mean(dim=1)
    target_M_std = target_M_graphs.std(dim=1, unbiased=True)
    true_sparse_omega = target_response[:10].reshape(5, 2)[:, 0]
    true_sparse_M = target_response[10:].reshape(5, 2)[:, 0]

    if archive_directory is not None:
        arrays = {
            "forces": forces.numpy(),
            "sparse_forces": sparse_forces.numpy(),
            "target_omega_graphs": target_omega_graphs.numpy(),
            "target_M_graphs": target_M_graphs.numpy(),
            "target_omega_mean": target_omega_mean.numpy(),
            "target_omega_std": target_omega_std.numpy(),
            "target_M_mean": target_M_mean.numpy(),
            "target_M_std": target_M_std.numpy(),
            "mutaco_omega": mutaco_omega.numpy(),
            "mutaco_M": mutaco_M.numpy(),
            "true_sparse_omega": true_sparse_omega.numpy(),
            "true_sparse_M": true_sparse_M.numpy(),
            "mutaco_sparse_omega": mutaco_sparse_omega.numpy(),
            "mutaco_sparse_M": mutaco_sparse_M.numpy(),
        }
        provenance = {
            "target_position": TARGET_POSITION,
            "target_record_id": target["target_record_id"],
            "target_seed": target["target_generation_seed"],
            "assessment_seed": assessment["evaluation_seed"],
            "target_context": {
                "family_identity": target["target_family_id"],
                "namespace": "target", "position": TARGET_POSITION, "replicate": 0,
            },
            "assessment_context": {
                "family_identity": assessment["family_identity"],
                "namespace": "paired_held_out_assessment",
                "position": TARGET_POSITION, "replicate": assessment["replicate"],
                "coordinate": assessment["coordinate"],
            },
            "simulator_configuration_identity": SIMULATOR_CONFIGURATION_ID,
            "simulator_configuration": config.identity_document(),
            "execution_device": config.device,
            "dense_grid": dense_grid.document(),
            "canonical_grid": CANONICAL_CONDITION_GRID.document(),
            "max_conditions_per_chunk": 10,
            "evaluation_condition_counts": [200, 200, 10],
            "theta_star": target["theta_star"],
            "theta_mutaco": mutaco_result["recommended_theta"],
            "reductions": {
                "target_graph_curves": "raw_omega/raw_synchronization.mean(dim=-1)",
                "target_mean": "graph_curves.mean(dim=1)",
                "target_std": "graph_curves.std(dim=1, unbiased=True)",
                "mutaco_dense": "raw_omega/raw_synchronization.mean(dim=(1, 2))",
                "mutaco_sparse": "raw_omega/raw_synchronization[0::2].mean(dim=(1, 2))",
                "true_sparse": "target_response halves.reshape(5, 2)[:, 0]",
            },
        }
        archive, manifest = write_archive(archive_directory, arrays, provenance, repo)
        print("Frozen target-98 arrays:", archive)
        print("Frozen target-98 provenance:", manifest)

    _configure_plotting()

    # Preserve the original target forward-response figure.
    fig_forward, axes = plt.subplots(
        1, 2, figsize=(10, 4), constrained_layout=True
    )
    for axis, values, ylabel in (
        (axes[0], target_omega_graphs, r"$\langle\bar{\omega}\rangle$"),
        (axes[1], target_M_graphs, r"$\langle M\rangle$"),
    ):
        for graph_index in range(values.shape[1]):
            axis.plot(
                forces,
                values[:, graph_index],
                color="gray",
                alpha=0.15,
                linewidth=0.6,
            )
        mean = values.mean(dim=1)
        std = values.std(dim=1, unbiased=True)
        axis.fill_between(
            forces,
            mean - std,
            mean + std,
            color="#56B4E9",
            alpha=0.25,
            label=r"$\pm 1\sigma$ across graphs",
        )
        axis.plot(
            forces,
            mean,
            color="#0072B2",
            linewidth=2,
            label="Ensemble mean",
        )
        axis.set_xlabel(r"$F$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    # Preserve the original target-and-MUTACO reconstruction grammar.
    fig_reconstruction, axes = plt.subplots(
        1, 2, figsize=(10, 4), constrained_layout=True
    )
    panels = (
        (
            axes[0],
            target_omega_mean,
            target_omega_std,
            true_sparse_omega,
            mutaco_omega,
            mutaco_sparse_omega,
            r"$\langle\bar{\omega}\rangle$",
        ),
        (
            axes[1],
            target_M_mean,
            target_M_std,
            true_sparse_M,
            mutaco_M,
            mutaco_sparse_M,
            r"$\langle M\rangle$",
        ),
    )
    for (
        axis,
        target_mean,
        target_std,
        true_sparse,
        mutaco_dense,
        mutaco_sparse,
        ylabel,
    ) in panels:
        axis.fill_between(
            forces,
            target_mean - target_std,
            target_mean + target_std,
            color="lightgray",
            alpha=0.7,
            label=r"True mean $\pm 1\sigma$",
        )
        axis.plot(forces, target_mean, color="black", linewidth=2)
        axis.plot(
            sparse_forces,
            true_sparse,
            linestyle="none",
            marker="o",
            markersize=10,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=2,
            label="True",
        )
        axis.plot(
            forces,
            mutaco_dense,
            color="#0072B2",
            linestyle=":",
            linewidth=2,
        )
        axis.plot(
            sparse_forces,
            mutaco_sparse,
            linestyle="none",
            marker="s",
            markersize=7,
            markerfacecolor="#0072B2",
            markeredgecolor="#0072B2",
            label="MUTACO",
        )
        axis.set_xlabel(r"$F$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    output_directory = figure_directory or (
        repo / "experiments/paper/exploration/outputs" / "paper120_target098_mutaco_only_v1"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forward_pdf": output_directory / "paper120_target_098_forward.pdf",
        "forward_png": output_directory / "paper120_target_098_forward.png",
        "reconstruction_pdf": (
            output_directory / "paper120_target_098_mutaco_reconstruction.pdf"
        ),
        "reconstruction_png": (
            output_directory / "paper120_target_098_mutaco_reconstruction.png"
        ),
    }
    fixed_pdf_metadata = {
        "Creator": "MUTACO visualization replay",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fixed_png_metadata = {"Software": "MUTACO visualization replay"}
    fig_forward.savefig(
        outputs["forward_pdf"], bbox_inches="tight", metadata=fixed_pdf_metadata
    )
    fig_forward.savefig(
        outputs["forward_png"],
        bbox_inches="tight",
        dpi=300,
        metadata=fixed_png_metadata,
    )
    fig_reconstruction.savefig(
        outputs["reconstruction_pdf"],
        bbox_inches="tight",
        metadata=fixed_pdf_metadata,
    )
    fig_reconstruction.savefig(
        outputs["reconstruction_png"],
        bbox_inches="tight",
        dpi=300,
        metadata=fixed_png_metadata,
    )
    plt.close(fig_forward)
    plt.close(fig_reconstruction)

    print("Condition-grid visualization replays performed: 3")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-directory", type=Path,
                        help="write frozen target-98 arrays and provenance here, without overwriting")
    parser.add_argument("--figure-directory", type=Path,
                        help="write the original plot outputs here instead of the default directory")
    args = parser.parse_args()
    main(archive_directory=args.archive_directory, figure_directory=args.figure_directory)
