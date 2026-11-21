"""Legacy physical reconstruction notebook; import does not execute the cells.

Explicit script execution uses the checkout root and retains the original
scientific body, including physical evaluations and output writes. It is not
an offline renderer or a portable tutorial. See the exploration README for
the frozen-response renderer. Scientific execution requires separate approval.
"""

if __name__ == "__main__":
    import json
    from pathlib import Path

    import torch


    #%%
    import json
    from pathlib import Path
    import torch

    repo = Path(__file__).resolve().parents[3]

    assert repo.is_dir()
    print("Repository:", repo)

    #%%
    # Rank targets where MUTACO has low L1 error and outperforms NPE.
    records_file = repo / "results/paper/graph_paper120/paper120/paired_records.jsonl"

    ranked_targets = []

    with records_file.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            mutaco_l1 = row["methods"]["mutaco"]["response_error_l1_sum"]
            npe_l1 = row["methods"]["sbi_npe"]["response_error_l1_sum"]

            # A high score means NPE error is large relative to MUTACO error.
            score = npe_l1 / mutaco_l1

            ranked_targets.append(
                {
                    "index": row["target_position"],
                    "score": score,
                    "mutaco_l1": mutaco_l1,
                    "npe_l1": npe_l1,
                }
            )

    ranked_targets.sort(
        key=lambda row: row["score"],
        reverse=True,
    )

    print(" rank | target | score | MUTACO L1 | NPE L1")
    print("-" * 49)

    for rank, row in enumerate(ranked_targets[:-1], start=1):
        print(
            f"{rank:5d} | "
            f"{row['index']:6d} | "
            f"{row['score']:5.2f} | "
            f"{row['mutaco_l1']:9.4f} | "
            f"{row['npe_l1']:6.4f}"
        )

    #%%
    # Choose one target from the Paper-120 cohort.
    target_index =98  # Change this to any integer from 0 to 119.


    target_file = repo / "experiments/graph_inversion/frozen_targets/paper120_v3" / f"target_{target_index:03d}.json"

    with target_file.open(encoding="utf-8") as file:
        target = json.load(file)

    theta_star = torch.tensor(target["theta_star"], dtype=torch.float32)
    y_star = torch.tensor(target["target_response"], dtype=torch.float32)

    print("Target index:", target_index)
    print("theta*:", theta_star)
    print("y*:", y_star)

    #%%

    from mutaco import SimulationContext
    from mutaco_graph import (
        GraphSimulationConfig,
        OverdampedScientificSimulator,
        ScientificConditionGrid,
    )


    # Reproduce the Paper-120 simulator configuration.
    with (
        repo / "experiments/graph_inversion/configs/paper/paper120_v3_target_config.json"
    ).open(encoding="utf-8") as file:
        simulator_settings = json.load(file)["simulator"]

    config = GraphSimulationConfig(
        model_identity=simulator_settings["identifier"],
        parameter_profile_id=simulator_settings["parameter_profile"],
        response_profile_id=simulator_settings["response_profile"],
        node_count=simulator_settings["node_count"],
        batch_size=simulator_settings["batch_size"],       # 80 graphs
        repeat_size=simulator_settings["repeat_size"],     # 2 thermal repeats
        temperature=simulator_settings["temperature"],
        dt=simulator_settings["dt"],
        burn_in_steps=simulator_settings["burn_in_steps"],
        sample_steps=simulator_settings["sample_steps"],
        sample_stride=simulator_settings["sample_stride"],
        device="cpu",
        dtype=torch.float32,
    )

    # Two hundred force values from 0 to 5, with gauge fixed at zero.
    number_of_forces = 200

    dense_grid = ScientificConditionGrid(
        force_ticks=tuple(range(number_of_forces)),
        force_step=5.0 / (number_of_forces - 1),
        gauges=(0.0,),
        identifier="paper120_dense_force_gauge_zero",
    )

    # Replay the selected target's named random coordinate.
    target_context = SimulationContext(
        target["target_family_id"],
        "target",
        target_index,
        0,
        target["target_generation_seed"],
    )

    simulator = OverdampedScientificSimulator(config)

    dense_run, dense_metadata = simulator.evaluate_condition_grid(
        theta_star,
        target_context,
        dense_grid,
        max_conditions_per_chunk=10,
    )

    forces = torch.tensor(dense_grid.forces)

    # raw tensors are [force, graph, repeat].
    # Average the two repeats, but retain all 80 graph realizations.
    omega = dense_run.raw_omega.mean(dim=-1)
    M = dense_run.raw_synchronization.mean(dim=-1)

    print("forces:", forces.shape)                    # [200]
    print("omega:", omega.shape)                      # [200, 80]
    print("M:", M.shape)  # [200, 80]


    #%%
    import json
    import matplotlib.pyplot as plt
    import torch

    from experiments.paper.common.population_v3 import (
        load_population_declaration_v3,
    )
    from experiments.paper.common.protocol_v3 import (
        paired_assessment_context,
    )
    from mutaco_graph.scientific import CANONICAL_CONDITION_GRID


    # ------------------------------------------------------------------
    # Load the B40 MUTACO and NPE recommendations for the selected target.
    # ------------------------------------------------------------------

    records_file = repo / "results/paper/graph_paper120/paper120/paired_records.jsonl"

    result = None

    with records_file.open(encoding="utf-8") as file:
        for line in file:
            candidate = json.loads(line)

            if candidate["target_position"] == target_index:
                result = candidate
                break

    if result is None:
        raise ValueError(f"No B40 result found for target {target_index}")

    theta_mutaco = torch.tensor(
        result["methods"]["mutaco"]["recommended_theta"],
        dtype=torch.float32,
    )

    theta_npe = torch.tensor(
        result["methods"]["sbi_npe"]["recommended_theta"],
        dtype=torch.float32,
    )

    #%%
    # ------------------------------------------------------------------
    # Construct the shared held-out assessment randomness.
    # ------------------------------------------------------------------

    declaration_file = (
        repo / "results/paper/graph_paper120/paper120/executed_declaration.json"
    )

    with declaration_file.open(encoding="utf-8") as file:
        declaration = load_population_declaration_v3(json.load(file))

    assessment = paired_assessment_context(
        declaration.comparison,
        target["target_record_id"],
    )

    assessment_context = SimulationContext(
        assessment["family_identity"],
        "paired_held_out_assessment",
        target_index,
        assessment["replicate"],
        assessment["evaluation_seed"],
    )

    #%%
    # ------------------------------------------------------------------
    # Dense reconstruction curves.
    # ------------------------------------------------------------------

    mutaco_run, _ = simulator.evaluate_condition_grid(
        theta_mutaco,
        assessment_context,
        dense_grid,
        max_conditions_per_chunk=10,
    )

    npe_run, _ = simulator.evaluate_condition_grid(
        theta_npe,
        assessment_context,
        dense_grid,
        max_conditions_per_chunk=10,
    )

    # Average over the 80 graphs and two thermal repeats.
    mutaco_omega = mutaco_run.raw_omega.mean(dim=(1, 2))
    mutaco_M = mutaco_run.raw_synchronization.mean(dim=(1, 2))

    npe_omega = npe_run.raw_omega.mean(dim=(1, 2))
    npe_M = npe_run.raw_synchronization.mean(dim=(1, 2))

    #%%
    # ------------------------------------------------------------------
    # Explicit evaluations at the five sparse inference forces.
    # ------------------------------------------------------------------

    sparse_forces = torch.tensor(
        [0.0, 1.25, 2.5, 3.75, 5.0],
        dtype=torch.float32,
    )

    mutaco_sparse_run, _ = simulator.evaluate_condition_grid(
        theta_mutaco,
        assessment_context,
        CANONICAL_CONDITION_GRID,
        max_conditions_per_chunk=10,
    )

    npe_sparse_run, _ = simulator.evaluate_condition_grid(
        theta_npe,
        assessment_context,
        CANONICAL_CONDITION_GRID,
        max_conditions_per_chunk=10,
    )

    # The canonical grid alternates gauge zero and gauge pi.
    # The 0::2 slice therefore selects gauge zero.
    mutaco_sparse_omega = (
        mutaco_sparse_run.raw_omega[0::2].mean(dim=(1, 2))
    )
    mutaco_sparse_M = (
        mutaco_sparse_run.raw_synchronization[0::2].mean(dim=(1, 2))
    )

    npe_sparse_omega = (
        npe_sparse_run.raw_omega[0::2].mean(dim=(1, 2))
    )
    npe_sparse_M = (
        npe_sparse_run.raw_synchronization[0::2].mean(dim=(1, 2))
    )




    #%%
    # ------------------------------------------------------------------
    # Target statistics and original sparse target data.
    # ------------------------------------------------------------------

    # omega and M have shape [dense force, 80 graphs].
    target_omega_mean = omega.mean(dim=1)
    target_omega_std = omega.std(dim=1, unbiased=True)

    target_M_mean = M.mean(dim=1)
    target_M_std = M.std(dim=1, unbiased=True)

    # The stored M=20 response alternates gauge zero and gauge pi.
    true_sparse_omega = y_star[:10].reshape(5, 2)[:, 0]
    true_sparse_M = y_star[10:].reshape(5, 2)[:, 0]



    #%%

    import matplotlib.pyplot as plt

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",

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
    })

    #%% Forward figure
    fig_forward, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for ax, values, ylabel in [
        (axes[0], omega, r"$\langle\bar{\omega}\rangle$"),
        (axes[1], M, r"$\langle M\rangle$"),
    ]:
        # Individual graph realizations.
        for graph_index in range(values.shape[1]):
            ax.plot(
                forces,
                values[:, graph_index],
                color="gray",
                alpha=0.15,
                linewidth=0.6,
            )

        # Mean and standard deviation across the 80 graphs.
        mean = values.mean(dim=1)
        std = values.std(dim=1, unbiased=True)

        ax.fill_between(
            forces,
            mean - std,
            mean + std,
            color="#56B4E9",
            alpha=0.25,
            label=r"$\pm 1\sigma$ across graphs",
        )

        ax.plot(
            forces,
            mean,
            color="#0072B2",
            linewidth=2,
            label="Ensemble mean",
        )

        ax.set_xlabel(r"$F$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)


    axes[1].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    plt.show()


    #%%
    # ------------------------------------------------------------------
    # Reconstruction figure.
    # ------------------------------------------------------------------

    fig_reconstruction, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            target_omega_mean,
            target_omega_std,
            true_sparse_omega,
            mutaco_omega,
            mutaco_sparse_omega,
            npe_omega,
            npe_sparse_omega,
            r"$\langle\bar{\omega}\rangle$",
        ),
        (
            axes[1],
            target_M_mean,
            target_M_std,
            true_sparse_M,
            mutaco_M,
            mutaco_sparse_M,
            npe_M,
            npe_sparse_M,
            r"$\langle M\rangle$",
        ),
    ]

    for (
        ax,
        target_mean,
        target_std,
        true_sparse,
        mutaco_dense,
        mutaco_sparse,
        npe_dense,
        npe_sparse,
        ylabel,
    ) in panels:

        # Target uncertainty across the 80 graph realizations.
        ax.fill_between(
            forces,
            target_mean - target_std,
            target_mean + target_std,
            color="lightgray",
            alpha=0.7,
            label=r"True mean $\pm 1\sigma$",
        )

        # Dense target ensemble mean.
        ax.plot(
            forces,
            target_mean,
            color="black",
            linewidth=2,
        )

        # Original sparse target data: large open circles.
        ax.plot(
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

        # Dense MUTACO reconstruction: dotted blue line.
        ax.plot(
            forces,
            mutaco_dense,
            color="#0072B2",
            linestyle=":",
            linewidth=2,
        )

        # Explicit MUTACO values at sparse forces: filled blue squares.
        ax.plot(
            sparse_forces,
            mutaco_sparse,
            linestyle="none",
            marker="s",
            markersize=7,
            markerfacecolor="#0072B2",
            markeredgecolor="#0072B2",
            label="MUTACO",
        )

        # Dense NPE reconstruction: dotted orange line.
        ax.plot(
            forces,
            npe_dense,
            color="#E66B2C",
            linestyle=":",
            linewidth=2,
        )

        # Explicit NPE values at sparse forces: orange triangles.
        ax.plot(
            sparse_forces,
            npe_sparse,
            linestyle="none",
            marker="^",
            markersize=8,
            markerfacecolor="#E66B2C",
            markeredgecolor="black",
            markeredgewidth=0.7,
            label="NPE",
        )

        ax.set_xlabel(r"$F$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    axes[1].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    plt.show()

    #%%
    # ------------------------------------------------------------------
    # Save the  figures.
    # ------------------------------------------------------------------

    output_directory = (
        repo
        / "experiments/paper/exploration/outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_name = (
        f"paper120_target_{target_index:03d}_reconstruction"
    )



    fig_reconstruction.savefig(
        output_directory / f"{figure_name}.pdf",
        bbox_inches="tight",
    )

    fig_reconstruction.savefig(
        output_directory / f"{figure_name}.svg",
        bbox_inches="tight",
    )

    figure_name = (
        f"paper120_target_{target_index:03d}_forward"
    )



    fig_forward.savefig(
        output_directory / f"{figure_name}.pdf",
        bbox_inches="tight",
    )

    fig_forward.savefig(
        output_directory / f"{figure_name}.svg",
        bbox_inches="tight",
    )


    print("Saved figures to:", output_directory)

    #%%
    # ------------------------------------------------------------------
    # MUTACO terminal-GP uncertainty at the recommended theta.
    # ------------------------------------------------------------------

    posterior = result.terminal_surrogate.posterior(
        theta_mutaco.unsqueeze(0)
    )

    covariance_physical = result.transform.covariance_to_physical(
        posterior.covariance,
        response_names=result.transform.metadata.response_names,
    )

    mutaco_gp_std = torch.sqrt(
        torch.diagonal(covariance_physical[0])
    )

    # The response ordering is:
    # omega[5 forces, 2 gauges], M[5 forces, 2 gauges].
    mutaco_gp_std_omega = mutaco_gp_std[:10].reshape(5, 2)[:, 0]
    mutaco_gp_std_M = mutaco_gp_std[10:].reshape(5, 2)[:, 0]

    print("mutaco_gp_std_omega:", mutaco_gp_std_omega)
    print("mutaco_gp_std_M:", mutaco_gp_std_M)