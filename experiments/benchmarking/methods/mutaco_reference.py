"""Simulator-independent MUTACO reference method."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Mapping

import torch

from mutaco import (
    RunnerStatus,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    SequentialRunResult,
    SequentialRunnerConfig,
    SequentialScientificIntent,
    SimulationContext,
    TerminalRecommendationContext,
    checked_evaluate,
    run_sequential,
    strict_metadata_document,
)

from ..contracts import MutacoBudget, SimulatorAdapter
from ..statistics import vector_errors
from .mutaco_projection import scientific_projection


def _graph_paper_enabled(config: Mapping[str, Any]) -> bool:
    return (
        config.get("simulator", {}).get("adapter") == "graph_scientific_d4"
        and ("proposal_effort" in config or "proposal_controls" in config)
    )


def _context(run_id: str, purpose: str, index: int, seed: int) -> SimulationContext:
    return SimulationContext(run_id, purpose, index, 0, seed)


def _intent(
    simulator: Any,
    *,
    target: torch.Tensor,
    profile: Mapping[str, Any],
    runner: Mapping[str, Any],
    dtype: torch.dtype,
    evaluation_seed_family_id: str | None = None,
    caller_initial_points: torch.Tensor | None = None,
) -> SequentialScientificIntent:
    capabilities = simulator.capabilities
    return SequentialScientificIntent(
        simulator_identity=capabilities.simulator_identity,
        simulator_definition_metadata=strict_metadata_document(
            capabilities.definition_metadata
        ),
        parameter_spec=capabilities.parameter_spec,
        response_spec=capabilities.response_spec,
        physical_target=tuple(float(value) for value in target.detach().cpu().tolist()),
        initial_design_policy=(
            "caller_supplied" if caller_initial_points is not None else "scrambled_sobol"
        ),
        initial_point_count=int(profile["initial_points"]),
        caller_initial_points=(
            None
            if caller_initial_points is None
            else tuple(
                tuple(float(value) for value in row)
                for row in caller_initial_points.detach().cpu().tolist()
            )
        ),
        sequential_rounds=int(profile["sequential_rounds"]),
        transform_settings=runner["transform"],
        gp_settings=runner["gp"],
        acquisition_settings=runner["acquisition"],
        recommendation_policy=str(
            runner.get("recommendation_policy", "final_posterior_standardized_l1")
        ),
        duplicate_policy=str(
            runner.get("duplicate_policy", "reject_exact_duplicate_v1")
        ),
        dtype=dtype,
        root_seed=int(profile["root_seed"]),
        seed_coordinate_schema={
            "initial_design": ["phase", "repeat"],
            "simulator_evaluation": ["purpose", "logical_index", "repeat"],
            "gp_fit": ["training_version"],
            "raw_candidates": ["sequential_round"],
        },
        evaluation_seed_policy=(
            "paired_coordinate_diagnostic_v1"
            if evaluation_seed_family_id is not None
            else "scientific_identity_coordinate_v1"
        ),
        evaluation_seed_family_id=evaluation_seed_family_id,
    )


def run_mutaco_target(
    *,
    adapter: SimulatorAdapter,
    index: int,
    theta_star: torch.Tensor,
    config: Mapping[str, Any],
    campaign_science: ScientificIdentity,
    device: torch.device,
    dtype: torch.dtype,
    verbose_callback: Callable[[Any], None] | None = None,
    accounting_callback: Callable[[Mapping[str, Any]], None] | None = None,
    supplied_target_response: torch.Tensor | None = None,
    supplied_target_seed: int | None = None,
    supplied_target_identity: str | None = None,
    supplied_initial_points: torch.Tensor | None = None,
    perform_held_out_assessment: bool = True,
    diagnostic_result_callback: Callable[[SequentialRunResult], None] | None = None,
    supplied_run_seed: int | None = None,
    supplied_terminal_seed_deriver: SeedDeriver | None = None,
) -> dict[str, Any]:
    """Run one hidden-target MUTACO inversion and one held-out assessment."""

    if supplied_run_seed is not None and (
        isinstance(supplied_run_seed, bool)
        or not isinstance(supplied_run_seed, int)
        or supplied_run_seed < 0
    ):
        raise ValueError("supplied MUTACO run seed must be a nonnegative integer")

    started = perf_counter()
    if config.get("identity_policy") == "target_method_execution_v1":
        method_configuration = {
            "runner": config["runner"],
            "proposal_effort": config.get("proposal_effort"),
            "terminal_recommendation": config.get("terminal_recommendation"),
            "simulation_budget_per_target": config["simulation_budget_per_target"],
            "dtype": config["dtype"],
            "method_seed_namespace": config["randomness"].get(
                "method_seed_namespace"
            ),
        }
        if "proposal_controls" in config:
            method_configuration["proposal_controls"] = dict(
                config["proposal_controls"]
            )
        method_science = ScientificIdentity.create(
            {
                "schema": "benchmark-method-run-v1",
                "target_family_id": campaign_science.value,
                "target_record_id": supplied_target_identity,
                "method": "mutaco",
                "method_configuration": method_configuration,
            }
        )
    else:
        method_science = campaign_science
    deriver = SeedDeriver(method_science, int(config["campaign"]["campaign_seed"]))
    simulator = adapter.build_simulator(config, device=device, dtype=dtype)
    target_seed = (
        int(supplied_target_seed)
        if supplied_target_seed is not None
        else deriver.seed_for(SeedCoordinate("target_response_noise", (index,)))
    )
    constructs_target = supplied_target_response is None
    accounting = {
        "phase": "target_construction" if constructs_target else "target_reuse",
        "target_construction_calls_started": int(constructs_target),
        "target_construction_calls_completed": 0,
        "inference_calls_started": 0,
        "inference_calls_completed": 0,
        "assessment_calls_started": 0,
        "assessment_calls_completed": 0,
        "terminal_recommendations_started": 0,
        "terminal_recommendations_completed": 0,
    }

    def notify() -> None:
        if accounting_callback is not None:
            try:
                accounting_callback(dict(accounting))
            except Exception:
                pass

    notify()
    target = (
        checked_evaluate(
            simulator,
            theta_star,
            device=device,
            dtype=dtype,
            context=_context(campaign_science.value, "target", index, target_seed),
        ).values
        if constructs_target
        else supplied_target_response.to(device=device, dtype=dtype).contiguous()
    )
    accounting["target_construction_calls_completed"] = int(constructs_target)
    notify()
    if _graph_paper_enabled(config):
        from experiments.graph_inversion.paper_policies import (
            graph_paper_randomness_family,
        )

        graph_seed_family = graph_paper_randomness_family(
            target_record_id=supplied_target_identity,
            simulator_identity=simulator.capabilities.simulator_identity,
            policy_identity=config.get("graph_paper_policy"),
        )
        run_seed = (
            int(supplied_run_seed)
            if supplied_run_seed is not None
            else SeedDeriver(
                graph_seed_family, int(config["campaign"]["campaign_seed"])
            ).seed_for(SeedCoordinate("mutaco_run", (index,)))
        )
    else:
        graph_seed_family = None
        run_seed = (
            int(supplied_run_seed)
            if supplied_run_seed is not None
            else deriver.seed_for(SeedCoordinate("mutaco_run", (index,)))
        )
    budget = MutacoBudget.resolve(config)
    profile = {
        "initial_points": budget.initial_points,
        "sequential_rounds": budget.sequential_rounds,
        "root_seed": run_seed,
    }
    runner_settings = {
        key: config["runner"][key]
        for key in (
            "transform", "gp", "acquisition", "recommendation_policy",
            "duplicate_policy",
        )
        if key in config["runner"]
    }
    terminal_context = None
    if _graph_paper_enabled(config):
        from experiments.graph_inversion.paper_policies import (
            resolved_proposal_settings,
        )

        resolved = resolved_proposal_settings(config)
        runner_settings["acquisition"] = {
            key: value
            for key, value in resolved.items()
            if key not in {"profile_id", "classification"}
        }
        # Preserve the pre-19C sequential intent identity, which addresses the
        # initial-design, GP, and proposal streams. The explicit terminal
        # context below selects the one authoritative physical recommendation.
        runner_settings["recommendation_policy"] = (
            "final_posterior_standardized_l1"
        )
        assert graph_seed_family is not None
        terminal_context = TerminalRecommendationContext(
            policy_id=str(config["terminal_recommendation"]),
            target_id=str(supplied_target_identity or campaign_science.value),
            target_index=index,
            seed_deriver=(
                supplied_terminal_seed_deriver
                if supplied_terminal_seed_deriver is not None
                else SeedDeriver(
                    graph_seed_family, int(config["campaign"]["campaign_seed"])
                )
            ),
        )
    intent = _intent(
        simulator,
        target=target,
        profile=profile,
        runner=runner_settings,
        dtype=dtype,
        evaluation_seed_family_id=(
            graph_seed_family.value
            if _graph_paper_enabled(config)
            else None
        ),
        caller_initial_points=supplied_initial_points,
    )
    def progress(event: Any) -> None:
        kind = event.kind.value
        if kind == "simulator_attempt_started":
            accounting["inference_calls_started"] += 1
        elif kind == "simulator_attempt_finished":
            accounting["inference_calls_completed"] += 1
        elif kind == "recommendation_started":
            accounting["terminal_recommendations_started"] = 1
        accounting["phase"] = event.phase
        accounting["runner_event"] = event.provenance()
        notify()
        if verbose_callback is not None:
            verbose_callback(event)

    result = run_sequential(
        simulator,
        config=SequentialRunnerConfig(
            intent,
            device=device,
            terminal_recommendation_context=terminal_context,
        ),
        progress_callback=progress,
    )
    if diagnostic_result_callback is not None:
        diagnostic_result_callback(result)
    if result.status is RunnerStatus.COMPLETED:
        accounting["terminal_recommendations_completed"] = 1
        notify()
    row: dict[str, Any] = {
        "schema_version": "noisy-cohort-target-v1",
        "campaign_id": campaign_science.value,
        "target_family_id": campaign_science.value,
        "target_record_id": supplied_target_identity,
        "method_identity": method_science.value,
        "method": "mutaco",
        "target_index": index,
        "theta_star": theta_star.detach().cpu().tolist(),
        "target_response": target.detach().cpu().tolist(),
        "target_seed": target_seed,
        "mutaco_root_seed": run_seed,
        "status": result.status.value,
        "failure_explanation": result.failure_explanation,
        "target_construction_calls": int(constructs_target),
        "inference_calls": result.physical_accounting["simulator_successes"],
        "assessment_calls": 0,
        "gp_fits": result.physical_accounting["gp_fit_successes"],
        "runtime_seconds": perf_counter() - started,
        "accounting_completeness": "exact",
        "runner_provenance": result.provenance(),
        "resolved_mutaco_budget": budget.document(),
        "scientific_projection": (
            scientific_projection(result)
            if result.status is RunnerStatus.COMPLETED
            else None
        ),
    }
    if result.status is not RunnerStatus.COMPLETED:
        return row
    assert result.recommendation is not None
    if _graph_paper_enabled(config):
        theta_rec = result.recommendation.selected_theta
        row["graph_terminal_recommendation"] = result.recommendation.document()
        row["recommendation_seconds"] = result.timings["recommendation_seconds"]
        row["proposal_effort"] = str(
            config.get("proposal_effort", "explicit_controls_v1")
        )
        row["resolved_proposal_controls"] = resolved_proposal_settings(config)
    else:
        theta_rec = result.recommendation.selected_theta
    if not perform_held_out_assessment:
        row.update(
            {
                "recommended_theta": theta_rec.detach().cpu().tolist(),
                "assessment_calls": 0,
                "runtime_seconds": perf_counter() - started,
            }
        )
        return row
    assessment_deriver = (
        SeedDeriver(graph_seed_family, int(config["campaign"]["campaign_seed"]))
        if graph_seed_family is not None
        else deriver
    )
    assessment_seed = assessment_deriver.seed_for(
        SeedCoordinate("held_out_assessment", (index,))
    )
    accounting["phase"] = "held_out_assessment"
    accounting["assessment_calls_started"] = 1
    notify()
    assessment_started = perf_counter()
    try:
        assessment = checked_evaluate(
            simulator,
            theta_rec,
            device=device,
            dtype=dtype,
            context=_context(method_science.value, "assessment", index, assessment_seed),
        ).values
    except Exception as exc:
        row.update(
            {
                "status": "assessment_failed",
                "failure_explanation": f"{type(exc).__name__}: {exc}",
                "recommended_theta": theta_rec.detach().cpu().tolist(),
                "assessment_seed": assessment_seed,
                "assessment_calls": 0,
                "assessment_seconds": perf_counter() - assessment_started,
                "runtime_seconds": perf_counter() - started,
            }
        )
        return row
    accounting["assessment_calls_completed"] = 1
    notify()
    target_errors = vector_errors(assessment, target)
    design_errors = vector_errors(theta_rec, theta_star)
    latent_target = adapter.latent_response(theta_star)
    latent_recommendation = adapter.latent_response(theta_rec)
    latent_errors = (
        vector_errors(latent_recommendation, latent_target)
        if latent_target is not None and latent_recommendation is not None
        else None
    )
    row.update(
        {
            "recommended_theta": theta_rec.detach().cpu().tolist(),
            "assessment_response": assessment.detach().cpu().tolist(),
            "assessment_seed": assessment_seed,
            "target_error_l1": target_errors["l1"],
            "target_error_l2": target_errors["l2"],
            "design_error_l2": design_errors["l2"],
            "design_error_l1": design_errors["l1"],
            "latent_error_l1": latent_errors["l1"] if latent_errors else None,
            "latent_error_l2": latent_errors["l2"] if latent_errors else None,
            "assessment_calls": 1,
            "assessment_seconds": perf_counter() - assessment_started,
            "runtime_seconds": perf_counter() - started,
        }
    )
    return row
