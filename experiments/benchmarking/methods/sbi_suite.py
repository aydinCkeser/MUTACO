"""Lazy-loaded typed SBI method suite."""

from __future__ import annotations

from importlib.metadata import version
from time import perf_counter, process_time
import traceback
import warnings
from typing import Any, Callable, Mapping
from pathlib import Path

import torch

from mutaco import (
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    SimulationContext,
    checked_evaluate,
    checked_evaluate_batch,
)

from ..contracts import REGISTERED_SBI_METHODS, SimulatorAdapter
from ..statistics import vector_errors
from ..timing import TimingLedger, TimingValidationError, summarize_sbi_timing
from .direct_posterior_diagnostics import (
    posterior_diagnostics,
    safe_training_diagnostics,
)


class BenchmarkDeadlineExceeded(RuntimeError):
    pass


def sequential_allocation(total: int, rounds: int) -> tuple[int, ...]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("simulation budget must be a positive integer")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("round count must be a positive integer")
    quotient, remainder = divmod(total, rounds)
    allocation = tuple(
        quotient + (1 if index < remainder else 0) for index in range(rounds)
    )
    if any(value < 1 for value in allocation):
        raise ValueError("simulation budget must allocate at least one call per round")
    return allocation


def _lazy_sbi() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from sbi.inference import NLE, NPE, NRE
        from sbi.inference.posteriors.posterior_parameters import (
            MCMCPosteriorParameters,
        )
        from sbi.utils import BoxUniform
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "SBI methods require the separate pinned mutaco-sbi environment"
        ) from exc
    return NPE, NLE, NRE, BoxUniform, MCMCPosteriorParameters, SummaryWriter


def _uniform(
    count: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=lower.device)
    generator.manual_seed(seed)
    return (
        lower
        + torch.rand(
            (count, lower.numel()),
            dtype=lower.dtype,
            device=lower.device,
            generator=generator,
        )
        * (upper - lower)
    ).contiguous()


def _simulate_batch(
    simulator: Any,
    theta: torch.Tensor,
    *,
    run_id: str,
    target_index: int,
    round_index: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    context_policy: str = "single_batch_context_v1",
    logical_start_index: int = 0,
    context_ledger: list[dict[str, Any]] | None = None,
    fixed_evaluation_seed: int | None = None,
) -> torch.Tensor:
    if fixed_evaluation_seed is not None:
        if (
            isinstance(fixed_evaluation_seed, bool)
            or not isinstance(fixed_evaluation_seed, int)
            or fixed_evaluation_seed < 0
        ):
            raise ValueError("fixed diagnostic evaluation seed must be nonnegative")
        outputs = []
        for row_index in range(theta.shape[0]):
            logical_index = logical_start_index + row_index
            context = SimulationContext(
                run_id,
                "sbi_training",
                logical_index,
                0,
                fixed_evaluation_seed,
            )
            if context_ledger is not None:
                context_ledger.append(
                    {
                        "purpose": "sbi_simulation_row",
                        "round_index": round_index,
                        "logical_index": logical_index,
                        "seed": fixed_evaluation_seed,
                        "context_policy": "legacy_full_stream_crn_diagnostic",
                        "classification": "diagnostic-only",
                    }
                )
            outputs.append(
                checked_evaluate(
                    simulator,
                    theta[row_index].contiguous(),
                    device=device,
                    dtype=dtype,
                    context=context,
                ).values
            )
        return torch.stack(outputs).contiguous()
    if context_policy == "per_logical_simulation_index_v1":
        row_deriver = SeedDeriver(ScientificIdentity(run_id), seed)
        contexts = tuple(
            SimulationContext(
                run_id,
                "sbi_training",
                logical_start_index + row_index,
                0,
                row_deriver.seed_for(
                    SeedCoordinate(
                        "sbi_simulation_row",
                        (
                            target_index,
                            round_index,
                            logical_start_index + row_index,
                        ),
                    )
                ),
            )
            for row_index in range(theta.shape[0])
        )
        if context_ledger is not None:
            context_ledger.extend(
                {
                    "purpose": "sbi_simulation_row",
                    "round_index": round_index,
                    "logical_index": context.logical_index,
                    "seed": context.evaluation_seed,
                    "context_policy": context_policy,
                }
                for context in contexts
            )
        return checked_evaluate_batch(
            simulator,
            theta,
            contexts,
            device=device,
            dtype=dtype,
        ).values
    if context_policy != "single_batch_context_v1":
        raise ValueError(f"unknown SBI batch context policy: {context_policy!r}")
    context = SimulationContext(
        run_id,
        "sbi_training",
        target_index,
        round_index,
        seed,
    )
    return checked_evaluate(
        simulator, theta, device=device, dtype=dtype, context=context
    ).values


def run_sbi_target(
    *,
    method: str,
    adapter: SimulatorAdapter,
    index: int,
    theta_star: torch.Tensor,
    target_response: torch.Tensor,
    config: Mapping[str, Any],
    campaign_science: Any,
    device: torch.device,
    dtype: torch.dtype,
    deadline_seconds: float | None = None,
    artifact_output: Path | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    target_record_id: str | None = None,
    fixed_inference_evaluation_seed: int | None = None,
) -> dict[str, Any]:
    """Train one declared SBI policy and return its posterior-mean recommendation."""

    started = perf_counter()
    cpu_started = process_time()
    if method not in REGISTERED_SBI_METHODS:
        raise ValueError(f"unknown SBI method: {method!r}")
    (
        NPE,
        NLE,
        NRE,
        BoxUniform,
        MCMCPosteriorParameters,
        SummaryWriter,
    ) = _lazy_sbi()
    settings = config["sbi"]
    budget = int(config["simulation_budget_per_target"])
    sequential = method in {"sbi_snpe", "sbi_snle", "sbi_snre"}
    allocations = sequential_allocation(budget, 3) if sequential else (budget,)
    lower = torch.tensor(
        adapter.build_simulator(config, device=device, dtype=dtype)
        .capabilities.parameter_spec.lower,
        device=device,
        dtype=dtype,
    )
    upper = torch.tensor(
        adapter.build_simulator(config, device=device, dtype=dtype)
        .capabilities.parameter_spec.upper,
        device=device,
        dtype=dtype,
    )
    prior = BoxUniform(lower, upper, device=str(device))
    inference_class = {
        "sbi_npe": NPE,
        "sbi_snpe": NPE,
        "sbi_nle": NLE,
        "sbi_snle": NLE,
        "sbi_nre": NRE,
        "sbi_snre": NRE,
    }[method]
    estimator_key = "classifier" if inference_class is NRE else "density_estimator"
    estimator_value = (
        settings["classifier"] if inference_class is NRE else settings["density_estimator"]
    )
    log_directory = artifact_output or Path("runs") / "sbi_method_logs" / method
    writer = SummaryWriter(log_dir=str(log_directory))
    inference = inference_class(
        prior=prior,
        device=str(device),
        show_progress_bars=bool(settings["show_progress_bars"]),
        summary_writer=writer,
        **{estimator_key: estimator_value},
    )
    randomness = config.get("randomness", {})
    method_configuration = {
        "sbi": config["sbi"],
        "simulation_budget_per_target": config["simulation_budget_per_target"],
        "dtype": config["dtype"],
        "method_seed_namespace": randomness.get(
            "method_seed_namespace"
        ),
    }
    if "sbi_batch_context_policy" in randomness:
        method_configuration["sbi_batch_context_policy"] = randomness[
            "sbi_batch_context_policy"
        ]
    if fixed_inference_evaluation_seed is not None:
        method_configuration["diagnostic_inference_seed_control"] = {
            "identifier": "legacy_full_stream_crn_diagnostic",
            "classification": "diagnostic-only",
            "fixed_evaluation_seed": fixed_inference_evaluation_seed,
        }
    method_science = (
        ScientificIdentity.create(
            {
                "schema": "benchmark-method-run-v1",
                "target_family_id": campaign_science.value,
                "target_record_id": target_record_id,
                "method": method,
                "method_configuration": method_configuration,
            }
        )
        if config.get("identity_policy") == "target_method_execution_v1"
        else campaign_science
    )
    deriver = SeedDeriver(method_science, int(config["campaign"]["campaign_seed"]))
    ledger = TimingLedger()
    simulator = adapter.build_simulator(config, device=device, dtype=dtype)
    posterior = None
    proposal = None
    failure_stage = "initialization"
    seed_ledger: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    accounting = {
        "phase": "initialization",
        "inference_calls_started": 0,
        "inference_calls_completed": 0,
        "assessment_calls_started": 0,
        "assessment_calls_completed": 0,
    }

    def notify() -> None:
        if progress_callback is not None:
            try:
                progress_callback(dict(accounting))
            except Exception:
                pass

    def finish_timing(status: str) -> tuple[float, float, dict[str, Any]]:
        runtime_seconds = perf_counter() - started
        process_cpu_seconds = process_time() - cpu_started
        try:
            summary = summarize_sbi_timing(
                ledger.events,
                method_total_wall_seconds=runtime_seconds,
                method_total_process_cpu_seconds=process_cpu_seconds,
                status=status,
                failure_phase=None if status == "completed" else failure_stage,
            )
        except TimingValidationError as timing_error:
            summary = {
                "schema_version": "mutaco-paper-sbi-timing-v1",
                "scope": "in_process_target_method",
                "completeness": "partial_unknown",
                "termination_status": status,
                "failure_phase": None if status == "completed" else failure_stage,
                "timing_validation_error": (
                    f"{type(timing_error).__name__}: {timing_error}"
                ),
                "raw_events": [dict(event) for event in ledger.events],
                "method_total": {
                    "wall_seconds": runtime_seconds,
                    "process_cpu_seconds": process_cpu_seconds,
                },
            }
        return runtime_seconds, process_cpu_seconds, summary

    notify()
    try:
        for round_index, count in enumerate(allocations):
            failure_stage = f"round_{round_index}_deadline_check"
            if (
                deadline_seconds is not None
                and perf_counter() - started >= deadline_seconds
            ):
                raise BenchmarkDeadlineExceeded(
                    f"equal-wall-clock deadline exhausted before round {round_index}"
                )
            simulation_seed = deriver.seed_for(
                SeedCoordinate("sbi_simulation", (method, index, round_index))
            )
            seed_ledger.append(
                {
                    "purpose": "sbi_simulation",
                    "round_index": round_index,
                    "seed": simulation_seed,
                }
            )
            failure_stage = f"round_{round_index}_simulation"
            accounting["phase"] = failure_stage
            accounting["inference_calls_started"] += count
            notify()
            with ledger.event(
                "simulation",
                method=method,
                target_index=index,
                round_index=round_index,
                work_count=count,
                simulator_calls=count,
                device=str(device),
            ):
                if proposal is None:
                    theta = _uniform(
                        count, lower, upper, seed=simulation_seed
                    )
                else:
                    with torch.random.fork_rng(
                        devices=[device] if device.type == "cuda" else []
                    ):
                        torch.manual_seed(simulation_seed)
                        theta = proposal.sample((count,)).to(
                            device=device, dtype=dtype
                        ).contiguous()
                x = _simulate_batch(
                    simulator,
                    theta,
                    run_id=method_science.value,
                    target_index=index,
                    round_index=round_index,
                    seed=simulation_seed,
                    device=device,
                    dtype=dtype,
                    context_policy=str(
                        randomness.get(
                            "sbi_batch_context_policy", "single_batch_context_v1"
                        )
                    ),
                    logical_start_index=sum(allocations[:round_index]),
                    context_ledger=seed_ledger,
                    fixed_evaluation_seed=fixed_inference_evaluation_seed,
                )
            accounting["inference_calls_completed"] += count
            notify()
            training_seed = deriver.seed_for(
                SeedCoordinate("sbi_training", (method, index, round_index))
            )
            seed_ledger.append(
                {
                    "purpose": "sbi_training",
                    "round_index": round_index,
                    "seed": training_seed,
                }
            )
            append_kwargs: dict[str, Any] = {}
            if inference_class is NPE and proposal is not None:
                append_kwargs["proposal"] = proposal
            if inference_class in {NLE, NRE}:
                append_kwargs["from_round"] = round_index
            with ledger.event(
                "neural_training",
                method=method,
                target_index=index,
                round_index=round_index,
                work_count=count,
                device=str(device),
            ):
                failure_stage = f"round_{round_index}_training"
                accounting["phase"] = failure_stage
                notify()
                with torch.random.fork_rng(
                    devices=[device] if device.type == "cuda" else []
                ):
                    torch.manual_seed(training_seed)
                    validation_fraction = float(settings["validation_fraction"])
                    if method == "sbi_snre":
                        validation_fraction = float(
                            settings["snre_validation_fraction"]
                        )
                    training_kwargs = {
                        "training_batch_size": min(
                            int(settings["training_batch_size"]), count
                        ),
                        "validation_fraction": validation_fraction,
                        "stop_after_epochs": int(settings["stop_after_epochs"]),
                        "max_num_epochs": int(settings["max_num_epochs"]),
                        "show_train_summary": False,
                        "learning_rate": float(settings.get("learning_rate", 5e-4)),
                    }
                    if inference_class is NRE:
                        training_kwargs["num_atoms"] = 2
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator = inference.append_simulations(
                            theta, x, **append_kwargs
                        ).train(**training_kwargs)
            diagnostic_context = {
                "round_index": round_index,
                "simulation_count": count,
                "proposal_class": (
                    None if proposal is None else
                    f"{type(proposal).__module__}.{type(proposal).__qualname__}"
                ),
            }
            diagnostic_arguments = {
                "max_num_epochs": int(settings["max_num_epochs"]),
                "stop_after_epochs": int(settings["stop_after_epochs"]),
                "round_index": round_index,
                "sbi_version": version("sbi"),
                "captured_warnings": tuple(
                    f"{item.category.__name__}: {item.message}" for item in caught
                ),
            }
            diagnostic_context.update(
                safe_training_diagnostics(
                    inference, estimator, **diagnostic_arguments
                )
            )
            training_history.append(diagnostic_context)
            with ledger.event(
                "posterior_construction",
                method=method,
                target_index=index,
                round_index=round_index,
                device=str(device),
            ):
                if inference_class is NPE:
                    failure_stage = f"round_{round_index}_posterior_build"
                    posterior = inference.build_posterior(
                        estimator, sample_with="direct"
                    )
                else:
                    failure_stage = f"round_{round_index}_mcmc_posterior_build"
                    posterior = inference.build_posterior(
                        estimator,
                        posterior_parameters=MCMCPosteriorParameters(
                            method=settings["mcmc"]["method"],
                            num_chains=int(settings["mcmc"]["num_chains"]),
                            warmup_steps=int(settings["mcmc"]["warmup_steps"]),
                            thin=int(settings["mcmc"]["thin"]),
                            init_strategy=settings["mcmc"]["init_strategy"],
                        ),
                    )
                proposal = posterior.set_default_x(target_response)
        assert posterior is not None
        if deadline_seconds is not None and perf_counter() - started >= deadline_seconds:
            raise BenchmarkDeadlineExceeded(
                "equal-wall-clock deadline exhausted before posterior sampling"
            )
        recommendation_seed = deriver.seed_for(
            SeedCoordinate("sbi_recommendation", (method, index))
        )
        seed_ledger.append(
            {
                "purpose": "sbi_recommendation",
                "round_index": None,
                "seed": recommendation_seed,
            }
        )
        posterior_samples = int(settings["posterior_samples"])
        failure_stage = "posterior_sampling"
        accounting["phase"] = failure_stage
        notify()
        with ledger.event(
            "posterior_sampling",
            method=method,
            target_index=index,
            work_count=posterior_samples,
            device=str(device),
        ):
            with torch.random.fork_rng(
                devices=[device] if device.type == "cuda" else []
            ):
                torch.manual_seed(recommendation_seed)
                samples = proposal.sample(
                    (posterior_samples,),
                    show_progress_bars=bool(settings["show_progress_bars"]),
                ).to(device=device, dtype=dtype)
        with ledger.event(
            "posterior_diagnostics_recommendation",
            method=method,
            target_index=index,
            work_count=posterior_samples,
            device=str(device),
        ):
            direct_diagnostics = (
                posterior_diagnostics(
                    samples, lower=lower, upper=upper, posterior=proposal,
                    target_response=target_response,
                )
                if inference_class is NPE else None
            )
            theta_rec = samples.mean(dim=0).contiguous()
        assessment_seed = deriver.seed_for(
            SeedCoordinate("held_out_assessment", (method, index))
        )
        seed_ledger.append(
            {
                "purpose": "held_out_assessment",
                "round_index": None,
                "seed": assessment_seed,
            }
        )
        failure_stage = "held_out_assessment"
        accounting["phase"] = failure_stage
        accounting["assessment_calls_started"] = 1
        notify()
        with ledger.event(
            "held_out_assessment",
            method=method,
            target_index=index,
            simulator_calls=1,
            budget_classification="assessment_outside_inference",
            device=str(device),
        ):
            assessment = checked_evaluate(
                simulator,
                theta_rec,
                device=device,
                dtype=dtype,
                context=SimulationContext(
                    method_science.value,
                    "assessment",
                    index,
                    0,
                    assessment_seed,
                ),
            ).values
        accounting["assessment_calls_completed"] = 1
        notify()
        target_errors = vector_errors(assessment, target_response)
        design_errors = vector_errors(theta_rec, theta_star)
        latent_target = adapter.latent_response(theta_star)
        latent_rec = adapter.latent_response(theta_rec)
        latent_errors = (
            vector_errors(latent_rec, latent_target)
            if latent_target is not None and latent_rec is not None
            else None
        )
        runtime_seconds, process_cpu_seconds, timing_summary = finish_timing(
            "completed"
        )
        return {
            "schema_version": "mutaco-sbi-target-v1",
            "campaign_id": campaign_science.value,
            "target_family_id": campaign_science.value,
            "target_record_id": target_record_id,
            "method_identity": method_science.value,
            "target_index": index,
            "method": method,
            "status": "completed",
            "failure_explanation": None,
            "theta_star": theta_star.detach().cpu().tolist(),
            "target_response": target_response.detach().cpu().tolist(),
            "recommended_theta": theta_rec.detach().cpu().tolist(),
            "assessment_response": assessment.detach().cpu().tolist(),
            "assessment_seed": assessment_seed,
            "recommendation_seed": recommendation_seed,
            "target_error_l1": target_errors["l1"],
            "target_error_l2": target_errors["l2"],
            "design_error_l2": design_errors["l2"],
            "latent_error_l1": latent_errors["l1"] if latent_errors else None,
            "latent_error_l2": latent_errors["l2"] if latent_errors else None,
            "target_construction_calls": 0,
            "inference_calls": sum(allocations),
            "assessment_calls": 1,
            "gp_fits": 0,
            "sequential_round_allocation": allocations,
            "posterior_samples": posterior_samples,
            "training_diagnostics": training_history,
            "posterior_diagnostics": direct_diagnostics,
            "diagnostic_inference_seed_policy": (
                {
                    "identifier": "legacy_full_stream_crn_diagnostic",
                    "classification": "diagnostic-only",
                    "fixed_evaluation_seed": fixed_inference_evaluation_seed,
                }
                if fixed_inference_evaluation_seed is not None
                else None
            ),
            "seed_ledger": seed_ledger,
            "effective_sbi_settings": dict(settings),
            "mcmc_fallback_used": False,
            "timing_events": ledger.events,
            "timing_summary": timing_summary,
            "runtime_seconds": runtime_seconds,
            "runtime_process_cpu_seconds": process_cpu_seconds,
            "deadline_seconds": deadline_seconds,
            "accounting_completeness": "exact",
        }
    except Exception as exc:
        runtime_seconds, process_cpu_seconds, timing_summary = finish_timing("failed")
        return {
            "schema_version": "mutaco-sbi-target-v1",
            "campaign_id": campaign_science.value,
            "target_family_id": campaign_science.value,
            "target_record_id": target_record_id,
            "method_identity": method_science.value,
            "target_index": index,
            "method": method,
            "status": "failed",
            "failure_explanation": f"{type(exc).__name__}: {exc}",
            "failure_type": (
                "mcmc_failure"
                if "mcmc" in failure_stage or failure_stage == "posterior_sampling"
                else "training_failure"
                if "training" in failure_stage
                else "method_failure"
            ),
            "failure_stage": failure_stage,
            "failure_traceback": traceback.format_exc(),
            "theta_star": theta_star.detach().cpu().tolist(),
            "target_response": target_response.detach().cpu().tolist(),
            "inference_calls": sum(
                event["simulator_calls"] for event in ledger.events
            ),
            "assessment_calls": 0,
            "gp_fits": 0,
            "sequential_round_allocation": allocations,
            "seed_ledger": seed_ledger,
            "effective_sbi_settings": dict(settings),
            "mcmc_fallback_used": False,
            "timing_events": ledger.events,
            "timing_summary": timing_summary,
            "runtime_seconds": runtime_seconds,
            "runtime_process_cpu_seconds": process_cpu_seconds,
            "deadline_seconds": deadline_seconds,
            "accounting_completeness": (
                "exact"
                if accounting["inference_calls_started"]
                == accounting["inference_calls_completed"]
                and accounting["assessment_calls_started"]
                == accounting["assessment_calls_completed"]
                else "partial_known_lower_bound"
            ),
        }
    finally:
        writer.close()
