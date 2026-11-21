"""Shared-training amortized SBI bank execution."""

from __future__ import annotations

from importlib.metadata import version
from time import perf_counter
from typing import Any, Mapping
from pathlib import Path
import warnings

import torch

from mutaco import SeedCoordinate, SeedDeriver, SimulationContext, checked_evaluate, stable_hash

from ..contracts import SimulatorAdapter
from ..statistics import vector_errors
from ..timing import TimingLedger
from .sbi_suite import _lazy_sbi, _simulate_batch, _uniform
from .direct_posterior_diagnostics import (
    posterior_diagnostics,
    safe_training_diagnostics,
)


AMORTIZED_METHODS = ("sbi_npe", "sbi_nle", "sbi_nre")


def run_amortized_bank(
    *,
    method: str,
    adapter: SimulatorAdapter,
    target_indices: tuple[int, ...],
    theta_bank: torch.Tensor,
    target_responses: Mapping[int, torch.Tensor],
    config: Mapping[str, Any],
    campaign_science: Any,
    device: torch.device,
    dtype: torch.dtype,
    artifact_output: Path | None = None,
) -> list[dict[str, Any]]:
    """Train one estimator once, then infer every declared target."""

    if method not in AMORTIZED_METHODS:
        raise ValueError(f"method is not amortizable: {method!r}")
    (
        NPE,
        NLE,
        NRE,
        BoxUniform,
        MCMCPosteriorParameters,
        SummaryWriter,
    ) = _lazy_sbi()
    settings = config["sbi"]
    simulator = adapter.build_simulator(config, device=device, dtype=dtype)
    lower = torch.tensor(
        simulator.capabilities.parameter_spec.lower, device=device, dtype=dtype
    )
    upper = torch.tensor(
        simulator.capabilities.parameter_spec.upper, device=device, dtype=dtype
    )
    prior = BoxUniform(lower, upper, device=str(device))
    inference_class = {"sbi_npe": NPE, "sbi_nle": NLE, "sbi_nre": NRE}[method]
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
    deriver = SeedDeriver(campaign_science, int(config["campaign"]["campaign_seed"]))
    total_budget = int(config["simulation_budget_per_target"]) * len(target_indices)
    target_key = ",".join(str(index) for index in target_indices)
    simulation_seed = deriver.seed_for(
        SeedCoordinate("sbi_amortized_simulation", (method, target_key))
    )
    training_seed = deriver.seed_for(
        SeedCoordinate("sbi_amortized_training", (method, target_key))
    )
    ledger = TimingLedger()
    started = perf_counter()
    with ledger.event(
        "shared_simulation",
        method=method,
        work_count=total_budget,
        simulator_calls=total_budget,
        device=str(device),
    ):
        theta = _uniform(total_budget, lower, upper, seed=simulation_seed)
        x = _simulate_batch(
            simulator,
            theta,
            run_id=campaign_science.value,
            target_index=0,
            round_index=0,
            seed=simulation_seed,
            device=device,
            dtype=dtype,
            context_policy=str(
                config.get("randomness", {}).get(
                    "sbi_batch_context_policy", "single_batch_context_v1"
                )
            ),
        )
    with ledger.event(
        "shared_neural_training",
        method=method,
        work_count=total_budget,
        device=str(device),
    ):
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(training_seed)
            training_kwargs: dict[str, Any] = {
                "training_batch_size": min(
                    int(settings["training_batch_size"]), total_budget
                ),
                "learning_rate": float(settings.get("learning_rate", 5e-4)),
                "validation_fraction": float(settings["validation_fraction"]),
                "stop_after_epochs": int(settings["stop_after_epochs"]),
                "max_num_epochs": int(settings["max_num_epochs"]),
                "show_train_summary": False,
            }
            if inference_class is NRE:
                training_kwargs["num_atoms"] = 2
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator = inference.append_simulations(theta, x).train(**training_kwargs)
    shared_training_diagnostics = safe_training_diagnostics(
        inference, estimator, max_num_epochs=int(settings["max_num_epochs"]),
        stop_after_epochs=int(settings["stop_after_epochs"]), round_index=0,
        sbi_version=version("sbi"),
        captured_warnings=tuple(
            f"{item.category.__name__}: {item.message}" for item in caught
        ),
    )
    if inference_class is NPE:
        posterior = inference.build_posterior(estimator, sample_with="direct")
    else:
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
    shared_id = stable_hash(
        "sbi-amortized-training-v1",
        {
            "campaign": campaign_science.value,
            "method": method,
            "targets": target_indices,
            "budget": total_budget,
            "simulation_seed": simulation_seed,
            "training_seed": training_seed,
        },
    )
    shared_seconds = perf_counter() - started
    rows: list[dict[str, Any]] = []
    for position, index in enumerate(target_indices):
        target = target_responses[index]
        recommendation_seed = deriver.seed_for(
            SeedCoordinate("sbi_recommendation", (method, "amortized", index))
        )
        target_started = perf_counter()
        conditioned = posterior.set_default_x(target)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(recommendation_seed)
            samples = conditioned.sample(
                (int(settings["posterior_samples"]),),
                show_progress_bars=bool(settings["show_progress_bars"]),
            ).to(device=device, dtype=dtype)
        theta_rec = samples.mean(dim=0).contiguous()
        diagnostics = posterior_diagnostics(
            samples, lower=lower, upper=upper, posterior=conditioned,
            target_response=target,
        )
        assessment_seed = deriver.seed_for(
            SeedCoordinate("held_out_assessment", (method, "amortized", index))
        )
        assessment = checked_evaluate(
            simulator,
            theta_rec,
            device=device,
            dtype=dtype,
            context=SimulationContext(
                campaign_science.value,
                "assessment",
                index,
                0,
                assessment_seed,
            ),
        ).values
        target_errors = vector_errors(assessment, target)
        design_errors = vector_errors(theta_rec, theta_bank[index])
        latent_target = adapter.latent_response(theta_bank[index])
        latent_rec = adapter.latent_response(theta_rec)
        latent_errors = (
            vector_errors(latent_rec, latent_target)
            if latent_target is not None and latent_rec is not None
            else None
        )
        rows.append(
            {
                "schema_version": "mutaco-sbi-target-v1",
                "campaign_id": campaign_science.value,
                "target_index": index,
                "method": method,
                "status": "completed",
                "failure_explanation": None,
                "theta_star": theta_bank[index].detach().cpu().tolist(),
                "target_response": target.detach().cpu().tolist(),
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
                "inference_calls": total_budget if position == 0 else 0,
                "shared_inference_calls": total_budget,
                "assessment_calls": 1,
                "gp_fits": 0,
                "posterior_samples": int(settings["posterior_samples"]),
                "training_diagnostics": shared_training_diagnostics,
                "posterior_diagnostics": diagnostics,
                "shared_training_id": shared_id,
                "shared_training_seconds": shared_seconds,
                "shared_cost_charged_here": position == 0,
                "timing_events": ledger.events if position == 0 else [],
                "runtime_seconds": perf_counter() - target_started,
            }
        )
    writer.close()
    return rows
