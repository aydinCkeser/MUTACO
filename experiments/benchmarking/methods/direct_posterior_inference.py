"""Prospective decision-only NPE/SNPE execution for v3 paper protocols."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Mapping, Sequence
import warnings

import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver
from experiments.contracts import PhysicalCallCounts
from ..contracts import SimulatorAdapter
from ..timing import TimingLedger, summarize_sbi_timing
from .direct_posterior_decision import posterior_sample_map_decision
from .direct_posterior_diagnostics import posterior_diagnostics, safe_training_diagnostics
from .sbi_suite import _lazy_sbi, _simulate_batch, _uniform


def validate_sequential_round_allocation(
    allocation: Sequence[int], *, budget: int
) -> tuple[int, ...]:
    """Validate an explicit positive allocation against its scientific budget."""

    values = tuple(allocation)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("sequential round allocation must contain positive integers")
    if sum(values) != budget:
        raise ValueError("sequential round allocation must sum to the inference budget")
    return values


def _method_configuration_for_identity(
    *,
    method: str,
    config: Mapping[str, Any],
    budget: int,
    allocations: Sequence[int],
) -> dict[str, Any]:
    """Return the compatibility-preserving direct-posterior identity document."""

    method_configuration = {
        "sbi": config["sbi"],
        "simulation_budget_per_target": budget,
        "dtype": config["dtype"],
        "method_seed_namespace": config["randomness"].get("method_seed_namespace"),
        "sbi_batch_context_policy": config["randomness"].get(
            "sbi_batch_context_policy"
        ),
        "final_decision_policy": "posterior_sample_map_physical_v1",
    }
    # A single full-budget round is intrinsic to cold-start NPE and was not a
    # field in the accepted Paper-120 identity.  SNPE's multi-round allocation
    # is an explicit scientific choice and remains identity-bearing.
    if method == "sbi_snpe":
        method_configuration["sequential_round_allocation"] = list(allocations)
    return method_configuration


def run_direct_posterior_sample_map_target(
    *,
    method: str,
    adapter: SimulatorAdapter,
    index: int,
    theta_star: torch.Tensor,
    target_response: torch.Tensor,
    config: Mapping[str, Any],
    campaign_science: ScientificIdentity,
    device: torch.device,
    dtype: torch.dtype,
    artifact_output: Path,
    target_record_id: str,
    sequential_round_allocation: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Train target-specific NPE/SNPE and return a decision-only sample-MAP.

    This prospective path deliberately stops before held-out assessment. The
    caller must persist the all-method decision freeze and then invoke the v3
    paired-assessment primitive.
    """

    if method not in {"sbi_npe", "sbi_snpe"}:
        raise ValueError("direct posterior inference supports only NPE and SNPE")
    started, cpu_started = perf_counter(), process_time()
    NPE, _, _, BoxUniform, _, SummaryWriter = _lazy_sbi()
    settings = config["sbi"]
    budget = int(config["simulation_budget_per_target"])
    if budget <= 0:
        raise ValueError("Paper-v3 direct posterior inference requires a positive budget")
    if method == "sbi_npe":
        allocations = validate_sequential_round_allocation(
            (budget,) if sequential_round_allocation is None else sequential_round_allocation,
            budget=budget,
        )
        if allocations != (budget,):
            raise ValueError("cold-start NPE requires one round containing the full budget")
    else:
        raw_allocation = (
            settings.get("sequential_round_allocation")
            if sequential_round_allocation is None
            else sequential_round_allocation
        )
        if raw_allocation is None:
            raise ValueError("SNPE requires an explicit sequential round allocation")
        allocations = validate_sequential_round_allocation(raw_allocation, budget=budget)
        if len(allocations) < 2:
            raise ValueError("SNPE requires at least two sequential rounds")
    simulator = adapter.build_simulator(config, device=device, dtype=dtype)
    lower = torch.tensor(
        simulator.capabilities.parameter_spec.lower, device=device, dtype=dtype
    )
    upper = torch.tensor(
        simulator.capabilities.parameter_spec.upper, device=device, dtype=dtype
    )
    prior = BoxUniform(lower, upper, device=str(device))
    method_configuration = _method_configuration_for_identity(
        method=method,
        config=config,
        budget=budget,
        allocations=allocations,
    )
    method_science = ScientificIdentity.create(
        {
            "schema": "benchmark-method-run-v1",
            "target_family_id": campaign_science.value,
            "target_record_id": target_record_id,
            "method": method,
            "method_configuration": method_configuration,
        }
    )
    deriver = SeedDeriver(method_science, int(config["campaign"]["campaign_seed"]))
    ledger = TimingLedger()
    writer = SummaryWriter(log_dir=str(artifact_output))
    inference = NPE(
        prior=prior,
        device=str(device),
        show_progress_bars=bool(settings["show_progress_bars"]),
        summary_writer=writer,
        density_estimator=settings["density_estimator"],
    )
    seed_ledger: list[dict[str, Any]] = []
    try:
        conditioned = None
        training_diagnostics = []
        logical_start_index = 0
        for round_index, round_count in enumerate(allocations):
            simulation_seed = deriver.seed_for(
                SeedCoordinate("sbi_simulation", (method, index, round_index))
            )
            seed_ledger.append({
                "purpose": "sbi_simulation",
                "round_index": round_index,
                "seed": simulation_seed,
            })
            with ledger.event(
                "simulation",
                method=method,
                target_index=index,
                round_index=round_index,
                work_count=round_count,
                simulator_calls=round_count,
                device=str(device),
            ):
                if conditioned is None:
                    theta = _uniform(round_count, lower, upper, seed=simulation_seed)
                else:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(simulation_seed)
                        theta = conditioned.sample(
                            (round_count,),
                            show_progress_bars=bool(settings["show_progress_bars"]),
                        ).to(device=device, dtype=dtype).contiguous()
                x = _simulate_batch(
                    simulator,
                    theta,
                    run_id=method_science.value,
                    target_index=index,
                    round_index=round_index,
                    seed=simulation_seed,
                    device=device,
                    dtype=dtype,
                    context_policy=str(config["randomness"]["sbi_batch_context_policy"]),
                    logical_start_index=logical_start_index,
                    context_ledger=seed_ledger,
                )
            training_seed = deriver.seed_for(
                SeedCoordinate("sbi_training", (method, index, round_index))
            )
            seed_ledger.append({
                "purpose": "sbi_training",
                "round_index": round_index,
                "seed": training_seed,
            })
            append_kwargs = {} if conditioned is None else {"proposal": conditioned}
            with ledger.event(
                "neural_training",
                method=method,
                target_index=index,
                round_index=round_index,
                work_count=round_count,
                device=str(device),
            ):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(training_seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator = inference.append_simulations(
                            theta, x, **append_kwargs
                        ).train(
                            training_batch_size=min(
                                int(settings["training_batch_size"]), round_count
                            ),
                            validation_fraction=float(settings["validation_fraction"]),
                            stop_after_epochs=int(settings["stop_after_epochs"]),
                            max_num_epochs=int(settings["max_num_epochs"]),
                            show_train_summary=False,
                            learning_rate=float(settings.get("learning_rate", 5e-4)),
                        )
            diagnostic = safe_training_diagnostics(
                inference,
                estimator,
                max_num_epochs=int(settings["max_num_epochs"]),
                stop_after_epochs=int(settings["stop_after_epochs"]),
                round_index=round_index,
                sbi_version=version("sbi"),
                captured_warnings=tuple(
                    f"{item.category.__name__}: {item.message}" for item in caught
                ),
            )
            diagnostic.update({
                "simulation_count": round_count,
                "proposal_class": (
                    None
                    if conditioned is None
                    else f"{type(conditioned).__module__}.{type(conditioned).__qualname__}"
                ),
            })
            training_diagnostics.append(diagnostic)
            with ledger.event(
                "posterior_construction",
                method=method,
                target_index=index,
                round_index=round_index,
                device=str(device),
            ):
                posterior = inference.build_posterior(estimator, sample_with="direct")
                conditioned = posterior.set_default_x(target_response)
            logical_start_index += round_count
        assert conditioned is not None
        recommendation_seed = deriver.seed_for(
            SeedCoordinate("sbi_recommendation", (method, index))
        )
        seed_ledger.append(
            {"purpose": "sbi_recommendation", "round_index": None, "seed": recommendation_seed}
        )
        sample_count = int(settings["posterior_samples"])
        if sample_count != 10000:
            raise ValueError("graph Paper-v3 direct posterior inference requires 10,000 samples")
        with ledger.event(
            "posterior_sampling",
            method=method,
            target_index=index,
            work_count=sample_count,
            device=str(device),
        ):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(recommendation_seed)
                samples = conditioned.sample(
                    (sample_count,),
                    show_progress_bars=bool(settings["show_progress_bars"]),
                ).to(device=device, dtype=dtype)
        with ledger.event(
            "posterior_diagnostics_recommendation",
            method=method,
            target_index=index,
            work_count=sample_count,
            device=str(device),
        ):
            diagnostics = posterior_diagnostics(
                samples,
                lower=lower,
                upper=upper,
                posterior=conditioned,
                target_response=target_response,
            )
            posterior_mean = samples.mean(dim=0).contiguous()
        map_result = posterior_sample_map_decision(
            method=method,
            posterior=conditioned,
            samples=samples,
            lower=lower,
            upper=upper,
            method_configuration_identity=method_science.value,
            inference_budget=budget,
            inference_calls=PhysicalCallCounts(budget, budget),
            target_index=index,
        )
        map_event = dict(map_result.timing_event)
        map_event["event_id"] = f"timing-{len(ledger.events):06d}"
        ledger.events.append(map_event)
        wall_seconds = perf_counter() - started
        cpu_seconds = process_time() - cpu_started
        timing_summary = summarize_sbi_timing(
            ledger.events,
            method_total_wall_seconds=wall_seconds,
            method_total_process_cpu_seconds=cpu_seconds,
            status="completed" if map_result.decision.status == "recommended" else "failed",
            failure_phase=(
                None
                if map_result.decision.status == "recommended"
                else "posterior_sample_map_scoring_selection"
            ),
        )
        return {
            "schema_version": "mutaco-sbi-v3-decision-only-v1",
            "campaign_id": campaign_science.value,
            "target_record_id": target_record_id,
            "method_identity": method_science.value,
            "target_index": index,
            "method": method,
            "status": map_result.decision.status,
            "failure_type": map_result.decision.failure_type,
            "failure_explanation": map_result.decision.failure_explanation,
            "theta_star": theta_star.detach().cpu().tolist(),
            "target_response": target_response.detach().cpu().tolist(),
            "recommended_theta": (
                None
                if map_result.decision.recommended_theta is None
                else list(map_result.decision.recommended_theta)
            ),
            "posterior_mean_diagnostic": posterior_mean.detach().cpu().tolist(),
            "recommendation_seed": recommendation_seed,
            "target_construction_calls": 0,
            "inference_calls": budget,
            "assessment_calls": 0,
            "posterior_samples": sample_count,
            "sequential_round_allocation": list(allocations),
            "training_diagnostics": training_diagnostics,
            "posterior_diagnostics": diagnostics,
            "seed_ledger": seed_ledger,
            "effective_sbi_settings": dict(settings),
            "timing_events": ledger.events,
            "timing_summary": timing_summary,
            "runtime_seconds": wall_seconds,
            "runtime_process_cpu_seconds": cpu_seconds,
            "accounting_completeness": "exact",
            "decision_outcome": map_result.decision,
            "decision_timing_event": map_event,
        }
    finally:
        writer.close()


def run_npe_sample_map_target(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible one-round NPE entry point."""

    return run_direct_posterior_sample_map_target(method="sbi_npe", **kwargs)
