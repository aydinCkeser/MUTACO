"""Target-bank creation and independent noisy MUTACO runs."""

from __future__ import annotations

from pathlib import Path
import json
from time import perf_counter
from typing import Any, Callable

import torch

from mutaco import (
    RunnerStatus,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    SequentialRunnerConfig,
    SimulationContext,
    canonical_json_bytes,
    checked_evaluate,
    run_sequential,
    stable_hash,
)
from mutaco_analytic import HandwrittenAnalyticSimulator, analytic_response

from experiments.analytic_target_matching._support import _intent, _scientific_projection

from ._artifacts import recover_target, write_target
from ._metrics import vector_errors


def campaign_identity(config: dict[str, Any]) -> ScientificIdentity:
    return ScientificIdentity.create(
        {
            "schema": config["schema_version"],
            "campaign": config["campaign"],
            "simulator": config["simulator"],
            "runner": config["runner"],
            "dtype": config["dtype"],
        }
    )


def target_bank(
    config: dict[str, Any], *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, int, ScientificIdentity]:
    count = config["campaign"]["target_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("target_count must be a positive integer")
    if config["campaign"]["target_bank_policy"] != "scrambled_sobol":
        raise ValueError("unsupported target-bank policy")
    if config["campaign"]["replay_policy"] != "first_middle_last":
        raise ValueError("unsupported replay policy")
    if config["simulator"]["identifier"] != "handwritten":
        raise ValueError("unsupported cohort simulator")
    identity = campaign_identity(config)
    bank_family = ScientificIdentity.create(
        {
            "schema": "noisy-cohort-target-bank-family-v1",
            "policy": config["campaign"]["target_bank_policy"],
            "campaign_seed": config["campaign"]["campaign_seed"],
            "bounds": [[-1.0, -1.0], [1.0, 1.0]],
            "dtype": config["dtype"],
        }
    )
    seed = SeedDeriver(bank_family, config["campaign"]["campaign_seed"]).seed_for(
        SeedCoordinate("target_bank", ("scrambled_sobol",))
    )
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    theta = (-1.0 + 2.0 * engine.draw(count, dtype=dtype)).to(device).contiguous()
    if torch.unique(theta, dim=0).shape[0] != count:
        raise RuntimeError("target bank contains duplicates")
    return theta, seed, identity


def replay_indices(count: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, (count - 1) // 2, count - 1)))


def _context(run_id: str, purpose: str, index: int, seed: int) -> SimulationContext:
    return SimulationContext(run_id, purpose, index, 0, seed)


def run_target(
    *,
    index: int,
    theta_star: torch.Tensor,
    config: dict[str, Any],
    campaign_science: ScientificIdentity,
    device: torch.device,
    dtype: torch.dtype,
    verbose_callback: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    deriver = SeedDeriver(campaign_science, config["campaign"]["campaign_seed"])
    simulator = HandwrittenAnalyticSimulator(
        device=device,
        dtype=dtype,
        noise_mode=config["simulator"]["noise_mode"],
        noise_std=tuple(config["simulator"]["noise_std"]),
    )
    target_seed = deriver.seed_for(SeedCoordinate("target_response_noise", (index,)))
    target = checked_evaluate(
        simulator,
        theta_star,
        device=device,
        dtype=dtype,
        context=_context(campaign_science.value, "target", index, target_seed),
    ).values
    run_seed = deriver.seed_for(SeedCoordinate("mutaco_run", (index,)))
    profile = {
        "initial_points": config["runner"]["initial_points"],
        "sequential_rounds": config["runner"]["sequential_rounds"],
        "root_seed": run_seed,
    }
    runner_settings = {
        key: config["runner"][key]
        for key in ("transform", "gp", "acquisition", "recommendation_policy")
    }
    intent = _intent(simulator, target=target, profile=profile, runner=runner_settings, dtype=dtype)
    result = run_sequential(
        simulator,
        config=SequentialRunnerConfig(intent, device=device),
        progress_callback=verbose_callback,
    )
    row: dict[str, Any] = {
        "schema_version": "noisy-cohort-target-v1",
        "campaign_id": campaign_science.value,
        "target_index": index,
        "theta_star": theta_star.detach().cpu().tolist(),
        "target_response": target.detach().cpu().tolist(),
        "target_seed": target_seed,
        "mutaco_root_seed": run_seed,
        "status": result.status.value,
        "failure_explanation": result.failure_explanation,
        "runtime_seconds": perf_counter() - started,
        "runner_provenance": result.provenance(),
        "scientific_projection": (
            _scientific_projection(result)
            if result.status is RunnerStatus.COMPLETED
            else None
        ),
    }
    if result.status is not RunnerStatus.COMPLETED:
        return row
    assert result.recommendation is not None
    theta_rec = result.recommendation.selected_theta
    assessment_seed = deriver.seed_for(
        SeedCoordinate("held_out_assessment", (index,))
    )
    assessment = checked_evaluate(
        simulator,
        theta_rec,
        device=device,
        dtype=dtype,
        context=_context(campaign_science.value, "assessment", index, assessment_seed),
    ).values
    target_errors = vector_errors(assessment, target)
    design_errors = vector_errors(theta_rec, theta_star)
    latent_errors = vector_errors(analytic_response(theta_rec), analytic_response(theta_star))
    row.update(
        {
            "recommended_theta": theta_rec.detach().cpu().tolist(),
            "assessment_response": assessment.detach().cpu().tolist(),
            "assessment_seed": assessment_seed,
            "target_error_l1": target_errors["l1"],
            "target_error_l2": target_errors["l2"],
            "design_error_l2": design_errors["l2"],
            "latent_error_l1": latent_errors["l1"],
            "latent_error_l2": latent_errors["l2"],
            "target_construction_calls": 1,
            "inference_calls": result.physical_accounting["simulator_successes"],
            "assessment_calls": 1,
            "gp_fits": result.physical_accounting["gp_fit_successes"],
            "runtime_seconds": perf_counter() - started,
        }
    )
    return row


def run_cohort(
    *,
    config: dict[str, Any],
    theta_bank: torch.Tensor,
    campaign_science: ScientificIdentity,
    output: Path,
    device: torch.device,
    dtype: torch.dtype,
    restart: bool,
    verbose_events: bool,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    rows = []
    start = perf_counter()
    count = theta_bank.shape[0]
    recovered_count = 0
    executed_count = 0
    for index in range(count):
        directory = output / "targets" / f"target_{index:03d}"
        recovered = recover_target(directory, campaign_science.value) if restart else None
        if recovered is not None:
            row = recovered
            recovered_count += 1
        else:
            executed_count += 1
            callback = (
                lambda event, i=index: print(f"[target {i:03d}] {event.message}")
                if verbose_events
                else None
            )
            row = run_target(
                index=index,
                theta_star=theta_bank[index],
                config=config,
                campaign_science=campaign_science,
                device=device,
                dtype=dtype,
                verbose_callback=callback,
            )
            write_target(directory, row)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "target_index": index,
                            "status": row["status"],
                            "elapsed_seconds": perf_counter() - start,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        rows.append(row)
        elapsed = perf_counter() - start
        eta = elapsed / (index + 1) * (count - index - 1)
        print(
            f"[target {index + 1:03d}/{count:03d}] {row['status']} "
            f"target_l2={row.get('target_error_l2')} "
            f"design_l2={row.get('design_error_l2')} elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )
    return rows, {
        "executed_target_count": executed_count,
        "recovered_target_count": recovered_count,
        "target_processing_wall_seconds": perf_counter() - start,
        "sum_recorded_target_runtime_seconds": sum(
            float(row["runtime_seconds"]) for row in rows
        ),
    }


def replay_audit(
    *,
    config: dict[str, Any],
    theta_bank: torch.Tensor,
    campaign_science: ScientificIdentity,
    rows: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    evidence = []
    for index in replay_indices(len(rows)):
        replay = run_target(
            index=index,
            theta_star=theta_bank[index],
            config=config,
            campaign_science=campaign_science,
            device=device,
            dtype=dtype,
        )
        primary_science = rows[index]["scientific_projection"]
        equal = primary_science is not None and (
            canonical_json_bytes(primary_science)
            == canonical_json_bytes(replay["scientific_projection"])
            and canonical_json_bytes(rows[index].get("assessment_response"))
            == canonical_json_bytes(replay.get("assessment_response"))
        )
        evidence.append(
            {
                "target_index": index,
                "equal": equal,
                "primary_hash": stable_hash("cohort-replay-v1", primary_science),
                "replay_hash": stable_hash(
                    "cohort-replay-v1", replay["scientific_projection"]
                ),
            }
        )
    return {"policy": "first_middle_last", "indices": replay_indices(len(rows)), "results": evidence}
