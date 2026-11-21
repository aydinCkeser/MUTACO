"""Zero-simulator-call terminal-domain ablation on archived MUTACO trajectories."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
from statistics import fmean, median
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from mutaco import (
    FrozenResponseTransform,
    GPFitConfig,
    GPFitStatus,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    SurrogateTrainingData,
    fit_exact_multitask_gp,
    physical_mae_scores,
    stable_hash,
)
from mutaco.recommendation.physical_mae import recommend_continuous_physical_mae
from mutaco_graph import SCIENTIFIC_M20, parameter_profile


SCHEMA = "mutaco-terminal-domain-ablation-b40-refit-v1"
ROW_SCHEMA = f"{SCHEMA}-row"
DOMAIN = "terminal-domain-ablation-b40-refit-v1"
B_POLICY = "B_prime_evaluated_posterior_mean_physical_mae_v1"
NEAR_EQUALITY_TOLERANCE = 1.0e-6
THREADS = 8


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_without_timings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _semantic_without_timings(item)
            for key, item in value.items()
            if not key.endswith("_seconds") and key not in {"component_timings", "timing"}
        }
    if isinstance(value, list):
        return [_semantic_without_timings(item) for item in value]
    return value


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires values")
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("distribution requires values")
    return {
        "minimum": min(values),
        "q05": _quantile(values, 0.05),
        "q25": _quantile(values, 0.25),
        "median": median(values),
        "mean": fmean(values),
        "q75": _quantile(values, 0.75),
        "q95": _quantile(values, 0.95),
        "maximum": max(values),
    }


def evaluated_posterior_mean_physical_mae(
    *,
    surrogate: Any,
    evaluated_theta: torch.Tensor,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
) -> dict[str, Any]:
    """Select the earliest evaluated point minimizing the canonical score."""

    posterior = surrogate.posterior_mean(evaluated_theta.contiguous()).mean
    predicted, scores = physical_mae_scores(
        posterior,
        transform=transform,
        target_physical=target_physical,
    )
    if scores.ndim != 1 or scores.shape[0] != evaluated_theta.shape[0]:
        raise ValueError("evaluated-domain scores have the wrong shape")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("evaluated-domain scores are nonfinite")
    index = int(torch.argmin(scores).item())
    return {
        "policy_id": B_POLICY,
        "candidate_domain": "archived_evaluated_theta_only",
        "selected_evaluation_index": index,
        "selected_theta": evaluated_theta[index].detach().cpu().tolist(),
        "predicted_physical_response": predicted[index].detach().cpu().tolist(),
        "score": float(scores[index].detach().cpu()),
        "tie_break": "earliest_evaluation_index",
        "posterior_calls": 1,
        "simulator_calls": 0,
    }


def normalized_distances(
    b_theta: torch.Tensor,
    c_theta: torch.Tensor,
    evaluated_theta: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, Any]:
    scale = upper - lower
    b_norm = (b_theta - lower) / scale
    c_norm = (c_theta - lower) / scale
    evaluated_norm = (evaluated_theta - lower) / scale
    difference = c_norm - b_norm
    nearest = torch.linalg.vector_norm(evaluated_norm - c_norm, dim=-1)
    return {
        "b_to_c_l2": float(torch.linalg.vector_norm(difference).detach().cpu()),
        "b_to_c_linf": float(difference.abs().max().detach().cpu()),
        "c_to_nearest_evaluated_l2": float(nearest.min().detach().cpu()),
        "exact_theta_equal": bool(torch.equal(b_theta, c_theta)),
        "near_theta_equal": bool(difference.abs().max() <= NEAR_EQUALITY_TOLERANCE),
        "near_equality_tolerance": NEAR_EQUALITY_TOLERANCE,
    }


def replay_target(
    *,
    record: Mapping[str, Any],
    target_position: int,
    target: Mapping[str, Any],
    configuration: Mapping[str, Any],
    population: Mapping[str, Any],
    source_identity: str,
) -> dict[str, Any]:
    started = perf_counter()
    trajectory = record["decision"]["trajectory"]
    if record.get("method") != "mutaco" or len(trajectory) != 40:
        raise ValueError("source is not one B40 MUTACO trajectory")
    target_id = str(record["target_id"])
    if target.get("target_record_id") != target_id:
        raise ValueError("cohort target identity mismatch")

    dtype = torch.float32
    device = torch.device("cpu")
    profile = parameter_profile(str(configuration["simulator"]["parameter_profile"]))
    parameter_spec = profile.parameter_spec
    response_spec = SCIENTIFIC_M20.response_spec
    theta = torch.tensor(
        [item["theta"] for item in trajectory], dtype=dtype, device=device
    ).contiguous()
    response = torch.tensor(
        [item["response_physical"] for item in trajectory], dtype=dtype, device=device
    ).contiguous()
    target_response = torch.tensor(
        target["target_response"], dtype=dtype, device=device
    ).contiguous()
    observation_ids = tuple(
        stable_hash(
            "terminal-ablation-archived-observation-v1",
            {
                "source_identity": source_identity,
                "target_id": target_id,
                "evaluation_index": index,
                "theta": item["theta"],
                "response_physical": item["response_physical"],
            },
        )
        for index, item in enumerate(trajectory)
    )
    transform = FrozenResponseTransform.fit(
        response[:16],
        response_spec=response_spec,
        observation_ids=observation_ids[:16],
        scale_floor=float(configuration["runner"]["transform"]["scale_floor"]),
    )
    standardized = transform.to_standardized(
        response, response_names=transform.metadata.response_names
    ).contiguous()
    data = SurrogateTrainingData(
        theta=theta,
        response_standardized=standardized,
        parameter_spec=parameter_spec,
        observation_ids=observation_ids,
        transform_id=transform.metadata.transform_id,
    )
    science = ScientificIdentity(
        stable_hash(
            DOMAIN,
            {
                "source_identity": source_identity,
                "target_position": target_position,
                "target_id": target_id,
                "training_data_id": data.identity(),
            },
        )
    )
    fit_deriver = SeedDeriver(
        science, int(population["comparison"]["inference_campaign_seed"])
    )
    fit_config = GPFitConfig.from_seed_deriver(
        fit_deriver,
        SeedCoordinate(
            "terminal_ablation_final_gp_refit",
            (target_position, 40, "all_observations_cold_single_attempt"),
        ),
        **dict(configuration["runner"]["gp"]),
    )
    fit_started = perf_counter()
    fit = fit_exact_multitask_gp(data, config=fit_config)
    fit_elapsed = perf_counter() - fit_started
    if fit.diagnostic.status is not GPFitStatus.SUCCESS or fit.surrogate is None:
        raise RuntimeError(
            f"final GP fit failed: {fit.diagnostic.failure_type}: "
            f"{fit.diagnostic.failure_message}"
        )
    surrogate = fit.surrogate
    b = evaluated_posterior_mean_physical_mae(
        surrogate=surrogate,
        evaluated_theta=theta,
        transform=transform,
        target_physical=target_response,
    )
    terminal_deriver = SeedDeriver(
        science, int(population["comparison"]["terminal_campaign_seed"])
    )
    c = recommend_continuous_physical_mae(
        surrogate=surrogate,
        transform=transform,
        target_physical=target_response,
        target_id=target_id,
        parameter_spec=parameter_spec,
        seed_deriver=terminal_deriver,
        target_index=target_position,
    ).document()
    if b["simulator_calls"] != 0 or c["simulator_calls"] != 0:
        raise RuntimeError("terminal policy violated zero-simulator-call boundary")
    b_theta = torch.tensor(b["selected_theta"], dtype=dtype, device=device)
    c_theta = torch.tensor(c["selected_theta"], dtype=dtype, device=device)
    lower = torch.tensor(parameter_spec.lower, dtype=dtype, device=device)
    upper = torch.tensor(parameter_spec.upper, dtype=dtype, device=device)
    distances = normalized_distances(
        b_theta, c_theta, theta, lower=lower, upper=upper
    )
    semantic = {
        "schema_version": ROW_SCHEMA,
        "target_position": target_position,
        "target_id": target_id,
        "source_record_identity": record["record_identity"],
        "source_identity": source_identity,
        "training_data_identity": data.identity(),
        "transform_identity": transform.metadata.transform_id,
        "model_identity": surrogate.model_id,
        "fit_configuration": fit_config.provenance(),
        "policy_b_prime": b,
        "policy_c_prime": _semantic_without_timings(c),
        "c_minus_b_posterior_physical_mae": float(c["score"] - b["score"]),
        "normalized_distance": distances,
        "simulator_constructed": False,
        "simulator_calls": 0,
    }
    return {
        **semantic,
        "row_identity": stable_hash(f"{DOMAIN}-row", semantic),
        "timing": {
            "gp_fit_seconds": fit_elapsed,
            "continuous_terminal_seconds": float(c["elapsed_seconds"]),
            "total_seconds": perf_counter() - started,
        },
        "fit_diagnostics": {
            "status": fit.diagnostic.status.value,
            "warnings": list(fit.diagnostic.warnings),
            "validation_posterior_calls": fit.diagnostic.validation_posterior_calls,
            "recovery_attempts": len(fit.diagnostic.recovery_attempts),
        },
        "continuous_optimizer": {
            "returned_candidate_source": c["returned_candidate_source"],
            "selected_restart_index": c["selected_restart_index"],
            "failed_restarts": sum(
                item["termination"] == "failed" for item in c["restart_histories"]
            ),
            "restart_histories": c["restart_histories"],
        },
    }


def run_population(
    *,
    declaration_path: Path,
    output_dir: Path,
    positions: set[int] | None = None,
) -> dict[str, Any]:
    declaration = _read(declaration_path)
    root = Path.cwd().resolve()
    source = declaration["source"]
    compact_path = resolve_recorded_path(root, source["compact_export"])
    actual_hash = _hash(compact_path)
    if actual_hash != source["compact_export_sha256"]:
        raise ValueError("compact export hash mismatch")
    compact = _read(compact_path)
    configuration = _read(resolve_recorded_path(root, source["configuration"]))
    population = _read(resolve_recorded_path(root, source["population_declaration"]))
    target_ids = list(population["target_record_identities"])
    mutaco = {record["target_id"]: record for record in compact["records"] if record["method"] == "mutaco"}
    selected = sorted(positions if positions is not None else range(120))
    if len(target_ids) != 120 or len(mutaco) != 120:
        raise ValueError("source population is incomplete")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    wall_started = perf_counter()
    for ordinal, position in enumerate(selected, start=1):
        try:
            target_id = target_ids[position]
            row = replay_target(
                record=mutaco[target_id],
                target_position=position,
                target=_read(resolve_recorded_path(root, source["cohort"]) / f"target_{position:03d}.json"),
                configuration=configuration,
                population=population,
                source_identity=actual_hash,
            )
            rows.append(row)
            print(json.dumps({"ordinal": ordinal, "total": len(selected), "target_position": position, "status": "completed", "delta": row["c_minus_b_posterior_physical_mae"]}, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {"target_position": position, "failure_type": type(exc).__name__, "failure_explanation": str(exc)}
            failures.append(failure)
            print(json.dumps({**failure, "status": "failed"}, sort_keys=True), flush=True)
    completed = len(rows)
    deltas = [row["c_minus_b_posterior_physical_mae"] for row in rows]
    displacements = [row["normalized_distance"]["b_to_c_l2"] for row in rows]
    nearest = [row["normalized_distance"]["c_to_nearest_evaluated_l2"] for row in rows]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "declaration": str(declaration_path.relative_to(root)),
        "declaration_sha256": _hash(declaration_path),
        "source_compact_export_sha256": actual_hash,
        "selected_targets": len(selected),
        "completed_targets": completed,
        "failed_targets": len(failures),
        "failures": failures,
        "exact_theta_equal": sum(row["normalized_distance"]["exact_theta_equal"] for row in rows),
        "near_theta_equal": sum(row["normalized_distance"]["near_theta_equal"] for row in rows),
        "c_prime_better": sum(value < 0.0 for value in deltas),
        "posterior_mae_c_minus_b": distribution(deltas) if deltas else None,
        "normalized_b_to_c_l2": distribution(displacements) if displacements else None,
        "normalized_c_to_nearest_evaluated_l2": distribution(nearest) if nearest else None,
        "raw_fallbacks": sum(row["continuous_optimizer"]["returned_candidate_source"] == "raw_fallback" for row in rows),
        "targets_with_failed_restarts": sum(row["continuous_optimizer"]["failed_restarts"] > 0 for row in rows),
        "total_gp_fitting_runtime_seconds": sum(row["timing"]["gp_fit_seconds"] for row in rows),
        "total_wall_runtime_seconds": perf_counter() - wall_started,
        "simulator_constructed": False,
        "simulator_calls": 0,
        "heldout_assessment_calls": 0,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "botorch": version("botorch"),
            "gpytorch": version("gpytorch"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "device": "cpu",
            "dtype": "float32",
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
    }
    semantic = _semantic_without_timings(summary)
    summary["package_identity"] = stable_hash(DOMAIN, {"summary": semantic, "row_identities": [row["row_identity"] for row in rows]})
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    summary["rows_sha256"] = _hash(rows_path)
    (output_dir / "manifest.json").write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-position", type=int, action="append")
    arguments = parser.parse_args()
    required_environment = {
        "OMP_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS),
        "OPENBLAS_NUM_THREADS": str(THREADS),
    }
    if any(os.environ.get(name) != value for name, value in required_environment.items()):
        raise SystemExit(f"terminal ablation requires thread environment {required_environment}")
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    positions = None if arguments.target_position is None else set(arguments.target_position)
    summary = run_population(
        declaration_path=arguments.declaration.resolve(),
        output_dir=arguments.output_dir.resolve(),
        positions=positions,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    if summary["failed_targets"]:
        raise SystemExit(1)


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.terminal_domain_ablation")


if __name__ == "__main__":
    main()
