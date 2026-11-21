"""Housekeeping for the human-readable analytic target-matching experiments.

The readable protocols live beside this module. This module isolates
configuration, profile construction, replay evidence, serialization, and
dependency-free SVG rendering from those protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import torch

from mutaco import (
    RunnerProgressEvent,
    RunnerStatus,
    SequentialRunResult,
    SequentialRunnerConfig,
    SequentialScientificIntent,
    canonical_json_bytes,
    checked_evaluate,
    run_sequential,
    stable_hash,
    strict_metadata_document,
)
from mutaco_analytic import (
    THETA_STAR,
    FrozenGPLandscapeConfig,
    FrozenGPLandscapeSimulator,
    FrozenGPLandscapeState,
    HandwrittenAnalyticSimulator,
)


SCHEMA = "mutaco-analytic-target-matching-v1"
LEGACY_SCHEMA = "mutaco-prompt5-acceptance-v1"


@dataclass(frozen=True)
class AcceptanceProfile:
    name: str
    simulator: Any
    hidden_theta: torch.Tensor
    hidden_theta_provenance: Mapping[str, Any]
    target_response: torch.Tensor
    target_construction_seconds: float
    intent: SequentialScientificIntent


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") not in {SCHEMA, LEGACY_SCHEMA}:
        raise ValueError("unsupported analytic target-matching configuration schema")
    return document


def _hidden_gp_theta(
    *,
    parameter_dim: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(
        dimension=parameter_dim,
        scramble=True,
        seed=seed,
    )
    unit = engine.draw(1, dtype=dtype).squeeze(0)
    return (-1.0 + 2.0 * unit).to(device=device).contiguous()


def _intent(
    simulator: Any,
    *,
    target: torch.Tensor,
    profile: Mapping[str, Any],
    runner: Mapping[str, Any],
    dtype: torch.dtype,
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
        initial_design_policy="scrambled_sobol",
        initial_point_count=int(profile["initial_points"]),
        caller_initial_points=None,
        sequential_rounds=int(profile["sequential_rounds"]),
        transform_settings=runner["transform"],
        gp_settings=runner["gp"],
        acquisition_settings=runner["acquisition"],
        recommendation_policy=str(runner["recommendation_policy"]),
        dtype=dtype,
        root_seed=int(profile["root_seed"]),
        seed_coordinate_schema={
            "initial_design": ["phase", "repeat"],
            "simulator_evaluation": ["purpose", "logical_index", "repeat"],
            "gp_fit": ["training_version"],
            "raw_candidates": ["sequential_round"],
        },
    )


def prepare_profiles(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[AcceptanceProfile, ...]:
    runner = config["runner"]
    declared_profiles = config["profiles"]
    unknown = set(declared_profiles) - {"handwritten", "gp_landscape"}
    if unknown:
        raise ValueError(f"unknown analytic target-matching profiles: {sorted(unknown)}")
    profiles = []

    if "handwritten" in declared_profiles:
        profiles.append(
            prepare_handwritten_profile(
                declared_profiles["handwritten"],
                runner=runner,
                device=device,
                dtype=dtype,
            )
        )
    if "gp_landscape" in declared_profiles:
        profiles.append(
            prepare_gp_landscape_profile(
                declared_profiles["gp_landscape"],
                runner=runner,
                device=device,
                dtype=dtype,
            )
        )
    if not profiles:
        raise ValueError("at least one analytic target-matching profile is required")
    return tuple(profiles)


def prepare_handwritten_profile(
    profile: Mapping[str, Any],
    *,
    runner: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> AcceptanceProfile:
    """Construct the transparent handwritten hidden-target experiment."""

    handwritten = HandwrittenAnalyticSimulator(device=device, dtype=dtype)
    handwritten_theta = torch.tensor(
        profile["hidden_theta"],
        dtype=dtype,
        device=device,
    ).contiguous()
    if tuple(handwritten_theta.detach().cpu().tolist()) != THETA_STAR:
        raise ValueError("handwritten hidden theta must match THETA_STAR")
    handwritten_target_start = perf_counter()
    handwritten_target = checked_evaluate(
        handwritten,
        handwritten_theta,
        device=device,
        dtype=dtype,
    ).values
    handwritten_target_seconds = perf_counter() - handwritten_target_start

    return AcceptanceProfile(
        "handwritten",
        handwritten,
        handwritten_theta,
        profile["hidden_theta_provenance"],
        handwritten_target,
        handwritten_target_seconds,
        _intent(
            handwritten,
            target=handwritten_target,
            profile=profile,
            runner=runner,
            dtype=dtype,
        ),
    )


def prepare_gp_landscape_profile(
    profile: Mapping[str, Any],
    *,
    runner: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> AcceptanceProfile:
    """Construct the deterministic frozen-GP hidden-target experiment."""

    state = FrozenGPLandscapeState.generate(
        FrozenGPLandscapeConfig(
            parameter_dim=int(profile["parameter_dim"]),
            response_dim=int(profile["response_dim"]),
            feature_count=int(profile["feature_count"]),
            latent_rank=int(profile["latent_rank"]),
        ),
        landscape_seed=int(profile["landscape_seed"]),
    )
    gp_landscape = FrozenGPLandscapeSimulator(state, device=device, dtype=dtype)
    gp_theta = _hidden_gp_theta(
        parameter_dim=state.config.parameter_dim,
        seed=int(profile["target_seed"]),
        dtype=dtype,
        device=device,
    )
    gp_target_start = perf_counter()
    gp_target = checked_evaluate(
        gp_landscape,
        gp_theta,
        device=device,
        dtype=dtype,
    ).values
    gp_target_seconds = perf_counter() - gp_target_start
    gp_provenance = {
        **profile["hidden_theta_provenance"],
        "target_seed": int(profile["target_seed"]),
        "landscape_seed": state.landscape_seed,
        "landscape_state_hash": state.fingerprint,
        "bounds": {
            "lower": list(state.config.lower),
            "upper": list(state.config.upper),
        },
    }
    return AcceptanceProfile(
        "gp_landscape",
        gp_landscape,
        gp_theta,
        gp_provenance,
        gp_target,
        gp_target_seconds,
        _intent(
            gp_landscape,
            target=gp_target,
            profile=profile,
            runner=runner,
            dtype=dtype,
        ),
    )


def _progress_printer(profile: str, replay: bool) -> Callable[[RunnerProgressEvent], None]:
    label = f"{profile}{' replay' if replay else ''}"

    def show(event: RunnerProgressEvent) -> None:
        if event.kind.value in {
            "initial_design_ready",
            "simulator_attempt_finished",
            "gp_fit_finished",
            "proposal_finished",
            "run_finished",
        }:
            print(
                f"[analytic target matching | {label} | "
                f"{event.elapsed_seconds:8.3f}s] {event.message}"
            )

    return show


def _vector_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    residual = actual - expected
    return {
        "absolute_per_component": residual.abs().detach().cpu().tolist(),
        "l2": float(torch.linalg.vector_norm(residual).detach().cpu()),
        "linf": float(residual.abs().max().detach().cpu()),
    }


def scientific_projection(result: SequentialRunResult) -> dict[str, Any]:
    """Replay payload with timing, machine, and execution placement removed."""

    assert result.initial_design is not None
    assert result.transform is not None
    assert result.standardized_target is not None
    assert result.recommendation is not None
    return {
        "status": result.status.value,
        "scientific_identity": result.scientific_identity.value,
        "initial_design": {
            "design_id": result.initial_design.design_id,
            "observation_ids": result.initial_design.observation_ids,
            "theta": result.initial_design.theta.detach().cpu().tolist(),
        },
        "transform": result.transform.metadata.provenance_document(),
        "standardized_target": result.standardized_target.values.detach().cpu().tolist(),
        "observations": tuple(
            {
                "observation_id": item.observation_id,
                "phase": item.phase.value,
                "logical_index": item.logical_index,
                "theta": item.theta.detach().cpu().tolist(),
                "response_physical": item.response_physical.detach().cpu().tolist(),
                "response_standardized": item.response_standardized.detach().cpu().tolist(),
                "simulator_metadata": strict_metadata_document(item.simulator_metadata),
                "evaluation_seed": item.context.evaluation_seed,
            }
            for item in result.observations
        ),
        "fits": tuple(
            {
                "training_version": item.training_version,
                "fit_id": item.diagnostic.fit_id,
                "training_data_id": item.diagnostic.training_data_id,
                "model_id": item.diagnostic.model_id,
                "configuration": item.diagnostic.configuration,
            }
            for item in result.fits
        ),
        "proposals": tuple(
            {
                "candidate_id": item.candidate_id,
                "candidate": item.candidate.detach().cpu().tolist(),
                "candidate_source": item.candidate_source.value,
                "branch": item.branch.value,
                "branch_reason": item.branch_reason.value,
                "pool_id": item.pool.pool_id,
                "seed_value": item.pool.seed_value,
                "raw_rankings": tuple(
                    {
                        "rank": rank.rank,
                        "raw_index": rank.raw_index,
                        "score": rank.score,
                        "margin": rank.margin,
                    }
                    for rank in item.raw_rankings
                ),
                "posterior_calls": item.posterior_calls.provenance(),
                "duplicate": item.duplicate.provenance() if item.duplicate else None,
            }
            for item in result.proposals
        ),
        "recommendation": {
            "selected_index": result.recommendation.selected_index,
            "selected_observation_id": result.recommendation.selected_observation_id,
            "selected_theta": result.recommendation.selected_theta.detach().cpu().tolist(),
            "score": float(result.recommendation.score.detach().cpu()),
            "posterior_model_id": result.recommendation.posterior_model_id,
        },
        "logical_accounting": result.logical_accounting,
        "physical_count_accounting": {
            key: value
            for key, value in result.physical_accounting.items()
            if not key.endswith("_seconds")
        },
    }


_scientific_projection = scientific_projection


def _history(
    profile: AcceptanceProfile,
    result: SequentialRunResult,
    target_response: torch.Tensor,
) -> dict[str, Any]:
    evaluations = []
    for observation in result.observations:
        evaluations.append(
            {
                "logical_index": observation.logical_index,
                "phase": observation.phase.value,
                "observation_id": observation.observation_id,
                "theta": observation.theta.detach().cpu().tolist(),
                "response": observation.response_physical.detach().cpu().tolist(),
                "parameter_error_to_hidden": _vector_errors(
                    observation.theta, profile.hidden_theta
                ),
                "response_error_to_target": _vector_errors(
                    observation.response_physical, target_response
                ),
                "attempts": tuple(item.provenance() for item in observation.attempts),
            }
        )
    acquisitions = tuple(
        {
            "round": index,
            "branch": proposal.branch.value,
            "branch_reason": proposal.branch_reason.value,
            "candidate_source": proposal.candidate_source.value,
            "candidate_id": proposal.candidate_id,
            "candidate": proposal.candidate.detach().cpu().tolist(),
            "raw_counts": proposal.raw_counts,
            "posterior_calls": proposal.posterior_calls.provenance(),
            "nearest_training_distance": (
                proposal.duplicate.nearest_training_distance
                if proposal.duplicate is not None
                else None
            ),
        }
        for index, proposal in enumerate(result.proposals)
    )
    return {"evaluations": evaluations, "acquisitions": acquisitions}


def _svg_polyline(
    *,
    title: str,
    series: Sequence[tuple[str, Sequence[float]]],
    y_label: str,
    width: int = 900,
    height: int = 480,
) -> str:
    margin_left, margin_right, margin_top, margin_bottom = 82, 24, 58, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_values = [float(value) for _, values in series for value in values]
    y_min, y_max = min(all_values), max(all_values)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    pad = 0.08 * (y_max - y_min)
    y_min, y_max = y_min - pad, y_max + pad
    count = max(len(values) for _, values in series)

    def x(index: int) -> float:
        return margin_left + (0 if count == 1 else index * plot_w / (count - 1))

    def y(value: float) -> float:
        return margin_top + (y_max - value) * plot_h / (y_max - y_min)

    colors = ("#2368a2", "#d1495b", "#2a9d8f", "#e9c46a", "#7b2cbf", "#f77f00")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" '
        f'x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<text x="{width / 2}" y="{height-20}" text-anchor="middle" '
        'font-family="sans-serif" font-size="14">evaluation index</text>',
        f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
    ]
    for tick in range(5):
        value = y_min + tick * (y_max - y_min) / 4
        yy = y(value)
        parts.extend(
            (
                f'<line x1="{margin_left}" y1="{yy:.2f}" x2="{width-margin_right}" '
                'y2="{:.2f}" stroke="#e5e7eb"/>'.format(yy),
                f'<text x="{margin_left-8}" y="{yy+4:.2f}" text-anchor="end" '
                f'font-family="monospace" font-size="11">{value:.4g}</text>',
            )
        )
    for series_index, (label, values) in enumerate(series):
        color = colors[series_index % len(colors)]
        points = " ".join(f"{x(i):.2f},{y(float(v)):.2f}" for i, v in enumerate(values))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.2"/>'
        )
        for i, value in enumerate(values):
            parts.append(
                f'<circle cx="{x(i):.2f}" cy="{y(float(value)):.2f}" r="3" '
                f'fill="{color}"/>'
            )
        legend_y = 48 + series_index * 18
        parts.append(
            f'<text x="{width-210}" y="{legend_y}" font-family="sans-serif" '
            f'font-size="12" fill="{color}">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _branch_svg(profile_name: str, acquisitions: Sequence[Mapping[str, Any]]) -> str:
    width, height = 900, 260
    left, right, axis_y = 80, 40, 135
    count = len(acquisitions)
    step = (width - left - right) / max(count - 1, 1)
    colors = {"barrier": "#2368a2", "exploration": "#d1495b"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20">{html.escape(profile_name)} acquisition branches</text>',
        f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" '
        'stroke="#333" stroke-width="2"/>',
    ]
    for index, item in enumerate(acquisitions):
        xx = left + index * step
        branch = str(item["branch"])
        color = colors.get(branch, "#555")
        parts.extend(
            (
                f'<circle cx="{xx:.2f}" cy="{axis_y}" r="13" fill="{color}"/>',
                f'<text x="{xx:.2f}" y="{axis_y+4}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11" fill="white">{index}</text>',
                f'<text x="{xx:.2f}" y="{axis_y+34}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11">{html.escape(branch)}</text>',
                f'<text x="{xx:.2f}" y="{axis_y-25}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="9">'
                f'{html.escape(str(item["candidate_source"]))}</text>',
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _write_plots(
    profile_dir: Path,
    profile: AcceptanceProfile,
    history: Mapping[str, Any],
) -> None:
    evaluations = history["evaluations"]
    theta_series = tuple(
        (
            f"theta_{index + 1}",
            [item["theta"][index] for item in evaluations],
        )
        for index in range(profile.hidden_theta.numel())
    )
    errors = (
        (
            "parameter L2",
            [item["parameter_error_to_hidden"]["l2"] for item in evaluations],
        ),
        (
            "response L2",
            [item["response_error_to_target"]["l2"] for item in evaluations],
        ),
    )
    plots = profile_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "parameter_trajectory.svg").write_text(
        _svg_polyline(
            title=f"{profile.name}: evaluated parameter trajectory",
            series=theta_series,
            y_label="physical parameter value",
        ),
        encoding="utf-8",
    )
    (plots / "error_trajectory.svg").write_text(
        _svg_polyline(
            title=f"{profile.name}: target-matching errors",
            series=errors,
            y_label="Euclidean error",
        ),
        encoding="utf-8",
    )
    (plots / "acquisition_branches.svg").write_text(
        _branch_svg(profile.name, history["acquisitions"]),
        encoding="utf-8",
    )


def run_and_assess_profile(
    profile: AcceptanceProfile,
    *,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    profile_dir = output / profile.name

    runner_config = SequentialRunnerConfig(profile.intent, device=device)
    primary = run_sequential(
        profile.simulator,
        config=runner_config,
        progress_callback=_progress_printer(profile.name, False),
    )
    if primary.status is not RunnerStatus.COMPLETED:
        raise RuntimeError(
            f"{profile.name} primary run failed: {primary.status.value}: "
            f"{primary.failure_explanation}"
        )
    replay = run_sequential(
        profile.simulator,
        config=runner_config,
        progress_callback=_progress_printer(profile.name, True),
    )
    if replay.status is not RunnerStatus.COMPLETED:
        raise RuntimeError(
            f"{profile.name} replay failed: {replay.status.value}: "
            f"{replay.failure_explanation}"
        )

    primary_science = _scientific_projection(primary)
    replay_science = _scientific_projection(replay)
    primary_bytes = canonical_json_bytes(primary_science)
    replay_bytes = canonical_json_bytes(replay_science)
    replay_equal = primary_bytes == replay_bytes
    replay_evidence = {
        "equal": replay_equal,
        "primary_hash": stable_hash(
            "analytic-target-matching-scientific-replay-v1", primary_science
        ),
        "replay_hash": stable_hash(
            "analytic-target-matching-scientific-replay-v1", replay_science
        ),
        "scope": (
            "same configuration, software, dtype, and device; excludes timings, "
            "machine provenance, and execution placement"
        ),
    }
    if not replay_equal:
        raise RuntimeError(f"{profile.name} deterministic scientific replay differed")

    assert primary.recommendation is not None
    recommendation = primary.recommendation
    recommended_observation = primary.observations[recommendation.selected_index]
    recommended_response = recommended_observation.response_physical
    target_response = profile.target_response
    history = _history(profile, primary, target_response)
    result_document = {
        "schema_version": SCHEMA,
        "profile": profile.name,
        "status": primary.status.value,
        "scientific_identity": primary.scientific_identity.value,
        "execution_identity": primary.execution_identity.value,
        "hidden_theta": profile.hidden_theta.detach().cpu().tolist(),
        "hidden_theta_provenance": strict_metadata_document(
            profile.hidden_theta_provenance
        ),
        "target_response": target_response.detach().cpu().tolist(),
        "recommended_theta": recommendation.selected_theta.detach().cpu().tolist(),
        "recommended_observation_id": recommendation.selected_observation_id,
        "recommended_response": recommended_response.detach().cpu().tolist(),
        "parameter_error": _vector_errors(
            recommendation.selected_theta, profile.hidden_theta
        ),
        "response_error": {
            "scope": "in_sample_evaluated_observation",
            "held_out_simulator_calls": 0,
            **_vector_errors(recommended_response, target_response),
        },
        "history": history,
        "logical_accounting": primary.logical_accounting,
        "physical_accounting": primary.physical_accounting,
        "timings": {
            **primary.timings,
            "target_construction_seconds": profile.target_construction_seconds,
            "simulator_seconds": primary.physical_accounting["simulator_seconds"],
            "gp_fit_seconds": primary.physical_accounting["gp_fit_seconds"],
            "proposal_seconds": primary.physical_accounting["proposal_seconds"],
        },
        "replay": replay_evidence,
        "runner_provenance": primary.provenance(),
    }
    result_document["target_construction"] = {
        "simulator_calls": 1,
        "budget_domain": "target_generation_outside_mutaco_inference",
        "elapsed_seconds": profile.target_construction_seconds,
    }
    write_json(profile_dir / "result.json", result_document)
    write_json(profile_dir / "scientific_history.json", primary_science)
    write_json(profile_dir / "replay_evidence.json", replay_evidence)
    _write_plots(profile_dir, profile, history)
    print(
        f"[analytic target matching | {profile.name}] recommended theta="
        f"{result_document['recommended_theta']}; parameter L2="
        f"{result_document['parameter_error']['l2']:.6g}; response L2="
        f"{result_document['response_error']['l2']:.6g} "
        "(in-sample evaluated observation)"
    )
    return result_document


def build_manifest(
    *,
    config: Mapping[str, Any],
    config_paths: Sequence[Path],
    runtime: Any,
    results: Sequence[Mapping[str, Any]],
    total_runtime_seconds: float,
    package_origins: Mapping[str, str],
    cuda_command: str,
) -> dict[str, Any]:
    """Build the execution manifest without making scientific decisions."""

    manifest = {
        "schema_version": SCHEMA,
        "milestone": "Prompt 5",
        "experiment": "analytic_target_matching",
        "description": "human-readable analytic target-matching experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "configuration_paths": tuple(path.as_posix() for path in config_paths),
        "runtime": runtime.provenance(),
        "package_origins": dict(package_origins),
        "cpu_fallback_demonstrated": (
            runtime.requested_device == "auto"
            and runtime.effective_device == "cpu"
            and runtime.fallback_reason == "cuda_unavailable"
        ),
        "same_cuda_capable_configuration": {
            "configuration_paths": tuple(path.as_posix() for path in config_paths),
            "command": cuda_command,
            "scientific_configuration_unchanged": True,
            "expected_difference": "execution provenance and floating-point results",
        },
        "profiles": tuple(
            {
                "name": item["profile"],
                "scientific_identity": item["scientific_identity"],
                "status": item["status"],
                "parameter_l2_error": item["parameter_error"]["l2"],
                "in_sample_response_l2_error": item["response_error"]["l2"],
                "response_error_scope": item["response_error"]["scope"],
                "target_generation_simulator_calls": item["target_construction"][
                    "simulator_calls"
                ],
                "replay_equal": item["replay"]["equal"],
            }
            for item in results
        ),
        "total_runtime_seconds": total_runtime_seconds,
    }
    return manifest
