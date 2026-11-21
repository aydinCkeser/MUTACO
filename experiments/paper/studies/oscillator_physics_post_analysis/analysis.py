"""Zero-call target-98 analysis and the gated eight-call sensitivity worker."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mutaco import SimulationContext, stable_hash
from mutaco.atomic_io import atomic_write_text
from mutaco_graph import GraphSimulationConfig, OverdampedScientificSimulator
from mutaco_graph.profiles import SCIENTIFIC_M20


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent
DECLARATION = PACKAGE / "configs/target098_sensitivity_v1.json"
DEFAULT_OUTPUT = ROOT / "experiments/paper/oscillator_physics_post_analysis/outputs/target098_v1"
TARGET_SHA256 = "2d05a098f42c6bd338a30d4e72d79e906dda55cf57f70156c30a22a14fdcc231"
CONFIG_SHA256 = "2614cdae3f2eb40ff1a0f0ee473056f05c8284d3ab59e2f159969ed1cc704a8a"
RECONSTRUCTION_RECORD = ROOT / (
    "runs/paper/graph_oscillator/graph_paper120_b40_mutaco_npe_v3_v1/"
    "worker-18/records/target-a12dae0d295ac006/mutaco.json"
)
RECONSTRUCTION_SHA256 = "4eb69e660b9ef9527ff9ee61d4803c23514a0fde3738c1cbed9a10b59fe8c207"
RECONSTRUCTION_RECORD_IDENTITY = (
    "paper-method-record-v1-"
    "76c8bf4b5a6ede5b1bcfc26142a86256c1c7f4d3a3a474e80868e3614749dd49"
)
RECONSTRUCTION_THETA = [
    1.9519236087799072,
    1.1798672676086426,
    1.1263821125030518,
    0.409557968378067,
]
AUTHORIZATION_PHRASE = "AUTHORIZE_EXACTLY_EIGHT_OSCILLATOR_SENSITIVITY_CALLS"


class PostAnalysisError(RuntimeError):
    """A fail-closed post-analysis contract violation."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostAnalysisError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise PostAnalysisError(f"refusing to overwrite append-only output: {path}")
    atomic_write_text(
        path,
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def load_declaration(path: Path = DECLARATION) -> dict[str, Any]:
    declaration = _read_json(path)
    if declaration.get("schema_version") != "mutaco-oscillator-physics-sensitivity-declaration-v1":
        raise PostAnalysisError("sensitivity declaration schema mismatch")
    identity = declaration.pop("declaration_identity", None)
    expected_identity = stable_hash(
        "mutaco-oscillator-physics-sensitivity-declaration-v1", declaration
    )
    declaration["declaration_identity"] = identity
    if identity != expected_identity:
        raise PostAnalysisError("sensitivity declaration identity mismatch")
    points = declaration.get("finite_difference", {}).get("ordered_points")
    if not isinstance(points, list) or len(points) != 8:
        raise PostAnalysisError("declaration must contain exactly eight ordered points")
    if [point.get("point_index") for point in points] != list(range(8)):
        raise PostAnalysisError("point indices must be exactly 0..7")
    if declaration.get("call_contract") != {
        "maximum_simulator_calls": 8,
        "required_successful_calls_for_jacobian": 8,
        "retry_calls": 0,
        "center_calls": 0,
        "calls_per_point": 1,
        "automatic_retry": False,
    }:
        raise PostAnalysisError("exact eight-call/no-retry contract mismatch")
    theta_star = torch.tensor(declaration["theta_star"], dtype=torch.float32)
    bounds = torch.tensor(declaration["parameter_bounds"], dtype=torch.float32)
    h = declaration["normalized_half_step"]
    normalized = (theta_star - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    for index in range(4):
        minus, plus = points[2 * index], points[2 * index + 1]
        if minus["sign"] != "minus" or plus["sign"] != "plus":
            raise PostAnalysisError("each parameter requires ordered minus/plus points")
        for coordinate in range(4):
            minus_normalized = normalized.clone()
            plus_normalized = normalized.clone()
            minus_normalized[index] -= h
            plus_normalized[index] += h
            expected_minus = (bounds[:, 0] + minus_normalized * (bounds[:, 1] - bounds[:, 0])).tolist()[coordinate]
            expected_plus = (bounds[:, 0] + plus_normalized * (bounds[:, 1] - bounds[:, 0])).tolist()[coordinate]
            if minus["theta"][coordinate] != expected_minus or plus["theta"][coordinate] != expected_plus:
                raise PostAnalysisError("declared finite-difference theta is not exact")
    crn = declaration["common_random_numbers"]
    if crn.get("policy") != "fixed_full_stream_crn_diagnostic_v1":
        raise PostAnalysisError("full-stream CRN policy mismatch")
    if set(crn.get("shared_namespaces", ())) != {
        "quenched_disorder", "initial_state", "thermal_noise"
    }:
        raise PostAnalysisError("full-stream CRN namespaces are incomplete")
    return declaration


def load_frozen_target(declaration: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_recorded_path(ROOT, declaration["target_record_path"])
    if _sha256(path) != TARGET_SHA256:
        raise PostAnalysisError("frozen target-98 byte hash mismatch")
    target = _read_json(path)
    if target.get("target_index") != 98:
        raise PostAnalysisError("frozen target position mismatch")
    if target.get("target_record_id") != declaration["target_record_id"]:
        raise PostAnalysisError("frozen target record identity mismatch")
    if target.get("theta_star") != declaration["theta_star"]:
        raise PostAnalysisError("audited theta_star mismatch")
    response = target.get("target_response")
    if not isinstance(response, list) or len(response) != 20:
        raise PostAnalysisError("target response must have exactly 20 components")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in response):
        raise PostAnalysisError("target response must be finite numeric data")
    return target


def load_frozen_reconstruction(declaration: Mapping[str, Any]) -> dict[str, Any]:
    if _sha256(RECONSTRUCTION_RECORD) != RECONSTRUCTION_SHA256:
        raise PostAnalysisError("frozen target-98 B40 MUTACO record byte hash mismatch")
    sidecar = RECONSTRUCTION_RECORD.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != RECONSTRUCTION_SHA256:
        raise PostAnalysisError("frozen target-98 B40 MUTACO hash sidecar mismatch")
    record = _read_json(RECONSTRUCTION_RECORD)
    if record.get("status") != "completed":
        raise PostAnalysisError("frozen target-98 B40 MUTACO record is incomplete")
    if record.get("target_record_identity") != declaration["target_record_id"]:
        raise PostAnalysisError("B40 MUTACO target identity mismatch")
    if record.get("record_content_hash") != RECONSTRUCTION_RECORD_IDENTITY:
        raise PostAnalysisError("B40 MUTACO record content identity mismatch")
    if record.get("recommended_theta") != RECONSTRUCTION_THETA:
        raise PostAnalysisError("illustrated B40 MUTACO recommendation mismatch")
    response = record.get("assessment_response")
    if not isinstance(response, list) or len(response) != 20:
        raise PostAnalysisError("B40 MUTACO assessment response must have 20 components")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in response):
        raise PostAnalysisError("B40 MUTACO assessment response must be finite numeric data")
    return record


def _trajectory_rows(
    response: Sequence[float], *, series: str
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    forces = SCIENTIFIC_M20.forces
    omega = response[:10]
    synchronization = response[10:]
    for force_index, force in enumerate(forces):
        for gauge_index, gauge_name in enumerate(("0", "pi")):
            offset = 2 * force_index + gauge_index
            rows.append(
                {
                    "series": series,
                    "force": float(force),
                    "gauge": gauge_name,
                    "omega_bar": float(omega[offset]),
                    "M": float(synchronization[offset]),
                }
            )
    return rows


def build_zero_call_documents(
    declaration: Mapping[str, Any], target: Mapping[str, Any], reconstruction: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [
        *_trajectory_rows(target["target_response"], series="target"),
        *_trajectory_rows(
            reconstruction["assessment_response"], series="mutaco_b40_reconstruction"
        ),
    ]
    trajectory_body = {
        "schema_version": "mutaco-oscillator-physics-trajectories-v1",
        "analysis_classification": "zero_call_frozen_archive_only",
        "scientific_calls": 0,
        "target_position": 98,
        "target_record_id": target["target_record_id"],
        "theta_star": target["theta_star"],
        "mutaco_b40_recommended_theta": reconstruction["recommended_theta"],
        "sources": {
            "target": {
                "path": declaration["target_record_path"],
                "sha256": TARGET_SHA256,
                "identity": target["target_record_id"],
            },
            "mutaco_b40_reconstruction": {
                "path": RECONSTRUCTION_RECORD.relative_to(ROOT).as_posix(),
                "sha256": RECONSTRUCTION_SHA256,
                "identity": reconstruction["record_content_hash"],
                "response_field": "assessment_response",
            },
        },
        "response_ordering": "quantity_then_force_major_then_gauge_0_pi",
        "rows": rows,
    }
    trajectories = {
        **trajectory_body,
        "trajectory_identity": stable_hash(
            "mutaco-oscillator-physics-trajectories-v1", trajectory_body
        ),
    }
    contrasts = []
    for series in ("target", "mutaco_b40_reconstruction"):
        for force in SCIENTIFIC_M20.forces:
            zero = next(row for row in rows if row["series"] == series and row["force"] == force and row["gauge"] == "0")
            pi = next(row for row in rows if row["series"] == series and row["force"] == force and row["gauge"] == "pi")
            contrasts.append(
                {
                    "series": series,
                    "force": float(force),
                    "omega_bar_pi_minus_zero": float(pi["omega_bar"] - zero["omega_bar"]),
                    "M_pi_minus_zero": float(pi["M"] - zero["M"]),
                }
            )
    contrast_body = {
        "schema_version": "mutaco-oscillator-physics-gauge-contrast-v1",
        "analysis_classification": "zero_call_frozen_archive_only",
        "scientific_calls": 0,
        "target_position": 98,
        "target_record_id": target["target_record_id"],
        "mutaco_b40_record_identity": reconstruction["record_content_hash"],
        "contrast_definition": "response(Phi=pi)-response(Phi=0)",
        "rows": contrasts,
    }
    gauge = {
        **contrast_body,
        "gauge_contrast_identity": stable_hash(
            "mutaco-oscillator-physics-gauge-contrast-v1", contrast_body
        ),
    }
    return gauge, trajectories


def render_preview(trajectories: Mapping[str, Any], destination: Path) -> str:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    rows = trajectories["rows"]
    colors = {"0": "#0072B2", "pi": "#D55E00"}
    series_style = {
        "target": ("-", "o", "Target"),
        "mutaco_b40_reconstruction": ("--", "s", "MUTACO B40"),
    }
    with mpl.rc_context({"font.family": "serif", "font.size": 11, "axes.grid": True, "grid.alpha": 0.2}):
        figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
        for axis, key, ylabel, panel in (
            (axes[0], "omega_bar", r"$\langle\bar{\omega}\rangle$", "(a)"),
            (axes[1], "M", r"$\langle M\rangle$", "(b)"),
        ):
            for series in ("target", "mutaco_b40_reconstruction"):
                linestyle, marker, series_label = series_style[series]
                for gauge in ("0", "pi"):
                    selected = [
                        row for row in rows
                        if row["series"] == series and row["gauge"] == gauge
                    ]
                    axis.plot(
                        [row["force"] for row in selected],
                        [row[key] for row in selected],
                        linestyle=linestyle, marker=marker, color=colors[gauge],
                        markerfacecolor=(colors[gauge] if series == "target" else "white"),
                        label=rf"{series_label}, $\Phi={gauge}$",
                    )
            axis.set_xlabel(r"Driving force $F$")
            axis.set_ylabel(ylabel)
            axis.text(0.02, 0.96, panel, transform=axis.transAxes, va="top", fontweight="bold")
        axes[1].set_ylim(0.0, 1.0)
        axes[0].legend(frameon=False, fontsize=8)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, metadata={"Software": "MUTACO zero-call replay"})
        plt.close(figure)
    return _sha256(destination)


def zero_call(output_root: Path) -> dict[str, Any]:
    declaration = load_declaration()
    target = load_frozen_target(declaration)
    reconstruction = load_frozen_reconstruction(declaration)
    gauge, trajectories = build_zero_call_documents(declaration, target, reconstruction)
    if output_root.exists():
        raise PostAnalysisError(f"refusing to overwrite output directory: {output_root}")
    output_root.mkdir(parents=True)
    gauge_path = output_root / "gauge_contrast.json"
    trajectory_path = output_root / "response_trajectories.json"
    figure_path = output_root / "target098_gauge_response_preview.png"
    _write_new_json(gauge_path, gauge)
    _write_new_json(trajectory_path, trajectories)
    figure_sha = render_preview(trajectories, figure_path)
    manifest_body = {
        "schema_version": "mutaco-oscillator-physics-zero-call-package-v1",
        "status": "complete_zero_call",
        "scientific_calls": 0,
        "target_position": 98,
        "artifacts": {
            "gauge_contrast.json": _sha256(gauge_path),
            "response_trajectories.json": _sha256(trajectory_path),
            figure_path.name: figure_sha,
        },
    }
    manifest = {
        **manifest_body,
        "package_identity": stable_hash(
            "mutaco-oscillator-physics-zero-call-package-v1", manifest_body
        ),
    }
    _write_new_json(output_root / "manifest.json", manifest)
    return manifest


def validate_preflight(output_root: Path) -> dict[str, Any]:
    declaration = load_declaration()
    load_frozen_target(declaration)
    config_path = ROOT / declaration["simulator"]["configuration_path"]
    if _sha256(config_path) != CONFIG_SHA256:
        raise PostAnalysisError("simulator configuration byte hash mismatch")
    if output_root.exists():
        raise PostAnalysisError("scientific output root must not exist before execution")
    return {
        "schema_version": "mutaco-oscillator-physics-preflight-v1",
        "status": "validated_zero_call",
        "scientific_calls_executed": 0,
        "job_submitted": False,
        "submission_authorized": False,
        "call_preview": {
            "maximum_simulator_calls": 8,
            "scientific_calls_executed": 0,
            "center_calls": 0,
            "retry_calls": 0,
        },
        "pending_theta": [point["theta"] for point in declaration["finite_difference"]["ordered_points"]],
        "evaluation_seed": declaration["common_random_numbers"]["evaluation_seed"],
    }


def _simulator(declaration: Mapping[str, Any]) -> OverdampedScientificSimulator:
    raw = _read_json(ROOT / declaration["simulator"]["configuration_path"])["simulator"]
    return OverdampedScientificSimulator(
        GraphSimulationConfig(
            model_identity=raw["identifier"], parameter_profile_id=raw["parameter_profile"],
            response_profile_id=raw["response_profile"], node_count=raw["node_count"],
            batch_size=raw["batch_size"], repeat_size=raw["repeat_size"],
            temperature=raw["temperature"], dt=raw["dt"],
            burn_in_steps=raw["burn_in_steps"], sample_steps=raw["sample_steps"],
            sample_stride=raw["sample_stride"], device="cpu", dtype=torch.float32,
        )
    )


def _context(declaration: Mapping[str, Any]) -> SimulationContext:
    crn = declaration["common_random_numbers"]
    return SimulationContext(
        crn["scientific_run_id"], crn["purpose"], crn["logical_index"],
        crn["repeat_index"], crn["evaluation_seed"], crn["attempt_index"],
    )


def _sensitivity_document(
    declaration: Mapping[str, Any], outcomes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(outcomes) != 8:
        raise PostAnalysisError("Jacobian requires exactly eight successful outcomes")
    bounds = declaration["parameter_bounds"]
    h = declaration["normalized_half_step"]
    physical_h = [h * (upper - lower) for lower, upper in bounds]
    normalized_columns = []
    physical_columns = []
    for parameter_index in range(4):
        minus = outcomes[2 * parameter_index]["response"]
        plus = outcomes[2 * parameter_index + 1]["response"]
        normalized_columns.append([(p - m) / (2.0 * h) for m, p in zip(minus, plus, strict=True)])
        physical_columns.append([(p - m) / (2.0 * physical_h[parameter_index]) for m, p in zip(minus, plus, strict=True)])
    jacobian_normalized = [list(row) for row in zip(*normalized_columns, strict=True)]
    jacobian_physical = [list(row) for row in zip(*physical_columns, strict=True)]
    scales = [5.0] * 10 + [1.0] * 10
    scaled = [[value / scales[row] for value in jacobian_normalized[row]] for row in range(20)]
    body = {
        "schema_version": "mutaco-oscillator-physics-sensitivity-v1",
        "target_position": 98,
        "parameter_names": declaration["parameter_names"],
        "response_names": list(SCIENTIFIC_M20.names),
        "normalized_half_step": h,
        "physical_half_steps": physical_h,
        "response_scales": scales,
        "jacobian_wrt_normalized": jacobian_normalized,
        "jacobian_wrt_physical": jacobian_physical,
        "scaled_jacobian_wrt_normalized": scaled,
        "point_outcome_identities": [outcome["outcome_identity"] for outcome in outcomes],
        "call_accounting": {"attempted_calls": 8, "successful_calls": 8, "failed_calls": 0, "retry_calls": 0, "center_calls": 0},
    }
    return {**body, "sensitivity_identity": stable_hash("mutaco-oscillator-physics-sensitivity-v1", body)}


def execute_worker(output_root: Path, authorization: str) -> dict[str, Any]:
    if authorization != AUTHORIZATION_PHRASE or os.environ.get("MUTACO_OSCILLATOR_EIGHT_CALLS_AUTHORIZED") != "1":
        raise PostAnalysisError("explicit eight-call authorization gate is closed")
    validate_preflight(output_root)
    declaration = load_declaration()
    output_root.mkdir(parents=True)
    points_root = output_root / "points"
    points_root.mkdir()
    simulator = _simulator(declaration)
    context = _context(declaration)
    outcomes = []
    reference_manifest = None
    for point in declaration["finite_difference"]["ordered_points"]:
        index = point["point_index"]
        intent = {
            "schema_version": "mutaco-oscillator-physics-point-intent-v1",
            "point_index": index, "theta": point["theta"],
            "evaluation_seed": context.evaluation_seed, "attempt_index": 0,
            "maximum_transport_calls": 1, "retry_calls": 0,
        }
        _write_new_json(points_root / f"point_{index:02d}.intent.json", intent)
        result = simulator.evaluate(torch.tensor(point["theta"], dtype=torch.float32), context)
        metadata = dict(result.metadata)
        if metadata.get("transport_call_count") != 1:
            raise PostAnalysisError("simulator transport-call accounting mismatch")
        manifest = metadata.get("seed_manifest")
        if reference_manifest is None:
            reference_manifest = manifest
        elif manifest != reference_manifest:
            raise PostAnalysisError("full-stream CRN seed manifest drifted across theta")
        body = {
            "schema_version": "mutaco-oscillator-physics-point-outcome-v1",
            "point_index": index, "parameter_index": point["parameter_index"],
            "parameter_name": point["parameter_name"], "sign": point["sign"],
            "theta": point["theta"], "response": result.values.detach().cpu().tolist(),
            "evaluation_seed": context.evaluation_seed, "seed_manifest": manifest,
            "transport_call_count": 1,
        }
        outcome = {**body, "outcome_identity": stable_hash("mutaco-oscillator-physics-point-outcome-v1", body)}
        _write_new_json(points_root / f"point_{index:02d}.outcome.json", outcome)
        outcomes.append(outcome)
    sensitivity = _sensitivity_document(declaration, outcomes)
    _write_new_json(output_root / "sensitivity.json", sensitivity)
    return sensitivity


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    zero = commands.add_parser("zero-call")
    zero.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    preview = commands.add_parser("preflight")
    preview.add_argument("--output-root", type=Path, required=True)
    worker = commands.add_parser("execute-worker")
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--authorization", required=True)
    args = parser.parse_args(argv)
    if args.command == "zero-call":
        print(json.dumps(zero_call(args.output_root), sort_keys=True))
    elif args.command == "preflight":
        print(json.dumps(validate_preflight(args.output_root), sort_keys=True))
    else:
        print(json.dumps(execute_worker(args.output_root, args.authorization), sort_keys=True))
    return 0


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.oscillator_physics_post_analysis.analysis")


if __name__ == "__main__":
    raise SystemExit(main())
