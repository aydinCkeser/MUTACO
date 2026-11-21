"""Zero-call design and explicitly gated worker for multi-target sensitivity."""

from __future__ import annotations

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
DECLARATION = PACKAGE / "configs/paper120_maximin12_sensitivity_v1.json"
COHORT_ROOT = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3"
COHORT_MANIFEST = COHORT_ROOT / "manifest.json"
CONFIG = ROOT / "experiments/graph_inversion/configs/paper/paper120_v3_target_config.json"
CONFIG_SHA256 = "2614cdae3f2eb40ff1a0f0ee473056f05c8284d3ab59e2f159969ed1cc704a8a"
COHORT_SHA256 = "359fc0bb59fc9a8a52f5da684e7a7eb352466bdc0d0a54fb8efa94131ce423bb"
PARAMETER_NAMES = (
    "weight_mean", "pairwise_coupling_mean", "bias_scale_mean",
    "pairwise_coupling_CV",
)
BOUNDS = ((0.5, 2.0), (0.5, 2.0), (0.5, 2.0), (0.0, 1.0))
TARGET_COUNT = 12
POINTS_PER_TARGET = 8
GLOBAL_CALL_CAP = 96
AUTHORIZATION_PHRASE = "AUTHORIZE_EXACTLY_96_PAPER120_OSCILLATOR_SENSITIVITY_CALLS"
AUTHORIZATION_ENV = "MUTACO_OSCILLATOR_96_CALLS_AUTHORIZED"
SEED_NAMESPACE = "oscillator-physics-paper120-maximin12-crn-v1"
TARGET098_DIAGNOSTIC_SEED = 9818052026


class StudyError(RuntimeError):
    """Fail-closed study contract violation."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StudyError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise StudyError(f"refusing to overwrite append-only output: {path}")
    atomic_write_text(
        path,
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _cohort() -> list[dict[str, Any]]:
    if _sha256(COHORT_MANIFEST) != COHORT_SHA256:
        raise StudyError("frozen Paper-120 cohort manifest byte hash mismatch")
    manifest = _read_json(COHORT_MANIFEST)
    if manifest.get("completed_count") != 120 or manifest.get("missing_count") != 0:
        raise StudyError("Paper-120 cohort is incomplete")
    records = []
    for index in range(120):
        path = COHORT_ROOT / f"target_{index:03d}.json"
        record = _read_json(path)
        if record.get("target_index") != index:
            raise StudyError("frozen target index mismatch")
        expected = manifest["records"][str(index)]["target_record_id"]
        if record.get("target_record_id") != expected:
            raise StudyError("frozen target identity mismatch")
        theta = record.get("theta_star")
        if not isinstance(theta, list) or len(theta) != 4:
            raise StudyError("frozen theta_star must have four coordinates")
        records.append({
            "target_index": index,
            "target_record_id": expected,
            "target_record_path": f"data/cohorts/paper120_v3/{path.name}",
            "target_record_sha256": _sha256(path),
            "theta_star": theta,
            "normalized_theta_star": [
                (float(value) - lower) / (upper - lower)
                for value, (lower, upper) in zip(theta, BOUNDS, strict=True)
            ],
        })
    return records


def select_targets(records: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], float]]:
    if len(records) != 120:
        raise StudyError("selection requires the complete 120-target cohort")
    center = (0.5, 0.5, 0.5, 0.5)
    first = min(
        records,
        key=lambda row: (math.dist(row["normalized_theta_star"], center), row["target_index"]),
    )
    selected: list[tuple[Mapping[str, Any], float]] = [
        (first, math.dist(first["normalized_theta_star"], center))
    ]
    while len(selected) < TARGET_COUNT:
        remaining = [
            row for row in records
            if all(row["target_index"] != chosen[0]["target_index"] for chosen in selected)
        ]
        distances = [
            (row, min(math.dist(row["normalized_theta_star"], chosen[0]["normalized_theta_star"])
                      for chosen in selected))
            for row in remaining
        ]
        selected.append(min(distances, key=lambda item: (-item[1], item[0]["target_index"])))
    return selected


def _evaluation_seed(target_index: int) -> int:
    payload = f"{SEED_NAMESPACE}/target={target_index:03d}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _target_design(row: Mapping[str, Any], rank: int, selection_distance: float) -> dict[str, Any]:
    theta = torch.tensor(row["theta_star"], dtype=torch.float32)
    bounds = torch.tensor(BOUNDS, dtype=torch.float32)
    normalized = (theta - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    coordinates = []
    points = []
    for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
        z = float(normalized[parameter_index])
        boundary_distance = min(z, 1.0 - z)
        nominal_h = min(0.05, 0.9 * boundary_distance)
        if nominal_h <= 0.0:
            raise StudyError("symmetric finite-difference half-step is not positive")
        pair = []
        for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
            shifted = normalized.clone()
            shifted[parameter_index] += torch.tensor(sign * nominal_h, dtype=torch.float32)
            persisted = (bounds[:, 0] + shifted * (bounds[:, 1] - bounds[:, 0])).tolist()
            if any(value < BOUNDS[a][0] or value > BOUNDS[a][1] for a, value in enumerate(persisted)):
                raise StudyError("constructed finite-difference point lies outside inference bounds")
            point = {
                "global_point_index": (rank - 1) * POINTS_PER_TARGET + len(points),
                "target_point_index": len(points),
                "parameter_index": parameter_index,
                "parameter_name": parameter_name,
                "sign": sign_name,
                "theta": persisted,
            }
            points.append(point)
            pair.append(persisted)
        actual_denominator = float(pair[1][parameter_index] - pair[0][parameter_index])
        if not actual_denominator > 0.0:
            raise StudyError("persisted plus-minus physical denominator is not positive")
        coordinates.append({
            "parameter_index": parameter_index,
            "parameter_name": parameter_name,
            "normalized_center": z,
            "nearest_boundary_distance": boundary_distance,
            "nominal_normalized_half_step": nominal_h,
            "boundary_reduced_from_0p05": nominal_h < 0.05,
            "persisted_float32_minus_theta": pair[0],
            "persisted_float32_plus_theta": pair[1],
            "actual_physical_denominator": actual_denominator,
        })
    seed = _evaluation_seed(int(row["target_index"]))
    if seed == TARGET098_DIAGNOSTIC_SEED:
        raise StudyError("multi-target CRN seed collides with target-98 diagnostic seed")
    return {
        "selection_rank": rank,
        "selection_distance": selection_distance,
        **dict(row),
        "coordinates": coordinates,
        "ordered_points": points,
        "common_random_numbers": {
            "policy": "target_addressed_full_stream_crn_v1",
            "scope": "all_eight_points_within_target_only",
            "scientific_run_id": f"oscillator-physics-paper120-maximin12-target{row['target_index']:03d}-v1",
            "purpose": "finite_difference_sensitivity",
            "logical_index": row["target_index"],
            "repeat_index": 0,
            "attempt_index": 0,
            "evaluation_seed": seed,
            "seed_policy": "sha256_first_u64_mask63_target_addressed_v1",
            "seed_namespace": SEED_NAMESPACE,
            "shared_namespaces": ["quenched_disorder", "initial_state", "thermal_noise"],
        },
    }


def build_declaration() -> dict[str, Any]:
    records = _cohort()
    selected = select_targets(records)
    targets = [_target_design(row, rank, distance) for rank, (row, distance) in enumerate(selected, 1)]
    seeds = [target["common_random_numbers"]["evaluation_seed"] for target in targets]
    if len(set(seeds)) != TARGET_COUNT:
        raise StudyError("target-addressed evaluation seeds are not unique")
    body = {
        "schema_version": "mutaco-oscillator-physics-multitarget-declaration-v1",
        "status": "prepared_unexecuted_zero_call",
        "selection": {
            "policy": "normalized_theta_star_greedy_farthest_point_maximin_v1",
            "information_inputs": ["frozen_theta_star", "p4_paper_baseline_parameter_bounds"],
            "forbidden_inputs": ["mutaco_performance", "parameter_error", "response_error", "sbi", "sensitivity", "manual_behavior"],
            "target_count": TARGET_COUNT,
            "selected_target_indices": [target["target_index"] for target in targets],
            "target098_selected": any(target["target_index"] == 98 for target in targets),
        },
        "cohort": {"manifest_path": "data/cohorts/paper120_v3/manifest.json", "manifest_sha256": COHORT_SHA256, "size": 120},
        "parameter_names": list(PARAMETER_NAMES),
        "parameter_bounds": [list(bound) for bound in BOUNDS],
        "finite_difference": {
            "policy": "boundary_aware_symmetric_central_float32_v1",
            "nominal_max_normalized_half_step": 0.05,
            "half_step_formula": "min(0.05,0.9*min(z,1-z))",
            "points_per_target": POINTS_PER_TARGET,
            "total_points": GLOBAL_CALL_CAP,
            "center_calls": 0,
            "one_sided_stencils": 0,
            "additional_step_sizes": 0,
            "derivative_denominator_policy": "persisted_float32_theta_plus_minus_theta_minus",
        },
        "response": {
            "ordering": "omega_components_0_9_then_M_components_10_19_force_major_gauge_0_then_pi",
            "physical_scales": {"omega_bar": 5.0, "M": 1.0},
        },
        "simulator": {
            "definition_identity": "graph-oscillator.overdamped-scientific.v1.p4-paper-baseline.m20-five-force-scientific.float32",
            "configuration_path": CONFIG.relative_to(ROOT).as_posix(),
            "configuration_sha256": CONFIG_SHA256,
            "device": "cpu", "dtype": "float32",
        },
        "call_contract": {
            "targets": TARGET_COUNT,
            "maximum_calls_per_target": POINTS_PER_TARGET,
            "required_successful_calls_per_target": POINTS_PER_TARGET,
            "maximum_simulator_calls_global": GLOBAL_CALL_CAP,
            "required_successful_calls_global": GLOBAL_CALL_CAP,
            "calls_per_point": 1, "center_calls": 0, "retry_calls": 0,
            "automatic_retry": False, "resubmission": False,
        },
        "execution_topology": {
            "jobs": 1, "array": False, "nodes": 1, "tasks": 1,
            "cpus_per_task": 20, "memory": "64G", "walltime": "00:30:00",
            "target_processing": "sequential_single_process",
        },
        "runtime_estimate": {
            "evidence_job": 6352257, "evidence_calls": 8,
            "evidence_elapsed_seconds": 52, "evidence_node": "barbun110",
            "evidence_cpus": 20, "evidence_memory": "64G",
            "lower_extrapolated_seconds": 480,
            "central_extrapolated_seconds": 624,
            "upper_extrapolated_seconds": 900,
            "qualification": "observed_elapsed_includes_worker_process_overhead_excludes_queue_delay",
        },
        "targets": targets,
        "scientific_calls_executed": 0,
        "jobs_submitted": 0,
        "execution_authorized": False,
        "submission_authorized": False,
    }
    return {**body, "declaration_identity": stable_hash(body["schema_version"], body)}


def load_declaration(path: Path = DECLARATION) -> dict[str, Any]:
    declaration = _read_json(path)
    identity = declaration.pop("declaration_identity", None)
    expected = stable_hash(declaration.get("schema_version", ""), declaration)
    declaration["declaration_identity"] = identity
    if identity != expected:
        raise StudyError("declaration identity mismatch")
    if declaration != build_declaration():
        raise StudyError("declaration does not exactly reproduce from frozen cohort")
    return declaration


def validate_preflight(output_root: Path) -> dict[str, Any]:
    declaration = load_declaration()
    if _sha256(CONFIG) != CONFIG_SHA256:
        raise StudyError("simulator configuration byte hash mismatch")
    if output_root.exists():
        raise StudyError("scientific output root must be fresh")
    return {
        "schema_version": "mutaco-oscillator-physics-multitarget-preflight-v1",
        "status": "validated_zero_call",
        "declaration_identity": declaration["declaration_identity"],
        "selected_target_indices": declaration["selection"]["selected_target_indices"],
        "call_preview": {"maximum_simulator_calls": 96, "scientific_calls_executed": 0, "center_calls": 0, "retry_calls": 0},
        "scientific_calls_executed": 0, "job_submitted": False,
        "submission_authorized": False,
    }


def _simulator(declaration: Mapping[str, Any]) -> OverdampedScientificSimulator:
    raw = _read_json(ROOT / declaration["simulator"]["configuration_path"])["simulator"]
    return OverdampedScientificSimulator(GraphSimulationConfig(
        model_identity=raw["identifier"], parameter_profile_id=raw["parameter_profile"],
        response_profile_id=raw["response_profile"], node_count=raw["node_count"],
        batch_size=raw["batch_size"], repeat_size=raw["repeat_size"],
        temperature=raw["temperature"], dt=raw["dt"], burn_in_steps=raw["burn_in_steps"],
        sample_steps=raw["sample_steps"], sample_stride=raw["sample_stride"],
        device="cpu", dtype=torch.float32,
    ))


def _context(target: Mapping[str, Any]) -> SimulationContext:
    crn = target["common_random_numbers"]
    return SimulationContext(crn["scientific_run_id"], crn["purpose"], crn["logical_index"],
                             crn["repeat_index"], crn["evaluation_seed"], crn["attempt_index"])


def _derived(target: Mapping[str, Any], outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(outcomes) != 8:
        raise StudyError("target sensitivity requires exactly eight outcomes")
    j_columns = []
    for a, coordinate in enumerate(target["coordinates"]):
        minus, plus = outcomes[2 * a]["response"], outcomes[2 * a + 1]["response"]
        denominator = coordinate["actual_physical_denominator"]
        j_columns.append([(p - m) / denominator for m, p in zip(minus, plus, strict=True)])
    J = torch.tensor(list(zip(*j_columns, strict=True)), dtype=torch.float64)
    ranges = torch.tensor([upper - lower for lower, upper in BOUNDS], dtype=torch.float64)
    D = J * ranges
    scales = torch.tensor([5.0] * 10 + [1.0] * 10, dtype=torch.float64)
    S = D / scales[:, None]
    _, singular, vh = torch.linalg.svd(S, full_matrices=False)
    norms = torch.linalg.vector_norm(S, dim=0)
    cosine = (S.T @ S) / (norms[:, None] * norms[None, :])
    contrasts = []
    for observable, offset, scale in (("omega_bar", 0, 5.0), ("M", 10, 1.0)):
        for force_index, force in enumerate(SCIENTIFIC_M20.forces):
            zero, pi = offset + 2 * force_index, offset + 2 * force_index + 1
            contrasts.append({"observable": observable, "force": float(force),
                              "J": (J[pi] - J[zero]).tolist(),
                              "D": (D[pi] - D[zero]).tolist(),
                              "S": ((D[pi] - D[zero]) / scale).tolist()})
    body = {
        "schema_version": "mutaco-oscillator-physics-target-sensitivity-v1",
        "target_index": target["target_index"],
        "parameter_names": list(PARAMETER_NAMES), "response_names": list(SCIENTIFIC_M20.names),
        "actual_physical_plus_minus_widths": [c["actual_physical_denominator"] for c in target["coordinates"]],
        "jacobian_physical_J": J.tolist(), "jacobian_normalized_D": D.tolist(),
        "dimensionless_S": S.tolist(), "scaled_column_norms": norms.tolist(),
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]),
        "weakest_right_singular_vector": vh[-1].tolist(),
        "scaled_column_cosine_similarities": cosine.tolist(),
        "gauge_contrast_sensitivities": contrasts,
        "point_outcome_identities": [outcome["outcome_identity"] for outcome in outcomes],
        "call_accounting": {"attempted": 8, "successful": 8, "failed": 0, "retry": 0, "center": 0},
    }
    return {**body, "sensitivity_identity": stable_hash(body["schema_version"], body)}


def execute_worker(output_root: Path, authorization: str) -> dict[str, Any]:
    if authorization != AUTHORIZATION_PHRASE or os.environ.get(AUTHORIZATION_ENV) != "1":
        raise StudyError("explicit 96-call authorization gate is closed")
    validate_preflight(output_root)
    declaration = load_declaration()
    output_root.mkdir(parents=True)
    simulator = _simulator(declaration)
    global_calls = 0
    target_results = []
    for target in declaration["targets"]:
        target_root = output_root / f"target_{target['target_index']:03d}"
        points_root = target_root / "points"
        points_root.mkdir(parents=True)
        context = _context(target)
        reference_manifest = None
        outcomes = []
        target_calls = 0
        for point in target["ordered_points"]:
            if target_calls >= 8 or global_calls >= 96:
                raise StudyError("scientific-call hard cap reached before declared design ended")
            intent = {"schema_version": "mutaco-oscillator-physics-multitarget-point-intent-v1",
                      "declaration_identity": declaration["declaration_identity"],
                      "target_index": target["target_index"], "target_point_index": point["target_point_index"],
                      "global_point_index": point["global_point_index"], "theta": point["theta"],
                      "evaluation_seed": context.evaluation_seed, "attempt_index": 0,
                      "maximum_transport_calls": 1, "retry_calls": 0}
            _write_new_json(points_root / f"point_{point['target_point_index']:02d}.intent.json", intent)
            target_calls += 1
            global_calls += 1
            result = simulator.evaluate(torch.tensor(point["theta"], dtype=torch.float32), context)
            metadata = dict(result.metadata)
            if metadata.get("transport_call_count") != 1:
                raise StudyError("simulator transport-call accounting mismatch")
            seed_manifest = metadata.get("seed_manifest")
            if reference_manifest is None:
                reference_manifest = seed_manifest
            elif seed_manifest != reference_manifest:
                raise StudyError("full-stream CRN manifest drifted within target")
            body = {"schema_version": "mutaco-oscillator-physics-multitarget-point-outcome-v1",
                    "target_index": target["target_index"], **point,
                    "response": result.values.detach().cpu().tolist(),
                    "evaluation_seed": context.evaluation_seed, "seed_manifest": seed_manifest,
                    "transport_call_count": 1}
            outcome = {**body, "outcome_identity": stable_hash(body["schema_version"], body)}
            _write_new_json(points_root / f"point_{point['target_point_index']:02d}.outcome.json", outcome)
            outcomes.append(outcome)
        derived = _derived(target, outcomes)
        _write_new_json(target_root / "sensitivity.json", derived)
        target_results.append(derived)
    if global_calls != 96 or len(target_results) != 12:
        raise StudyError("completed worker accounting is not exactly 96 calls across 12 targets")
    body = {"schema_version": "mutaco-oscillator-physics-multitarget-result-v1",
            "declaration_identity": declaration["declaration_identity"],
            "target_sensitivity_identities": [item["sensitivity_identity"] for item in target_results],
            "call_accounting": {"attempted": 96, "successful": 96, "failed": 0, "retry": 0, "center": 0}}
    manifest = {**body, "result_identity": stable_hash(body["schema_version"], body)}
    _write_new_json(output_root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("print-declaration")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output-root", type=Path, required=True)
    worker = commands.add_parser("execute-worker")
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--authorization", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "print-declaration":
        print(json.dumps(build_declaration(), indent=2, sort_keys=True, allow_nan=False))
    elif args.command == "preflight":
        print(json.dumps(validate_preflight(args.output_root), sort_keys=True))
    else:
        print(json.dumps(execute_worker(args.output_root, args.authorization), sort_keys=True))
    return 0


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.oscillator_physics_multi_target_sensitivity.study")


if __name__ == "__main__":
    raise SystemExit(main())
