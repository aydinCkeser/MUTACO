"""Validate job 6353452 and render its zero-call cohort sensitivity figure."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[5]
PACKAGE = Path(__file__).resolve().parent
WORKERS = PACKAGE / "workers"
LOGS = PACKAGE / "logs"
PROVENANCE = PACKAGE / "provenance"
DECLARATION = PROVENANCE / "declaration.json"
RECEIPT = PROVENANCE / "validation_receipt.json"
TARGET98 = (
    ROOT
    / "experiments/paper/oscillator_physics_post_analysis/outputs/"
    "target098_sensitivity_job6352257/derived_sensitivity.json"
)

JOB_ID = 6353452
SOURCE_COMMIT = "25ac98b25f78d6d79a5465df5fd8a3ce2d64b3aa"
SOURCE_TREE = "7ed7dd564b21c16a3420a45f5da04db2545e6132"
DECLARATION_IDENTITY = (
    "mutaco-oscillator-physics-multitarget-declaration-v1-"
    "22becfc33c2d62850044fec060cd3bec361c86f832a4a879adc94416f818f73c"
)
DECLARATION_SHA256 = "36bad7dcad86d2420797464299cdfe2872f6f83a28f3674ed0895a4b296ae636"
RECEIPT_IDENTITY = (
    "mutaco-truba-validation-receipt-v1-"
    "46b6c59ba95afd50c3313a7765ac8e5358351ae4f15a5c4afa1fc9e24efdc52e"
)
RESULT_IDENTITY = (
    "mutaco-oscillator-physics-multitarget-result-v1-"
    "1433783c0fce14526387a469002afe6f70d3c12cfe0ebf6102861b1d8df86a95"
)
TARGET_ORDER = [25, 64, 34, 48, 119, 86, 57, 13, 111, 96, 45, 58]
PARAMETER_NAMES = [
    "weight_mean",
    "pairwise_coupling_mean",
    "bias_scale_mean",
    "pairwise_coupling_CV",
]
PARAMETER_LABELS = [r"$\mu_W$", r"$\mu_K$", r"$\mu_A$", r"$\mathrm{CV}_K$"]
TARGET98_KAPPA = 85.7155562376652
FIGURE_STEM = "paper120_maximin12_oscillator_sensitivity_cohort"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def stable_hash(prefix: str, value: Any) -> str:
    return f"{prefix}-{hashlib.sha256(canonical(value)).hexdigest()}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_identity(document: dict[str, Any], field: str) -> None:
    identity = document[field]
    body = {key: value for key, value in document.items() if key != field}
    if identity != stable_hash(document["schema_version"], body):
        raise RuntimeError(f"Identity mismatch: {field}")


def close(actual: Any, expected: Any) -> bool:
    return torch.allclose(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def recompute(target: dict[str, Any], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    columns = []
    for coordinate in range(4):
        minus = outcomes[2 * coordinate]["response"]
        plus = outcomes[2 * coordinate + 1]["response"]
        width = target["coordinates"][coordinate]["actual_physical_denominator"]
        columns.append([(p - m) / width for m, p in zip(minus, plus, strict=True)])
    J = torch.tensor(list(zip(*columns, strict=True)), dtype=torch.float64)
    D = J * torch.tensor([1.5, 1.5, 1.5, 1.0], dtype=torch.float64)
    S = D / torch.tensor([5.0] * 10 + [1.0] * 10, dtype=torch.float64)[:, None]
    _, singular, vh = torch.linalg.svd(S, full_matrices=False)
    norms = torch.linalg.vector_norm(S, dim=0)
    cosines = (S.T @ S) / (norms[:, None] * norms[None, :])
    contrasts = []
    for observable, offset, scale in (("omega_bar", 0, 5.0), ("M", 10, 1.0)):
        for force_index, force in enumerate((0.0, 1.25, 2.5, 3.75, 5.0)):
            zero = offset + 2 * force_index
            pi = zero + 1
            contrasts.append(
                {
                    "observable": observable,
                    "force": force,
                    "J": (J[pi] - J[zero]).tolist(),
                    "D": (D[pi] - D[zero]).tolist(),
                    "S": ((D[pi] - D[zero]) / scale).tolist(),
                }
            )
    return {
        "J": J.tolist(),
        "D": D.tolist(),
        "S": S.tolist(),
        "norms": norms.tolist(),
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]),
        "weakest_vector": vh[-1].tolist(),
        "cosines": cosines.tolist(),
        "gauge_contrasts": contrasts,
    }


def validate_and_collect() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256(DECLARATION) != DECLARATION_SHA256:
        raise RuntimeError("Declaration byte hash mismatch")
    declaration = load_json(DECLARATION)
    verify_identity(declaration, "declaration_identity")
    if declaration["declaration_identity"] != DECLARATION_IDENTITY:
        raise RuntimeError("Declaration identity mismatch")
    if declaration["selection"]["selected_target_indices"] != TARGET_ORDER:
        raise RuntimeError("Frozen target order mismatch")
    if declaration["selection"]["target098_selected"] is not False:
        raise RuntimeError("Target 98 must not be a cohort member")

    receipt = load_json(RECEIPT)
    verify_identity(receipt, "receipt_identity")
    if receipt["receipt_identity"] != RECEIPT_IDENTITY:
        raise RuntimeError("Validation receipt identity mismatch")
    if receipt["source_commit"] != SOURCE_COMMIT or receipt["status"] != "validated_zero_call":
        raise RuntimeError("Validation receipt source/status mismatch")

    manifest = load_json(WORKERS / "manifest.json")
    verify_identity(manifest, "result_identity")
    if manifest["result_identity"] != RESULT_IDENTITY:
        raise RuntimeError("Worker result identity mismatch")
    expected_accounting = {"attempted": 96, "successful": 96, "failed": 0, "retry": 0, "center": 0}
    if manifest["call_accounting"] != expected_accounting:
        raise RuntimeError("Global call accounting mismatch")

    expected_worker_files = {"manifest.json"}
    outcome_ids: set[str] = set()
    global_indices: list[int] = []
    target_seeds: list[int] = []
    target_run_ids: list[str] = []
    target_manifest_hashes: list[str] = []
    sensitivity_ids: list[str] = []
    rows = []
    total_reduced = 0

    for target in declaration["targets"]:
        index = target["target_index"]
        target_root = WORKERS / f"target_{index:03d}"
        outcomes = []
        seed_manifests = []
        target_seed = target["common_random_numbers"]["evaluation_seed"]
        target_seeds.append(target_seed)
        target_run_ids.append(target["common_random_numbers"]["scientific_run_id"])
        reduced = [
            coordinate["parameter_name"]
            for coordinate in target["coordinates"]
            if coordinate["boundary_reduced_from_0p05"]
        ]
        total_reduced += len(reduced)
        for coordinate in target["coordinates"]:
            minus = coordinate["persisted_float32_minus_theta"][coordinate["parameter_index"]]
            plus = coordinate["persisted_float32_plus_theta"][coordinate["parameter_index"]]
            if plus - minus != coordinate["actual_physical_denominator"]:
                raise RuntimeError(f"Persisted denominator mismatch for target {index}")
        for point in target["ordered_points"]:
            point_index = point["target_point_index"]
            intent_rel = f"target_{index:03d}/points/point_{point_index:02d}.intent.json"
            outcome_rel = f"target_{index:03d}/points/point_{point_index:02d}.outcome.json"
            expected_worker_files.update({intent_rel, outcome_rel})
            intent = load_json(WORKERS / intent_rel)
            outcome = load_json(WORKERS / outcome_rel)
            if (
                intent["target_index"] != index
                or intent["target_point_index"] != point_index
                or intent["global_point_index"] != point["global_point_index"]
                or intent["theta"] != point["theta"]
                or intent["evaluation_seed"] != target_seed
                or intent["attempt_index"] != 0
                or intent["maximum_transport_calls"] != 1
                or intent["retry_calls"] != 0
            ):
                raise RuntimeError(f"Intent mismatch at target {index}, point {point_index}")
            verify_identity(outcome, "outcome_identity")
            if (
                outcome["target_index"] != index
                or outcome["target_point_index"] != point_index
                or outcome["global_point_index"] != point["global_point_index"]
                or outcome["theta"] != point["theta"]
                or outcome["evaluation_seed"] != target_seed
                or outcome["transport_call_count"] != 1
                or len(outcome["response"]) != 20
                or not all(math.isfinite(value) for value in outcome["response"])
            ):
                raise RuntimeError(f"Outcome mismatch at target {index}, point {point_index}")
            if outcome["outcome_identity"] in outcome_ids:
                raise RuntimeError("Duplicate outcome identity")
            outcome_ids.add(outcome["outcome_identity"])
            global_indices.append(outcome["global_point_index"])
            seed_manifests.append(outcome["seed_manifest"])
            outcomes.append(outcome)
        if not all(value == seed_manifests[0] for value in seed_manifests):
            raise RuntimeError(f"Within-target CRN drift for target {index}")
        target_manifest_hashes.append(hashlib.sha256(canonical(seed_manifests[0])).hexdigest())

        sensitivity_rel = f"target_{index:03d}/sensitivity.json"
        expected_worker_files.add(sensitivity_rel)
        stored = load_json(WORKERS / sensitivity_rel)
        verify_identity(stored, "sensitivity_identity")
        calculated = recompute(target, outcomes)
        comparisons = {
            "jacobian_physical_J": "J",
            "jacobian_normalized_D": "D",
            "dimensionless_S": "S",
            "scaled_column_norms": "norms",
            "singular_values": "singular_values",
            "scaled_column_cosine_similarities": "cosines",
        }
        for stored_key, calculated_key in comparisons.items():
            if not close(stored[stored_key], calculated[calculated_key]):
                raise RuntimeError(f"Recomputed {stored_key} mismatch for target {index}")
        if not math.isclose(stored["condition_number"], calculated["condition_number"], abs_tol=1e-10, rel_tol=1e-10):
            raise RuntimeError(f"Recomputed condition number mismatch for target {index}")
        weak_stored = torch.tensor(stored["weakest_right_singular_vector"], dtype=torch.float64)
        weak_calculated = torch.tensor(calculated["weakest_vector"], dtype=torch.float64)
        if min(
            torch.max(torch.abs(weak_stored - weak_calculated)),
            torch.max(torch.abs(weak_stored + weak_calculated)),
        ) >= 1e-10:
            raise RuntimeError(f"Recomputed weakest vector mismatch for target {index}")
        for observed, expected in zip(stored["gauge_contrast_sensitivities"], calculated["gauge_contrasts"], strict=True):
            if observed["observable"] != expected["observable"] or observed["force"] != expected["force"]:
                raise RuntimeError(f"Gauge contrast label mismatch for target {index}")
            if not all(close(observed[key], expected[key]) for key in ("J", "D", "S")):
                raise RuntimeError(f"Gauge contrast recomputation mismatch for target {index}")
        if stored["call_accounting"] != {"attempted": 8, "successful": 8, "failed": 0, "retry": 0, "center": 0}:
            raise RuntimeError(f"Per-target call accounting mismatch for target {index}")
        sensitivity_ids.append(stored["sensitivity_identity"])
        rows.append(
            {
                "target_index": index,
                "theta_star": target["theta_star"],
                "normalized_theta_star": target["normalized_theta_star"],
                "evaluation_seed": target_seed,
                "seed_manifest_sha256": target_manifest_hashes[-1],
                "boundary_reduced_coordinates": reduced,
                "actual_physical_denominators": [item["actual_physical_denominator"] for item in target["coordinates"]],
                "scaled_column_norms": stored["scaled_column_norms"],
                "singular_values": stored["singular_values"],
                "condition_number": stored["condition_number"],
                "weakest_right_singular_vector": stored["weakest_right_singular_vector"],
                "absolute_weakest_right_singular_vector": [abs(value) for value in stored["weakest_right_singular_vector"]],
                "scaled_column_cosine_similarities": stored["scaled_column_cosine_similarities"],
                "gauge_contrast_sensitivities": stored["gauge_contrast_sensitivities"],
                "sensitivity_identity": stored["sensitivity_identity"],
            }
        )

    actual_worker_files = {path.relative_to(WORKERS).as_posix() for path in WORKERS.rglob("*") if path.is_file()}
    if actual_worker_files != expected_worker_files:
        raise RuntimeError("Missing or extra worker artifacts")
    if [row["target_index"] for row in rows] != TARGET_ORDER:
        raise RuntimeError("Result target ordering mismatch")
    if global_indices != list(range(96)) or len(outcome_ids) != 96:
        raise RuntimeError("Global point accounting is not exactly 96 unique outcomes")
    if len(set(target_seeds)) != 12 or len(set(target_run_ids)) != 12 or len(set(target_manifest_hashes)) != 12:
        raise RuntimeError("Target CRN contexts are not mutually distinct")
    if 9818052026 in target_seeds:
        raise RuntimeError("Target-98 diagnostic seed was reused")
    if total_reduced != 13:
        raise RuntimeError("Boundary-reduced coordinate count mismatch")
    if manifest["target_sensitivity_identities"] != sensitivity_ids:
        raise RuntimeError("Global manifest sensitivity ordering mismatch")

    target98 = load_json(TARGET98)
    if target98["source_commit"] != "aeee1cc567abff6b0f23e1e6dbdce4115e882c7c":
        raise RuntimeError("Target-98 reference source mismatch")
    if target98["scheduler_state"] != "COMPLETED" or target98["scheduler_exit_code"] != "0:0":
        raise RuntimeError("Target-98 reference execution mismatch")
    if not math.isclose(target98["scaled_condition_number"], TARGET98_KAPPA, abs_tol=1e-12, rel_tol=0.0):
        raise RuntimeError("Target-98 reference condition mismatch")

    return declaration, {"manifest": manifest, "rows": rows}, target98


def distribution(values: list[list[float]]) -> dict[str, dict[str, float]]:
    columns = list(zip(*values, strict=True))
    return {
        PARAMETER_NAMES[index]: {
            "minimum": min(column),
            "median": statistics.median(column),
            "mean": statistics.fmean(column),
            "maximum": max(column),
        }
        for index, column in enumerate(columns)
    }


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def render(plotting: dict[str, Any], pdf: Path, png: Path) -> None:
    configure_plotting()
    norms = np.asarray(plotting["scaled_column_norms"], dtype=float)
    weakest = np.asarray(plotting["absolute_weakest_right_singular_vectors"], dtype=float)
    conditions = np.asarray(plotting["condition_numbers"], dtype=float)
    targets = plotting["target_order"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

    fig = plt.figure(figsize=(7.15, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.25, 1.12))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    positions = np.arange(4)
    box = ax_a.boxplot(
        [norms[:, column] for column in range(4)],
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"color": "#666666", "linewidth": 0.8},
        capprops={"color": "#666666", "linewidth": 0.8},
        boxprops={"edgecolor": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
    offsets = np.linspace(-0.12, 0.12, len(targets))
    for rank, offset in enumerate(offsets):
        ax_a.plot(positions + offset, norms[rank], color="#777777", alpha=0.22, linewidth=0.55, zorder=1)
        ax_a.scatter(positions + offset, norms[rank], s=13, color=colors, edgecolor="white", linewidth=0.25, zorder=3)
    ax_a.set_yscale("log")
    ax_a.set_xticks(positions, PARAMETER_LABELS)
    ax_a.set_ylabel(r"Local sensitivity norm $\|S_{\cdot a}\|_2$")
    ax_a.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.45, alpha=0.8)
    ax_a.spines[["top", "right"]].set_visible(False)

    image = ax_b.imshow(weakest, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax_b.set_xticks(np.arange(4), PARAMETER_LABELS)
    ax_b.set_yticks(np.arange(12), [str(target) for target in targets])
    ax_b.set_ylabel("Target (maximin order)")
    colorbar = fig.colorbar(image, ax=ax_b, fraction=0.055, pad=0.035)
    colorbar.set_label(r"$|v_{\min,a}|$")
    colorbar.set_ticks([0.0, 0.5, 1.0])

    ranks = np.arange(1, 13)
    ax_c.plot(ranks, conditions, color="#0072B2", linewidth=0.8, alpha=0.55, zorder=1)
    ax_c.scatter(ranks, conditions, s=24, color="#0072B2", edgecolor="white", linewidth=0.4, zorder=2, label="Maximin-12 targets")
    ax_c.axhline(TARGET98_KAPPA, color="#D55E00", linestyle=(0, (4, 2)), linewidth=1.0, label=r"Target 98 ($\kappa=85.7$; not selected)")
    ax_c.set_yscale("log")
    ax_c.set_xticks(ranks, [str(target) for target in targets], rotation=55, ha="right")
    ax_c.set_xlabel("Target (maximin order)")
    ax_c.set_ylabel(r"Condition number $\kappa(S)$")
    ax_c.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.45, alpha=0.8)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.legend(frameon=False, loc="upper left", handlelength=2.4, borderaxespad=0.25)

    for label, axis in zip(("(a)", "(b)", "(c)"), (ax_a, ax_b, ax_c), strict=True):
        axis.text(-0.17, 1.04, label, transform=axis.transAxes, fontsize=9.5, fontweight="bold", va="bottom")

    metadata = {
        "Title": "Paper-120 maximin-12 oscillator local sensitivity",
        "Author": "MUTACO",
        "Subject": "Zero-call post-analysis of TRUBA job 6353452",
        "Keywords": "oscillator sensitivity Jacobian singular values",
        "Creator": "MUTACO deterministic plotting",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(png, dpi=300, bbox_inches="tight", metadata={"Software": "MUTACO deterministic plotting"})
    plt.close(fig)


def main() -> None:
    final_names = [
        "cohort_summary.json",
        "plotting_data.json",
        "validation.json",
        f"{FIGURE_STEM}.pdf",
        f"{FIGURE_STEM}.png",
        "provenance/file_hashes.json",
        "provenance_manifest.json",
    ]
    for name in final_names:
        if (PACKAGE / name).exists():
            raise RuntimeError(f"Refusing to overwrite append-only artifact: {name}")

    declaration, collected, target98 = validate_and_collect()
    rows = collected["rows"]
    norms = [row["scaled_column_norms"] for row in rows]
    absolute_weakest = [row["absolute_weakest_right_singular_vector"] for row in rows]
    conditions = [row["condition_number"] for row in rows]
    target98_S = torch.tensor(target98["dimensionless_sensitivities_S"], dtype=torch.float64)
    _, target98_singular, target98_vh = torch.linalg.svd(target98_S, full_matrices=False)

    plotting_body = {
        "schema_version": "mutaco-oscillator-physics-multitarget-plotting-data-v1",
        "job_id": JOB_ID,
        "source_commit": SOURCE_COMMIT,
        "declaration_identity": DECLARATION_IDENTITY,
        "target_order": TARGET_ORDER,
        "target98_is_cohort_member": False,
        "parameter_names": PARAMETER_NAMES,
        "parameter_math_labels": PARAMETER_LABELS,
        "theta_star": [row["theta_star"] for row in rows],
        "scaled_column_norms": norms,
        "absolute_weakest_right_singular_vectors": absolute_weakest,
        "condition_numbers": conditions,
        "target98_reference": {
            "target_index": 98,
            "cohort_member": False,
            "job_id": 6352257,
            "condition_number": TARGET98_KAPPA,
            "scaled_column_norms": target98["scaled_column_norms"],
            "singular_values": target98_singular.tolist(),
            "absolute_weakest_right_singular_vector": torch.abs(target98_vh[-1]).tolist(),
        },
    }
    plotting = {**plotting_body, "plotting_data_identity": stable_hash(plotting_body["schema_version"], plotting_body)}

    norm_dominance = [PARAMETER_NAMES[max(range(4), key=lambda a: row[a])] for row in norms]
    norm_weakest = [PARAMETER_NAMES[min(range(4), key=lambda a: row[a])] for row in norms]
    singular_dominance = [PARAMETER_NAMES[max(range(4), key=lambda a: row[a])] for row in absolute_weakest]
    summary_body = {
        "schema_version": "mutaco-oscillator-physics-multitarget-cohort-summary-v1",
        "job_id": JOB_ID,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "declaration_identity": DECLARATION_IDENTITY,
        "result_identity": RESULT_IDENTITY,
        "target_order": TARGET_ORDER,
        "targets": 12,
        "successful_calls": 96,
        "failed_calls": 0,
        "retry_calls": 0,
        "center_calls": 0,
        "boundary_reduced_coordinate_pairs": 13,
        "parameter_names": PARAMETER_NAMES,
        "column_norm_distribution": distribution(norms),
        "column_norm_dominance_counts": {name: norm_dominance.count(name) for name in PARAMETER_NAMES},
        "column_norm_weakest_counts": {name: norm_weakest.count(name) for name in PARAMETER_NAMES},
        "weakest_singular_direction_dominance_counts": {name: singular_dominance.count(name) for name in PARAMETER_NAMES},
        "condition_number_distribution": {
            "minimum": min(conditions),
            "median": statistics.median(conditions),
            "mean": statistics.fmean(conditions),
            "maximum": max(conditions),
        },
        "per_target": rows,
        "target98_reference": plotting_body["target98_reference"],
    }
    summary = {**summary_body, "cohort_summary_identity": stable_hash(summary_body["schema_version"], summary_body)}

    validation_body = {
        "schema_version": "mutaco-oscillator-physics-multitarget-validation-v1",
        "status": "passed_zero_call_postprocessing",
        "job_id": JOB_ID,
        "scheduler": {"partition": "barbun", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:02:07", "node": "barbun10", "allocated_cpus": 20, "requested_memory": "64G"},
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "declaration_identity": DECLARATION_IDENTITY,
        "validation_receipt_identity": RECEIPT_IDENTITY,
        "result_identity": RESULT_IDENTITY,
        "targets": 12,
        "successful_outcomes_per_target": 8,
        "scientific_calls": 96,
        "failed_calls": 0,
        "retry_calls": 0,
        "center_calls": 0,
        "extra_or_duplicate_calls": 0,
        "response_length": 20,
        "all_responses_finite": True,
        "persisted_float32_denominators_used": True,
        "within_target_crn_manifests_identical": True,
        "between_target_crn_manifests_distinct": True,
        "unique_target_crn_contexts": 12,
        "boundary_reduced_coordinate_pairs": 13,
        "recomputed_J_D_S_and_svd_products_match": True,
        "simulator_calls_added_by_postprocessing": 0,
        "sbatch_invocations_added_by_postprocessing": 0,
    }
    validation = {**validation_body, "validation_identity": stable_hash(validation_body["schema_version"], validation_body)}

    with tempfile.TemporaryDirectory(prefix="build-tmp-", dir=PACKAGE) as temporary:
        temp = Path(temporary)
        temp_provenance = temp / "provenance"
        temp_provenance.mkdir()
        write_json(temp / "plotting_data.json", plotting)
        write_json(temp / "cohort_summary.json", summary)
        write_json(temp / "validation.json", validation)
        render(plotting, temp / f"{FIGURE_STEM}.pdf", temp / f"{FIGURE_STEM}.png")

        hash_entries = []
        for path in sorted((WORKERS).rglob("*")) + sorted((LOGS).rglob("*")):
            if path.is_file():
                hash_entries.append({"path": path.relative_to(PACKAGE).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
        for path in (DECLARATION, RECEIPT, Path(__file__)):
            hash_entries.append({"path": path.relative_to(PACKAGE).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
        for name in ("plotting_data.json", "cohort_summary.json", "validation.json", f"{FIGURE_STEM}.pdf", f"{FIGURE_STEM}.png"):
            path = temp / name
            hash_entries.append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size})
        hashes_body = {"schema_version": "mutaco-file-hash-inventory-v1", "files": sorted(hash_entries, key=lambda row: row["path"])}
        hashes = {**hashes_body, "inventory_identity": stable_hash(hashes_body["schema_version"], hashes_body)}
        write_json(temp_provenance / "file_hashes.json", hashes)

        provenance_body = {
            "schema_version": "mutaco-oscillator-physics-multitarget-result-package-v1",
            "status": "complete_validated_zero_call_postprocessing",
            "job": {"job_id": JOB_ID, "partition": "barbun", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:02:07", "node": "barbun10", "allocated_cpus": 20, "requested_memory": "64G"},
            "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE, "remote_exact_source": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/oscillator_physics_multitarget/25ac98b25f78/source-exact"},
            "remote_evidence": {"workers": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/oscillator_physics_multitarget/25ac98b25f78/workers", "logs": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/oscillator_physics_multitarget/25ac98b25f78/logs", "copy_count": 1},
            "declaration": {"identity": DECLARATION_IDENTITY, "sha256": DECLARATION_SHA256},
            "validation_receipt_identity": RECEIPT_IDENTITY,
            "worker_result_identity": RESULT_IDENTITY,
            "cohort_summary_identity": summary["cohort_summary_identity"],
            "plotting_data_identity": plotting["plotting_data_identity"],
            "validation_identity": validation["validation_identity"],
            "file_inventory_identity": hashes["inventory_identity"],
            "file_inventory_sha256": sha256(temp_provenance / "file_hashes.json"),
            "figure": {
                "pdf": f"{FIGURE_STEM}.pdf",
                "pdf_sha256": sha256(temp / f"{FIGURE_STEM}.pdf"),
                "png": f"{FIGURE_STEM}.png",
                "png_sha256": sha256(temp / f"{FIGURE_STEM}.png"),
            },
            "call_accounting": {"attempted": 96, "successful": 96, "failed": 0, "retry": 0, "center": 0, "postprocessing_scientific_calls": 0},
            "target_order": TARGET_ORDER,
            "target98_is_cohort_member": False,
        }
        provenance = {**provenance_body, "package_identity": stable_hash(provenance_body["schema_version"], provenance_body)}
        write_json(temp / "provenance_manifest.json", provenance)

        for name in final_names:
            source = temp / name
            destination = PACKAGE / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

    print(json.dumps({"package_identity": provenance["package_identity"], "plotting_data_identity": plotting["plotting_data_identity"], "cohort_summary_identity": summary["cohort_summary_identity"], "validation_identity": validation["validation_identity"], "pdf_sha256": provenance["figure"]["pdf_sha256"], "png_sha256": provenance["figure"]["png_sha256"], "targets": 12, "scientific_calls": 96, "postprocessing_scientific_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
