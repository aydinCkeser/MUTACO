"""Plot joint Paper-120 parameter recovery and response fidelity.

The historical MUTACO/NPE/SNPE and prospective NLE/NRE records are joined by
frozen target, budget, recommendation, and held-out result identities.  This
review plot performs no simulation, fitting, or recommendation generation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TARGET_COUNT = 120
PARAMETER_COUNT = 4
BUDGETS = (40, 100)
METHODS = ("mutaco", "npe", "snpe", "nle", "nre")
METHOD_STYLES = {
    "mutaco": ("MUTACO", "#0072B2"),
    "npe": ("NPE", "#CC6677"),
    "snpe": ("SNPE", "#332288"),
    "nle": ("NLE", "#999933"),
    "nre": ("NRE", "#444444"),
}
COHORT_SHA256 = (
    "359fc0bb59fc9a8a52f5da684e7a7eb352466bdc0d0a54fb8efa94131ce423bb"
)
PREPARATION_MANIFEST_SHA256 = (
    "29b4fece734e081d14820adba50bc80d5a08877cbccb4eece94d6a0ef7f2c11d"
)
TASKS_SHA256 = (
    "d24a14ed7ff0e95feda3b66e3938dd648801fef5576e0af6d444e2419adede09"
)
PREPARATION_PACKAGE_IDENTITY = (
    "repaired-sbi-assessment-preparation-package-v1-"
    "15191325a32642bc9da1f6e4514cc7f84208263f62e8bc642832a18722f8eb46"
)
TASKS_IDENTITY = (
    "repaired-sbi-assessment-task-list-v1-"
    "a9ae8aa77ea067d40f83d68d7079104893cba2e72beddb9ddad6e4a23a6f17c3"
)
REPAIRED_MANIFEST_SHA256 = (
    "46a0ed6b9667e13c77cb6d6a1ba12d65f4de5f4ebb11ccc23f4cc8558fee0c1c"
)
REPAIRED_ROWS_SHA256 = {
    40: "2a6c1d6835c9d818b1d942c89cdc39e56701baafa30a20d2955afb9641ef0e67",
    100: "8f7ebe1945db35b7d92f57c78eb95090599e71a94df99f8b49b6ee4db9bbbc5a",
}
REPAIRED_EVIDENCE_IDENTITY = (
    "repaired-sbi-assessment-evidence-v1-"
    "5c8c27a1c3669d61161a8601edda7cb82e895c6849f813ec8536af796c43bc0f"
)
REPAIRED_COLLECTION_IDENTITY = (
    "repaired-sbi-assessment-collection-v1-"
    "507d012c72151554bea319511e8a04d699382407abe6c1ee97bb8c4c15c19ec6"
)
ZOO_MANIFEST_SHA256 = (
    "251f26eec73b6dfe58f69678bd3b92ccc58bb909ae3d4d6958e62aa2ba6a6661"
)
ZOO_ROWS_SHA256 = (
    "400c82417f9b6c5c31509c9c433760ea0b9ee9fa2f0e9c80dfd9fd1cfabfcee0"
)
ZOO_PACKAGE_IDENTITY = (
    "paper120-sbi-zoo-analysis-package-v1-"
    "0568b1130e1f0913bdc252bb784dedf12c1d2ea77dc35bcf0b541e37f89b5abd"
)
P3_MANIFEST_SHA256 = (
    "28cf7be24796a48baae6636e473f013449eba4bc724c2fb129a5d6a00662c23e"
)
P3_COLLECTION_IDENTITY = (
    "mutaco-paper120-sbi-zoo-p3-collection-v1-"
    "d35fd77ef3a3e03e00f784f682315dd7314288ffa13cd48d418846bfdfe36589"
)
EXPECTED_SUMMARIES = {
    40: {
        "mutaco": (0.3068058104737574, 0.26643440247402206, 0.02228858642026008, 0.01657466237263918),
        "npe": (0.5099709617153423, 0.4975237552964847, 0.13151996656526227, 0.09360039633123735),
        "snpe": (0.4959962387432404, 0.45846422513926244, 0.13883172541877123, 0.10137297601028195),
        "nle": (0.38386332429983944, 0.36667484254297494, 0.09610796888559962, 0.07403186276405904),
        "nre": (0.5137367247083671, 0.49403358316074536, 0.2353600020674648, 0.19429141115469975),
    },
    100: {
        "mutaco": (0.2877885251688941, 0.21541392602945214, 0.02273630326705556, 0.017818431254147525),
        "npe": (0.3699191527303205, 0.3583801062931798, 0.06838273574858417, 0.049602398708702825),
        "snpe": (0.42839725431627873, 0.41539938633820034, 0.07379806015385801, 0.04837388540363463),
        "nle": (0.32040239559642253, 0.2727086545732353, 0.06752278483667987, 0.04797336951082798),
        "nre": (0.4584093277349333, 0.39535550933607644, 0.1618424007525422, 0.11102903374817288),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(f"Frozen input hash mismatch for {path}: {observed}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _stable_hash(prefix: str, value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(canonical).hexdigest()}"


def _vector(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != PARAMETER_COUNT:
        raise RuntimeError(f"{label} must have four physical-coordinate values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise RuntimeError(f"{label} contains a non-finite component")
    return result


def _parameter_error(recommended: tuple[float, ...], truth: tuple[float, ...]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(recommended, truth, strict=True)))


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and strictly positive")
    return number


def _load_targets(repo: Path) -> tuple[list[str], dict[int, tuple[float, ...]]]:
    root = repo / "experiments/graph_inversion/frozen_targets/paper120_v3"
    manifest_path = root / "manifest.json"
    _require_hash(manifest_path, COHORT_SHA256)
    manifest = _load_json(manifest_path)
    if (
        manifest.get("completed_count") != TARGET_COUNT
        or manifest.get("missing_count") != 0
        or manifest.get("selected_indices") != list(range(TARGET_COUNT))
    ):
        raise RuntimeError("Frozen Paper-120 cohort scope or ordering mismatch")
    target_ids: list[str] = []
    truth: dict[int, tuple[float, ...]] = {}
    for position in range(TARGET_COUNT):
        path = root / f"target_{position:03d}.json"
        expected_file_hash = path.with_suffix(".sha256").read_text(encoding="ascii").strip()
        _require_hash(path, expected_file_hash)
        record = _load_json(path)
        content = {
            key: value for key, value in record.items()
            if key not in {"content_hash", "target_record_id"}
        }
        content_hash = _stable_hash("frozen-target-content-v2", content)
        record_id = _stable_hash(
            "frozen-target-record-v2", {**content, "content_hash": content_hash}
        )
        if (
            record.get("target_index") != position
            or record.get("content_hash") != content_hash
            or record.get("target_record_id") != record_id
            or manifest["records"][str(position)] != {
                "content_hash": content_hash,
                "target_record_id": record_id,
            }
        ):
            raise RuntimeError(f"Frozen target identity mismatch at position {position}")
        target_ids.append(record_id)
        truth[position] = _vector(record.get("theta_star"), f"target {position} theta_star")
    return target_ids, truth


def _load_historical(
    repo: Path,
    target_ids: list[str],
    truth: dict[int, tuple[float, ...]],
) -> tuple[
    dict[int, dict[str, list[tuple[float, float]]]],
    dict[int, list[dict[str, Any]]],
    dict[int, dict[int, dict[str, Any]]],
]:
    prep_root = repo / "results/paper/repaired_mutaco_sbi_assessment_preparation/v1"
    prep_manifest_path = prep_root / "manifest.json"
    tasks_path = prep_root / "assessment_tasks.jsonl"
    repaired_root = repo / "results/paper/repaired_mutaco_sbi_assessment_6251303/v1"
    repaired_manifest_path = repaired_root / "manifest.json"
    _require_hash(prep_manifest_path, PREPARATION_MANIFEST_SHA256)
    _require_hash(tasks_path, TASKS_SHA256)
    _require_hash(repaired_manifest_path, REPAIRED_MANIFEST_SHA256)
    prep_manifest = _load_json(prep_manifest_path)
    repaired_manifest = _load_json(repaired_manifest_path)
    if (
        prep_manifest.get("package_identity") != PREPARATION_PACKAGE_IDENTITY
        or prep_manifest.get("tasks_identity") != TASKS_IDENTITY
        or prep_manifest.get("task_count") != 720
        or repaired_manifest.get("evidence_identity") != REPAIRED_EVIDENCE_IDENTITY
        or repaired_manifest.get("collection_identity") != REPAIRED_COLLECTION_IDENTITY
        or repaired_manifest.get("cohort_sha256") != COHORT_SHA256
    ):
        raise RuntimeError("Historical repaired SBI package identity mismatch")

    tasks = _load_jsonl(tasks_path)
    if len(tasks) != 720:
        raise RuntimeError("Expected exactly 720 historical assessment tasks")
    tasks_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for task in tasks:
        budget = task.get("budget")
        position = task.get("target_position")
        method = task.get("method")
        key = (budget, position, method)
        if (
            budget not in BUDGETS
            or not isinstance(position, int)
            or not 0 <= position < TARGET_COUNT
            or method not in {"mutaco", "sbi_npe", "sbi_snpe"}
            or task.get("target_record_id") != target_ids[position]
            or task.get("automatic_retry") is not False
            or key in tasks_by_key
        ):
            raise RuntimeError("Invalid or duplicate historical assessment task")
        if method == "mutaco":
            if task.get("recommendation_provenance") != "corrected_refined_terminal":
                raise RuntimeError("MUTACO task does not use corrected repaired recommendation")
        elif task.get("recommendation_provenance") != "immutable_native_sbi_recommendation":
            raise RuntimeError("NPE/SNPE task does not use immutable native recommendation")
        _vector(task.get("recommended_theta"), f"historical {key} recommendation")
        tasks_by_key[key] = task

    plot_mapping = {"mutaco": "mutaco", "sbi_npe": "npe", "sbi_snpe": "snpe"}
    points = {budget: {method: [] for method in plot_mapping.values()} for budget in BUDGETS}
    contexts: dict[int, list[dict[str, Any]]] = {}
    mutaco_rows: dict[int, dict[int, dict[str, Any]]] = {}
    for budget in BUDGETS:
        rows_path = repaired_root / f"analysis_rows_b{budget}.jsonl"
        _require_hash(rows_path, REPAIRED_ROWS_SHA256[budget])
        rows = _load_jsonl(rows_path)
        if len(rows) != 360:
            raise RuntimeError(f"Expected exactly 360 repaired B{budget} rows")
        rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
        for row in rows:
            position = row.get("target_position")
            method = row.get("method")
            key = (position, method)
            if (
                not isinstance(position, int)
                or not 0 <= position < TARGET_COUNT
                or method not in plot_mapping
                or row.get("target_id") != target_ids[position]
                or row.get("budget") != budget
                or row.get("evidence_statuses") != ["valid_current_policy_evidence"]
                or key in rows_by_key
            ):
                raise RuntimeError(f"Invalid or duplicate repaired B{budget} row")
            rows_by_key[key] = row

        contexts[budget] = []
        mutaco_rows[budget] = {}
        for position in range(TARGET_COUNT):
            trio = [rows_by_key[(position, source)] for source in plot_mapping]
            row_contexts = [row["assessment"]["context"] for row in trio]
            if not row_contexts[0] == row_contexts[1] == row_contexts[2]:
                raise RuntimeError(f"Historical B{budget} target {position} is not paired")
            contexts[budget].append(row_contexts[0])
            mutaco_rows[budget][position] = rows_by_key[(position, "mutaco")]
            for source_method, plot_method in plot_mapping.items():
                row = rows_by_key[(position, source_method)]
                task = tasks_by_key[(budget, position, source_method)]
                if (
                    task["recommendation_identity"]
                    != row["diagnostics"]["recommendation_identity"]
                    or task["assessment_context"] != row["assessment"]["context"]
                ):
                    raise RuntimeError(
                        f"Historical B{budget} {plot_method} recommendation/result mismatch"
                    )
                recommended = _vector(
                    task["recommended_theta"],
                    f"B{budget} {plot_method} target {position} recommendation",
                )
                parameter_error = _parameter_error(recommended, truth[position])
                response_error = _positive(
                    row["errors"]["heldout_assessment_error"]["metric"][
                        "component_mean_mae"
                    ],
                    f"B{budget} {plot_method} target {position} response MAE",
                )
                points[budget][plot_method].append((parameter_error, response_error))
    return points, contexts, mutaco_rows


def _load_prospective(
    repo: Path,
    target_ids: list[str],
    truth: dict[int, tuple[float, ...]],
    contexts: dict[int, list[dict[str, Any]]],
    mutaco_rows: dict[int, dict[int, dict[str, Any]]],
    points: dict[int, dict[str, list[tuple[float, float]]]],
) -> None:
    analysis_root = repo / "results/paper/sbi_zoo_comparison_v1"
    analysis_manifest_path = analysis_root / "manifest.json"
    rows_path = analysis_root / "rows.csv"
    p3_path = repo / "results/paper/sbi_zoo_p3_6265116/v1/manifest.json"
    _require_hash(analysis_manifest_path, ZOO_MANIFEST_SHA256)
    _require_hash(rows_path, ZOO_ROWS_SHA256)
    _require_hash(p3_path, P3_MANIFEST_SHA256)
    analysis_manifest = _load_json(analysis_manifest_path)
    p3 = _load_json(p3_path)
    if (
        analysis_manifest.get("package_identity") != ZOO_PACKAGE_IDENTITY
        or analysis_manifest.get("row_count") != 720
        or analysis_manifest.get("source_identities", {}).get("mutaco")
        != REPAIRED_COLLECTION_IDENTITY
        or analysis_manifest.get("source_identities", {}).get("p3")
        != P3_COLLECTION_IDENTITY
        or p3.get("collection_identity") != P3_COLLECTION_IDENTITY
        or p3.get("record_count") != 480
    ):
        raise RuntimeError("Prospective SBI-zoo package identity mismatch")

    with rows_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    canonical: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["budget"]), int(row["target_position"]), row["method"])
        if key in canonical or row["target_id"] != target_ids[key[1]]:
            raise RuntimeError("Invalid or duplicate canonical SBI-zoo row")
        canonical[key] = row
    if len(canonical) != 720:
        raise RuntimeError("Expected exactly 720 canonical SBI-zoo rows")

    p3_mapping = {"sbi_nle_single_round": "nle", "sbi_nre_single_round": "nre"}
    p3_records: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in p3["records"]:
        method = p3_mapping.get(record.get("method"))
        key = (record.get("budget"), record.get("target_position"), method)
        if (
            method is None
            or key in p3_records
            or record.get("target_record_id") != target_ids[record["target_position"]]
        ):
            raise RuntimeError("Invalid or duplicate P3 result record")
        p3_records[key] = record
    if len(p3_records) != 480:
        raise RuntimeError("Expected exactly 480 P3 result records")

    for budget in BUDGETS:
        points[budget]["nle"] = []
        points[budget]["nre"] = []
        for position in range(TARGET_COUNT):
            repaired = mutaco_rows[budget][position]
            mutaco = canonical[(budget, position, "mutaco")]
            computed_mutaco = points[budget]["mutaco"][position][0]
            repaired_response = repaired["errors"]["heldout_assessment_error"]["metric"][
                "component_mean_mae"
            ]
            if (
                mutaco["recommendation_identity"]
                != repaired["diagnostics"]["recommendation_identity"]
                or mutaco["result_identity"] != repaired["native_record_identity"]
                or float(mutaco["parameter_error_l2"]) != computed_mutaco
                or float(mutaco["response_error_mae"]) != repaired_response
            ):
                raise RuntimeError(f"Canonical B{budget} MUTACO row mismatch")

            for method in ("nle", "nre"):
                record = p3_records[(budget, position, method)]
                row = canonical[(budget, position, method)]
                recommended = _vector(
                    record["recommended_theta"],
                    f"B{budget} {method} target {position} recommendation",
                )
                computed_parameter = _parameter_error(recommended, truth[position])
                stored_parameter = float(record["parameter_error_l2"])
                response = _positive(
                    record["response_error_mae"],
                    f"B{budget} {method} target {position} response MAE",
                )
                if (
                    stored_parameter != computed_parameter
                    or float(row["parameter_error_l2"]) != stored_parameter
                    or float(row["response_error_mae"]) != response
                    or row["recommendation_identity"] != record["recommendation_identity"]
                    or row["result_identity"] != record["result_identity"]
                    or record["assessment_context"] != contexts[budget][position]
                ):
                    raise RuntimeError(
                        f"B{budget} {method.upper()} target {position} parameter/response mismatch"
                    )
                points[budget][method].append((stored_parameter, response))


def load_frozen_points(repo: Path) -> dict[int, dict[str, list[tuple[float, float]]]]:
    target_ids, truth = _load_targets(repo)
    points, contexts, mutaco_rows = _load_historical(repo, target_ids, truth)
    _load_prospective(repo, target_ids, truth, contexts, mutaco_rows, points)
    for budget in BUDGETS:
        for method in METHODS:
            cell = points[budget][method]
            if len(cell) != TARGET_COUNT:
                raise RuntimeError(f"B{budget} {method} does not contain 120 aligned points")
            parameter = [point[0] for point in cell]
            response = [point[1] for point in cell]
            observed = (
                statistics.fmean(parameter),
                statistics.median(parameter),
                statistics.fmean(response),
                statistics.median(response),
            )
            expected = EXPECTED_SUMMARIES[budget][method]
            if any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
                for left, right in zip(observed, expected, strict=True)
            ):
                raise RuntimeError(f"B{budget} {method} summary reproduction mismatch")
    return points


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage[T1]{fontenc}",
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def make_figure(
    points: dict[int, dict[str, list[tuple[float, float]]]]
) -> mpl.figure.Figure:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharex=True, sharey=True)
    all_parameter = [
        pair[0] for budget in BUDGETS for method in METHODS for pair in points[budget][method]
    ]
    all_response = [
        pair[1] for budget in BUDGETS for method in METHODS for pair in points[budget][method]
    ]
    xlim = (0.0, max(all_parameter) * 1.06)
    ylim = (min(all_response) / 1.35, max(all_response) * 1.35)

    for axis, budget in zip(axes, BUDGETS, strict=True):
        for method in METHODS:
            label, color = METHOD_STYLES[method]
            parameter = [pair[0] for pair in points[budget][method]]
            response = [pair[1] for pair in points[budget][method]]
            axis.scatter(
                parameter,
                response,
                s=28 if method == "mutaco" else 15,
                marker="*" if method == "mutaco" else "o",
                color=color,
                alpha=0.34,
                linewidths=0,
                zorder=2,
            )
            axis.scatter(
                [statistics.fmean(parameter)],
                [statistics.fmean(response)],
                s=240 if method == "mutaco" else 150,
                marker="*" if method == "mutaco" else "o",
                facecolor=color,
                edgecolor="#202020",
                linewidth=1.3,
                alpha=1.0,
                zorder=5,
                label=label,
            )
        axis.set_title(rf"$B = {budget}$")
        axis.set_xlim(*xlim)
        axis.set_yscale("log")
        axis.set_ylim(*ylim)
        axis.set_xlabel(r"Parameter L2 error $E_\theta$")
        axis.grid(axis="y", which="major", alpha=0.2, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=4)

    axes[0].set_ylabel("Held-out response MAE")
    handles = [
        Line2D(
            [], [], linestyle="none",
            marker="*" if method == "mutaco" else "o",
            markersize=11 if method == "mutaco" else 8,
            markerfacecolor=METHOD_STYLES[method][1], markeredgecolor="#202020",
            markeredgewidth=0.8, label=METHOD_STYLES[method][0],
        )
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.015))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.17, top=0.82, wspace=0.09)
    return fig


def _save_figure(fig: mpl.figure.Figure, pdf_path: Path, png_path: Path) -> None:
    fixed_date = datetime(1970, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        pdf_path,
        format="pdf",
        metadata={
            "Creator": "MUTACO frozen-evidence plotting",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    fig.savefig(
        png_path,
        format="png",
        metadata={"Software": "MUTACO frozen-evidence plotting"},
    )


def render_deterministically(
    points: dict[int, dict[str, list[tuple[float, float]]]], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "paper120_sbi_b40_b100_response_parameter_scatter.pdf"
    png_path = output_dir / "paper120_sbi_b40_b100_response_parameter_scatter.png"
    with tempfile.TemporaryDirectory(prefix="mutaco-sbi-joint-") as temp_name:
        temp = Path(temp_name)
        first_pdf, second_pdf = temp / "first.pdf", temp / "second.pdf"
        first_png, second_png = temp / "first.png", temp / "second.png"
        for pdf, png in ((first_pdf, first_png), (second_pdf, second_png)):
            fig = make_figure(points)
            _save_figure(fig, pdf, png)
            plt.close(fig)
        for first, second in ((first_pdf, second_pdf), (first_png, second_png)):
            if _sha256(first) != _sha256(second):
                raise RuntimeError(f"Non-deterministic render detected for {first.suffix}")
        shutil.copyfile(first_pdf, pdf_path)
        shutil.copyfile(first_png, png_path)
    return pdf_path, png_path


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    output_dir = (
        repo
        / "experiments/paper/exploration/outputs"
        / "paper120_sbi_response_parameter_scatter_v1"
    )
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    points = load_frozen_points(repo)
    for budget in BUDGETS:
        for method in METHODS:
            cell = points[budget][method]
            parameter = [pair[0] for pair in cell]
            response = [pair[1] for pair in cell]
            print(
                f"B{budget}_{METHOD_STYLES[method][0]}_n={len(cell)} "
                f"parameter_mean={statistics.fmean(parameter):.12f} "
                f"parameter_median={statistics.median(parameter):.12f} "
                f"response_mean={statistics.fmean(response):.12f} "
                f"response_median={statistics.median(response):.12f}"
            )
    pdf_path, png_path = render_deterministically(points, output_dir)
    print(f"PDF={pdf_path}")
    print(f"PDF_SHA256={_sha256(pdf_path)}")
    print(f"PNG={png_path}")
    print(f"PNG_SHA256={_sha256(png_path)}")
    print("simulator_calls=0")
    print("scientific_model_fits=0")
    print("recommendations_generated=0")
    print("new_scientific_evidence=0")


if __name__ == "__main__":
    main()
