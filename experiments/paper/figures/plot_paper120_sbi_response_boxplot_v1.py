"""Render the frozen Paper-120 historical/prospective SBI comparison.

MUTACO, NPE, and SNPE come from the repaired historical assessment.  NLE and
NRE come from the prospective SBI-zoo P3 assessment, which reused the exact
same target-specific held-out contexts.  This is not a jointly prospective
five-arm experiment.  The script performs no simulation, fitting, or
recommendation generation.
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
from matplotlib.colors import to_rgba


TARGET_COUNT = 120
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
REPAIRED_MANIFEST_SHA256 = (
    "46a0ed6b9667e13c77cb6d6a1ba12d65f4de5f4ebb11ccc23f4cc8558fee0c1c"
)
REPAIRED_EVIDENCE_IDENTITY = (
    "repaired-sbi-assessment-evidence-v1-"
    "5c8c27a1c3669d61161a8601edda7cb82e895c6849f813ec8536af796c43bc0f"
)
REPAIRED_COLLECTION_IDENTITY = (
    "repaired-sbi-assessment-collection-v1-"
    "507d012c72151554bea319511e8a04d699382407abe6c1ee97bb8c4c15c19ec6"
)
REPAIRED_ROWS_SHA256 = {
    40: "2a6c1d6835c9d818b1d942c89cdc39e56701baafa30a20d2955afb9641ef0e67",
    100: "8f7ebe1945db35b7d92f57c78eb95090599e71a94df99f8b49b6ee4db9bbbc5a",
}
ASSESSMENT_ROOTS = {40: 92021540, 100: 92021600}
ASSESSMENT_FAMILY_VECTOR_SHA256 = {
    40: "7718ed4e6e7e92644f053db2e97ee3410403f8151bb50be0e4c5c3b110116b91",
    100: "825d885466b47ce3b555753fc24213819c4a888838f86b8d06dd82e6452a4c96",
}
ZOO_MANIFEST_SHA256 = (
    "251f26eec73b6dfe58f69678bd3b92ccc58bb909ae3d4d6958e62aa2ba6a6661"
)
ZOO_ROWS_SHA256 = (
    "400c82417f9b6c5c31509c9c433760ea0b9ee9fa2f0e9c80dfd9fd1cfabfcee0"
)
ZOO_SUMMARY_SHA256 = (
    "8603fa0824b6c6f73f410c20bfe96f1f4093f74c536731d13b6c228f817b7fca"
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


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and strictly positive")
    return number


def _cohort_target_order(repo: Path) -> list[str]:
    path = repo / "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
    _require_hash(path, COHORT_SHA256)
    manifest = _load_json(path)
    if (
        manifest.get("completed_count") != TARGET_COUNT
        or manifest.get("missing_count") != 0
        or manifest.get("selected_indices") != list(range(TARGET_COUNT))
    ):
        raise RuntimeError("Frozen Paper-120 cohort scope or ordering mismatch")
    target_ids = [
        manifest["records"][str(position)]["target_record_id"]
        for position in range(TARGET_COUNT)
    ]
    if len(set(target_ids)) != TARGET_COUNT:
        raise RuntimeError("Frozen Paper-120 target identities are not unique")
    return target_ids


def _load_repaired(
    repo: Path, target_ids: list[str]
) -> tuple[
    dict[int, dict[str, list[float]]],
    dict[int, list[dict[str, Any]]],
    dict[int, dict[int, dict[str, Any]]],
]:
    root = repo / "results/paper/repaired_mutaco_sbi_assessment_6251303/v1"
    manifest_path = root / "manifest.json"
    _require_hash(manifest_path, REPAIRED_MANIFEST_SHA256)
    manifest = _load_json(manifest_path)
    if (
        manifest.get("evidence_identity") != REPAIRED_EVIDENCE_IDENTITY
        or manifest.get("collection_identity") != REPAIRED_COLLECTION_IDENTITY
        or manifest.get("cohort_sha256") != COHORT_SHA256
        or manifest.get("budgets") != [40, 100]
        or manifest.get("methods") != ["mutaco", "sbi_npe", "sbi_snpe"]
    ):
        raise RuntimeError("Repaired historical SBI evidence identity mismatch")

    values: dict[int, dict[str, list[float]]] = {}
    contexts: dict[int, list[dict[str, Any]]] = {}
    mutaco_rows: dict[int, dict[int, dict[str, Any]]] = {}
    method_mapping = {"mutaco": "mutaco", "sbi_npe": "npe", "sbi_snpe": "snpe"}
    for budget in BUDGETS:
        path = root / f"analysis_rows_b{budget}.jsonl"
        _require_hash(path, REPAIRED_ROWS_SHA256[budget])
        rows = _load_jsonl(path)
        if len(rows) != TARGET_COUNT * len(method_mapping):
            raise RuntimeError(f"Expected 360 repaired B{budget} rows")
        by_key: dict[tuple[int, str], dict[str, Any]] = {}
        row_ids: set[str] = set()
        for row in rows:
            position = row.get("target_position")
            method = row.get("method")
            row_id = row.get("row_identity")
            if (
                not isinstance(position, int)
                or not 0 <= position < TARGET_COUNT
                or method not in method_mapping
                or row.get("target_id") != target_ids[position]
                or row.get("budget") != budget
                or row.get("study_family") != "repaired_mutaco_sbi_comparison"
                or row.get("evidence_statuses") != ["valid_current_policy_evidence"]
            ):
                raise RuntimeError(f"Invalid repaired B{budget} target/method row")
            if not isinstance(row_id, str) or row_id in row_ids:
                raise RuntimeError(f"Missing or duplicate repaired B{budget} row identity")
            row_ids.add(row_id)
            key = (position, method)
            if key in by_key:
                raise RuntimeError(f"Duplicate repaired B{budget} target/method row")
            by_key[key] = row

        values[budget] = {method: [] for method in ("mutaco", "npe", "snpe")}
        contexts[budget] = []
        mutaco_rows[budget] = {}
        family_ids = []
        for position in range(TARGET_COUNT):
            trio = [by_key[(position, method)] for method in method_mapping]
            trio_contexts = [row["assessment"]["context"] for row in trio]
            if not trio_contexts[0] == trio_contexts[1] == trio_contexts[2]:
                raise RuntimeError(f"Repaired B{budget} target {position} is not paired")
            context = trio_contexts[0]
            if (
                context.get("family_definition", {}).get("assessment_campaign_seed")
                != ASSESSMENT_ROOTS[budget]
                or context.get("purpose") != "paired_heldout_assessment"
                or context.get("target_record_identity") != target_ids[position]
            ):
                raise RuntimeError(f"Wrong repaired B{budget} assessment context")
            contexts[budget].append(context)
            family_ids.append(context["family_identity"])
            mutaco_rows[budget][position] = by_key[(position, "mutaco")]
            for source_method, plot_method in method_mapping.items():
                metric = by_key[(position, source_method)]["errors"][
                    "heldout_assessment_error"
                ]
                if metric.get("available") is not True:
                    raise RuntimeError(f"Unavailable repaired B{budget} held-out endpoint")
                values[budget][plot_method].append(
                    _positive(
                        metric["metric"]["component_mean_mae"],
                        f"repaired B{budget} {plot_method} target {position}",
                    )
                )
        family_digest = hashlib.sha256(
            json.dumps(family_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if family_digest != ASSESSMENT_FAMILY_VECTOR_SHA256[budget]:
            raise RuntimeError(f"Repaired B{budget} assessment-family vector mismatch")

        canonical = manifest["reports"][str(budget)]["heldout_component_mean_mae"]
        for source_method, plot_method in method_mapping.items():
            cell = values[budget][plot_method]
            if not math.isclose(
                statistics.fmean(cell), canonical[source_method]["mean"],
                rel_tol=0.0, abs_tol=1e-15,
            ) or not math.isclose(
                statistics.median(cell), canonical[source_method]["median"],
                rel_tol=0.0, abs_tol=1e-15,
            ):
                raise RuntimeError(f"Repaired B{budget} {plot_method} summary mismatch")
    return values, contexts, mutaco_rows


def _load_zoo(
    repo: Path,
    target_ids: list[str],
    contexts: dict[int, list[dict[str, Any]]],
    mutaco_rows: dict[int, dict[int, dict[str, Any]]],
) -> dict[int, dict[str, list[float]]]:
    analysis_root = repo / "results/paper/sbi_zoo_comparison_v1"
    analysis_manifest_path = analysis_root / "manifest.json"
    rows_path = analysis_root / "rows.csv"
    summary_path = analysis_root / "summary.csv"
    p3_path = repo / "results/paper/sbi_zoo_p3_6265116/v1/manifest.json"
    _require_hash(analysis_manifest_path, ZOO_MANIFEST_SHA256)
    _require_hash(rows_path, ZOO_ROWS_SHA256)
    _require_hash(summary_path, ZOO_SUMMARY_SHA256)
    _require_hash(p3_path, P3_MANIFEST_SHA256)
    analysis_manifest = _load_json(analysis_manifest_path)
    p3 = _load_json(p3_path)
    if (
        analysis_manifest.get("package_identity") != ZOO_PACKAGE_IDENTITY
        or analysis_manifest.get("target_count") != TARGET_COUNT
        or analysis_manifest.get("row_count") != 720
        or analysis_manifest.get("source_identities", {}).get("mutaco")
        != REPAIRED_COLLECTION_IDENTITY
        or analysis_manifest.get("source_identities", {}).get("p3")
        != P3_COLLECTION_IDENTITY
        or p3.get("collection_identity") != P3_COLLECTION_IDENTITY
        or p3.get("record_count") != 480
    ):
        raise RuntimeError("Prospective SBI-zoo analysis identity mismatch")

    with rows_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 720:
        raise RuntimeError("Expected 720 canonical MUTACO/NLE/NRE analysis rows")
    by_key: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in rows:
        budget = int(row["budget"])
        position = int(row["target_position"])
        method = row["method"]
        key = (budget, position, method)
        if (
            budget not in BUDGETS
            or not 0 <= position < TARGET_COUNT
            or method not in {"mutaco", "nle", "nre"}
            or row["target_id"] != target_ids[position]
            or key in by_key
        ):
            raise RuntimeError("Invalid or duplicate SBI-zoo canonical analysis row")
        by_key[key] = row

    p3_mapping = {"sbi_nle_single_round": "nle", "sbi_nre_single_round": "nre"}
    p3_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in p3["records"]:
        method = p3_mapping.get(record.get("method"))
        key = (record.get("budget"), record.get("target_position"), method)
        if (
            method is None
            or key in p3_by_key
            or record.get("target_record_id") != target_ids[record["target_position"]]
        ):
            raise RuntimeError("Invalid or duplicate P3 result record")
        p3_by_key[key] = record
    if len(p3_by_key) != 480:
        raise RuntimeError("Expected exactly 480 P3 NLE/NRE result records")

    values = {budget: {method: [] for method in ("mutaco", "nle", "nre")} for budget in BUDGETS}
    for budget in BUDGETS:
        for position in range(TARGET_COUNT):
            repaired = mutaco_rows[budget][position]
            canonical_mutaco = by_key[(budget, position, "mutaco")]
            repaired_value = repaired["errors"]["heldout_assessment_error"]["metric"][
                "component_mean_mae"
            ]
            if (
                canonical_mutaco["source"] != "repaired_mutaco_sbi_assessment_6251303"
                or canonical_mutaco["result_identity"] != repaired["native_record_identity"]
                or float(canonical_mutaco["response_error_mae"]) != repaired_value
            ):
                raise RuntimeError(
                    f"B{budget} target {position} MUTACO is not exactly shared"
                )
            values[budget]["mutaco"].append(
                _positive(repaired_value, f"B{budget} MUTACO target {position}")
            )
            for method in ("nle", "nre"):
                canonical = by_key[(budget, position, method)]
                record = p3_by_key[(budget, position, method)]
                if (
                    canonical["source"] != "sbi_zoo_p3_6265116"
                    or canonical["result_identity"] != record["result_identity"]
                    or float(canonical["response_error_mae"]) != record["response_error_mae"]
                    or record["assessment_context"] != contexts[budget][position]
                ):
                    raise RuntimeError(
                        f"B{budget} {method.upper()} target {position} source/context mismatch"
                    )
                values[budget][method].append(
                    _positive(record["response_error_mae"], f"B{budget} {method} target {position}")
                )

    with summary_path.open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    summaries = {
        (int(row["budget"]), row["method"]): row
        for row in summary_rows
        if row["endpoint"] == "response"
    }
    for budget in BUDGETS:
        for method in ("mutaco", "nle", "nre"):
            canonical = summaries[(budget, method)]
            cell = values[budget][method]
            if int(canonical["n"]) != TARGET_COUNT or not math.isclose(
                statistics.fmean(cell), float(canonical["mean"]),
                rel_tol=0.0, abs_tol=1e-15,
            ) or not math.isclose(
                statistics.median(cell), float(canonical["median"]),
                rel_tol=0.0, abs_tol=1e-15,
            ):
                raise RuntimeError(f"Canonical B{budget} {method} summary mismatch")
    return values


def load_frozen_values(repo: Path) -> dict[int, dict[str, list[float]]]:
    target_ids = _cohort_target_order(repo)
    repaired, contexts, mutaco_rows = _load_repaired(repo, target_ids)
    zoo = _load_zoo(repo, target_ids, contexts, mutaco_rows)
    return {
        budget: {
            "mutaco": repaired[budget]["mutaco"],
            "npe": repaired[budget]["npe"],
            "snpe": repaired[budget]["snpe"],
            "nle": zoo[budget]["nle"],
            "nre": zoo[budget]["nre"],
        }
        for budget in BUDGETS
    }


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


def _target_jitter(position: int) -> float:
    rank = (position * 73) % TARGET_COUNT
    return -0.135 + 0.27 * rank / (TARGET_COUNT - 1)


def make_figure(values: dict[int, dict[str, list[float]]]) -> mpl.figure.Figure:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    all_values = [
        value
        for budget in BUDGETS
        for method in METHODS
        for value in values[budget][method]
    ]
    ymin, ymax = min(all_values) / 1.35, max(all_values) * 1.35

    for axis, budget in zip(axes, BUDGETS, strict=True):
        positions = list(range(1, len(METHODS) + 1))
        cells = [values[budget][method] for method in METHODS]
        boxes = axis.boxplot(
            cells,
            positions=positions,
            widths=0.48,
            whis=1.5,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#202020", "linewidth": 2.0},
            whiskerprops={"linewidth": 1.25},
            capprops={"linewidth": 1.25},
        )
        for method_index, method in enumerate(METHODS):
            _, color = METHOD_STYLES[method]
            box = boxes["boxes"][method_index]
            box.set_edgecolor(color)
            box.set_linewidth(1.6)
            box.set_facecolor(to_rgba(color, 0.18))
            for artist in (
                boxes["whiskers"][2 * method_index],
                boxes["whiskers"][2 * method_index + 1],
                boxes["caps"][2 * method_index],
                boxes["caps"][2 * method_index + 1],
            ):
                artist.set_color(color)
            axis.scatter(
                [
                    positions[method_index] + _target_jitter(position)
                    for position in range(TARGET_COUNT)
                ],
                values[budget][method],
                s=13,
                color=color,
                alpha=0.48,
                linewidths=0,
                zorder=3,
            )
        axis.set_title(rf"$B = {budget}$")
        axis.set_xticks(positions, [METHOD_STYLES[method][0] for method in METHODS])
        axis.set_yscale("log")
        axis.set_ylim(ymin, ymax)
        axis.grid(axis="y", which="major", alpha=0.2, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=4)

    axes[0].set_ylabel("Held-out response MAE")
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.16, top=0.90, wspace=0.09)
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
    values: dict[int, dict[str, list[float]]], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "paper120_sbi_b40_b100_response_boxplot.pdf"
    png_path = output_dir / "paper120_sbi_b40_b100_response_boxplot.png"
    with tempfile.TemporaryDirectory(prefix="mutaco-sbi-response-") as temp_name:
        temp = Path(temp_name)
        first_pdf, second_pdf = temp / "first.pdf", temp / "second.pdf"
        first_png, second_png = temp / "first.png", temp / "second.png"
        for pdf, png in ((first_pdf, first_png), (second_pdf, second_png)):
            fig = make_figure(values)
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
        / "paper120_sbi_response_boxplot_v1"
    )
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    values = load_frozen_values(repo)
    for budget in BUDGETS:
        for method in METHODS:
            cell = values[budget][method]
            print(
                f"B{budget}_{METHOD_STYLES[method][0]}_n={len(cell)} "
                f"mean={statistics.fmean(cell):.12f} "
                f"median={statistics.median(cell):.12f}"
            )
    pdf_path, png_path = render_deterministically(values, output_dir)
    print(f"PDF={pdf_path}")
    print(f"PDF_SHA256={_sha256(pdf_path)}")
    print(f"PNG={png_path}")
    print(f"PNG_SHA256={_sha256(png_path)}")
    print("simulator_calls=0")
    print("scientific_model_fits=0")
    print("recommendations_generated=0")
    print("new_scientific_evidence=0")


# Preserve historical global/pickle references across the compatibility import.
for _legacy_export in (
    _sha256,
    _require_hash,
    _load_json,
    _load_jsonl,
    _positive,
    _cohort_target_order,
    _load_repaired,
    _load_zoo,
    load_frozen_values,
    configure_plotting,
    _target_jitter,
    make_figure,
    _save_figure,
    render_deterministically,
    main,
):
    _legacy_export.__module__ = "experiments.paper.exploration.plot_paper120_sbi_response_boxplot_v1"
del _legacy_export


def _cli() -> None:
    """Describe the offline renderer without starting a render for --help."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()


if __name__ == "__main__":
    _cli()
