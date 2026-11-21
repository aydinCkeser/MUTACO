"""Render the frozen Paper-120 primary B40/B100 population comparison.

This review-only plot reads the deterministic cross-budget synthesis package.
It performs no simulation, fitting, recommendation generation, or scientific
analysis beyond reproducing the package's canonical descriptive summaries.
"""

from __future__ import annotations

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


PACKAGE_IDENTITY = (
    "scalar-primary-cross-budget-synthesis-package-v1-"
    "b3e3e28b695534b8ec550ed4da7d75a69c3ad028794da050f864535bd8ddab3f"
)
REPORT_IDENTITY = (
    "scalar-primary-cross-budget-report-v1-"
    "d695cc49c31c5c884c97058e3140a17abaf1be77f61dc54b15a21460621eba33"
)
SOURCE_EVIDENCE_IDENTITIES = {
    "40": (
        "scalar-primary-population-evidence-v1-"
        "9294220a23a94c7514dfd608ea6a7b0dbe3f8474222d30309e7b4a9aeef04009"
    ),
    "100": (
        "scalar-primary-population-evidence-v1-"
        "c72415fcb29e8d7379604d3cf0a33a3b0e962fed94ead446088c55ece83a9375"
    ),
}
EXPECTED_SHA256 = {
    "manifest.json": "72e39d9041abcc73b3d3bbaa0fef3839502041ecfec67bcc92c355cd9ff45e98",
    "cross_budget_rows.jsonl": (
        "25c33cce995cd4ab7f90ca26eefaa5b0a14ae0f83116eb3020db3028fbcf727d"
    ),
    "synthesis_report.json": (
        "140e683bea0f278c72136c01171faeb0a1a2953cabf583e3095b8e42a8f43aa9"
    ),
}
BUDGETS = ("40", "100")
METHODS = ("mutaco", "scalar_qlognei", "sobol")
METHOD_STYLES = {
    "mutaco": ("MUTACO", "#0072B2"),
    "scalar_qlognei": ("qLogNEI", "#E66B2C"),
    "sobol": ("Sobol", "#009E73"),
}
EXPECTED_CAMPAIGNS = {
    "40": "graph_paper120_scalar_primary_b40_v1",
    "100": "graph_paper120_scalar_primary_b100_v1",
}
TARGET_COUNT = 120


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(f"Frozen input hash mismatch for {path}: {observed}")


def load_frozen_values(
    package_root: Path,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, str]]]:
    """Load and validate all 720 plotted values and their target identities."""

    paths = {name: package_root / name for name in EXPECTED_SHA256}
    for name, expected in EXPECTED_SHA256.items():
        _require_hash(paths[name], expected)

    manifest = _load_json(paths["manifest.json"])
    report = _load_json(paths["synthesis_report.json"])
    rows = [
        json.loads(line)
        for line in paths["cross_budget_rows.jsonl"].read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]

    if manifest.get("package_identity") != PACKAGE_IDENTITY:
        raise RuntimeError("Cross-budget package identity mismatch")
    if manifest.get("source_evidence_identities") != SOURCE_EVIDENCE_IDENTITIES:
        raise RuntimeError("Primary B40/B100 source evidence mismatch")
    if manifest.get("target_count") != TARGET_COUNT or manifest.get("row_count") != 360:
        raise RuntimeError("Cross-budget package does not describe Paper-120")
    if manifest.get("methods") != list(METHODS) or manifest.get("budgets") != [40, 100]:
        raise RuntimeError("Cross-budget method or budget ordering mismatch")
    if report.get("synthesis_report_identity") != REPORT_IDENTITY:
        raise RuntimeError("Canonical synthesis report identity mismatch")
    if report.get("source_evidence_identities") != SOURCE_EVIDENCE_IDENTITIES:
        raise RuntimeError("Synthesis report source evidence mismatch")
    if report.get("target_count") != TARGET_COUNT or report.get("row_count") != 360:
        raise RuntimeError("Synthesis report does not describe Paper-120")
    if len(rows) != TARGET_COUNT * len(METHODS):
        raise RuntimeError(f"Expected 360 cross-budget rows, found {len(rows)}")

    values = {
        budget: {method: [] for method in METHODS}
        for budget in BUDGETS
    }
    identities = {method: {} for method in METHODS}
    row_ids: set[str] = set()

    for row in rows:
        method = row.get("method")
        position = row.get("target_position")
        target_id = row.get("target_id")
        row_id = row.get("row_identity")
        if method not in METHODS:
            raise RuntimeError(f"Unexpected method in synthesis rows: {method!r}")
        if not isinstance(position, int) or not 0 <= position < TARGET_COUNT:
            raise RuntimeError(f"Invalid Paper-120 target position: {position!r}")
        if not isinstance(target_id, str) or not target_id.startswith(
            "frozen-target-record-v2-"
        ):
            raise RuntimeError(f"Invalid target identity at position {position}")
        if not isinstance(row_id, str) or row_id in row_ids:
            raise RuntimeError("Missing or duplicate cross-budget row identity")
        row_ids.add(row_id)
        if position in identities[method]:
            raise RuntimeError(f"Duplicate {method} target position {position}")
        identities[method][position] = target_id

        endpoints = row.get("budget_endpoints")
        if not isinstance(endpoints, dict) or set(endpoints) != set(BUDGETS):
            raise RuntimeError(f"Missing budget endpoint at target {position}")
        for budget in BUDGETS:
            endpoint = endpoints[budget]
            value = float(endpoint["component_mean_mae"])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    f"Non-positive/non-finite {method} B{budget} value at {position}"
                )
            context = endpoint.get("assessment_context", {})
            coordinates = context.get("coordinate", {}).get("coordinates", [])
            if (
                len(coordinates) < 3
                or coordinates[0] != EXPECTED_CAMPAIGNS[budget]
                or coordinates[1] != target_id
                or coordinates[2] != int(budget)
                or context.get("purpose") != "paired_held_out_assessment"
            ):
                raise RuntimeError(
                    f"Wrong primary held-out context for {method} B{budget} target {position}"
                )
            values[budget][method].append((position, value))

    expected_positions = set(range(TARGET_COUNT))
    reference_targets = identities[METHODS[0]]
    for method in METHODS:
        if set(identities[method]) != expected_positions:
            raise RuntimeError(f"{method} does not cover exactly target positions 0..119")
        if identities[method] != reference_targets:
            raise RuntimeError(f"{method} target identities are not Paper-120 aligned")

    ordered_values: dict[str, dict[str, list[float]]] = {
        budget: {} for budget in BUDGETS
    }
    for budget in BUDGETS:
        for method in METHODS:
            pairs = sorted(values[budget][method])
            cell = [value for _, value in pairs]
            if len(cell) != TARGET_COUNT:
                raise RuntimeError(f"B{budget} {method} n={len(cell)}, expected 120")
            canonical = report["primary_endpoint_by_budget"][budget][
                "heldout_component_mean_mae"
            ][method]
            mean = statistics.fmean(cell)
            median = statistics.median(cell)
            if canonical.get("n") != TARGET_COUNT:
                raise RuntimeError(f"Canonical B{budget} {method} n mismatch")
            if not math.isclose(mean, canonical["mean"], rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError(f"Canonical B{budget} {method} mean mismatch")
            if not math.isclose(
                median, canonical["median"], rel_tol=0.0, abs_tol=1e-15
            ):
                raise RuntimeError(f"Canonical B{budget} {method} median mismatch")
            ordered_values[budget][method] = cell

    target_map = {str(position): target_id for position, target_id in reference_targets.items()}
    return ordered_values, {"target_id_by_position": target_map}


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
    """Return a deterministic, budget/method-independent target offset."""

    # Multiplication by 73 permutes the 120 target positions.  Centering the
    # resulting ranks gives a reproducible, even spread without ambient RNG.
    rank = (position * 73) % TARGET_COUNT
    return -0.135 + 0.27 * rank / (TARGET_COUNT - 1)


def make_figure(values: dict[str, dict[str, list[float]]]) -> mpl.figure.Figure:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)

    all_values = [
        value
        for budget in BUDGETS
        for method in METHODS
        for value in values[budget][method]
    ]
    ymin = min(all_values) / 1.35
    ymax = max(all_values) * 1.35

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

            x_values = [
                positions[method_index] + _target_jitter(position)
                for position in range(TARGET_COUNT)
            ]
            axis.scatter(
                x_values,
                values[budget][method],
                s=13,
                color=color,
                alpha=0.48,
                linewidths=0,
                zorder=3,
            )

        axis.set_title(rf"$B = {budget}$")
        axis.set_xticks(
            positions,
            [METHOD_STYLES[method][0] for method in METHODS],
        )
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
    values: dict[str, dict[str, list[float]]], output_dir: Path
) -> tuple[Path, Path]:
    """Double-render, byte-compare, and publish the two review outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "paper120_primary_b40_b100_boxplot.pdf"
    png_path = output_dir / "paper120_primary_b40_b100_boxplot.png"
    with tempfile.TemporaryDirectory(prefix="mutaco-primary-boxplot-") as temp_name:
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
    package_root = (
        repo / "results/paper/scalar_primary_b40_b100_synthesis_v1"
    )
    output_dir = (
        repo
        / "experiments/paper/exploration/outputs"
        / "paper120_primary_boxplot_v1"
    )

    # Matplotlib honors this for reproducible PDF metadata in supported builds.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    values, _ = load_frozen_values(package_root)
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


# Preserve historical global/pickle references across the compatibility import.
for _legacy_export in (
    _sha256,
    _load_json,
    _require_hash,
    load_frozen_values,
    configure_plotting,
    _target_jitter,
    make_figure,
    _save_figure,
    render_deterministically,
    main,
):
    _legacy_export.__module__ = "experiments.paper.exploration.plot_paper120_primary_boxplot_v1"
del _legacy_export


def _cli() -> None:
    """Describe the offline renderer without starting a render for --help."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()


if __name__ == "__main__":
    _cli()
