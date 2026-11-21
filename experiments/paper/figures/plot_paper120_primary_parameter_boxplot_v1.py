"""Render Paper-120 primary B40/B100 physical-parameter recovery.

The figure is derived only from the frozen Paper-120 target records and the
frozen scalar-primary compact exports.  It performs no simulation, fitting, or
recommendation generation.
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


TARGET_COUNT = 120
PARAMETER_COUNT = 4
BUDGETS = (40, 100)
METHODS = ("mutaco", "scalar_qlognei", "sobol")
METHOD_STYLES = {
    "mutaco": ("MUTACO", "#0072B2"),
    "scalar_qlognei": ("qLogNEI", "#E66B2C"),
    "sobol": ("Sobol", "#009E73"),
}
COHORT_MANIFEST_SHA256 = (
    "359fc0bb59fc9a8a52f5da684e7a7eb352466bdc0d0a54fb8efa94131ce423bb"
)
EVIDENCE = {
    40: {
        "manifest_sha256": (
            "ecfbf6621931a17e7bd62e1a8cda13137cbeb7949cc13f7aa4bcef7f60f985be"
        ),
        "evidence_identity": (
            "scalar-primary-population-evidence-v1-"
            "9294220a23a94c7514dfd608ea6a7b0dbe3f8474222d30309e7b4a9aeef04009"
        ),
        "compact_sha256": (
            "f02bd7343292cf0a9768fb83ffc85019177feceb0afb1003c65f834b3ffcbbe9"
        ),
        "collection_identity": (
            "scalar-primary-global-collection-v1-"
            "b0b4c16f86d9b7ee2a3bfedbef4838f4e730a4187a29c74dd89a152be134ea24"
        ),
        "campaign_id": "graph_paper120_scalar_primary_b40_v1",
        "compact_relative_path": (
            "experiments/paper/exploration/outputs/scalar_primary_smoke/"
            "population_b40_v3r2_compact_fetch/population/collection/compact_export.json"
        ),
    },
    100: {
        "manifest_sha256": (
            "7f1c79a1351457cd8f0ee918bcfc0077375635e334803f54beee6fd1428436e8"
        ),
        "evidence_identity": (
            "scalar-primary-population-evidence-v1-"
            "c72415fcb29e8d7379604d3cf0a33a3b0e962fed94ead446088c55ece83a9375"
        ),
        "compact_sha256": (
            "1e2aacfdafe754bdd2697c85df2731d50ec9160807f8cbfbe0634199bc12497a"
        ),
        "collection_identity": (
            "scalar-primary-global-collection-v1-"
            "03cf838d3651496e4544a4627396589cc4071e0cb2a23a53288cf2057d478685"
        ),
        "campaign_id": "graph_paper120_scalar_primary_b100_v1",
        "compact_relative_path": (
            "experiments/paper/exploration/outputs/scalar_primary_smoke/"
            "population_b100_v3r2_compact_fetch/population/collection/compact_export.json"
        ),
    },
}
# These values are validation-only reproductions from the read-only preflight.
# They are never used to construct plotted observations.
EXPECTED_SUMMARIES = {
    40: {
        "mutaco": (0.3070161494413837, 0.2369729089385459),
        "scalar_qlognei": (0.5554034225654267, 0.5405906995213006),
        "sobol": (0.6456479163075896, 0.6259898092044596),
    },
    100: {
        "mutaco": (0.30611849989549006, 0.2042701803367109),
        "scalar_qlognei": (0.38256131599398396, 0.3360013608144075),
        "sobol": (0.5381257953212492, 0.5136764835044759),
    },
}


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


def _stable_hash(prefix: str, value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(canonical).hexdigest()}"


def _finite_vector(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != PARAMETER_COUNT:
        raise RuntimeError(f"{label} must contain exactly four physical parameters")
    vector = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in vector):
        raise RuntimeError(f"{label} contains a non-finite parameter")
    return vector


def _load_frozen_targets(repo: Path) -> dict[str, tuple[int, tuple[float, ...]]]:
    cohort_root = repo / "experiments/graph_inversion/frozen_targets/paper120_v3"
    manifest_path = cohort_root / "manifest.json"
    _require_hash(manifest_path, COHORT_MANIFEST_SHA256)
    sidecar = manifest_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="ascii").strip() != COHORT_MANIFEST_SHA256:
        raise RuntimeError("Paper-120 cohort manifest sidecar mismatch")
    manifest = _load_json(manifest_path)
    if manifest.get("completed_count") != TARGET_COUNT or manifest.get("missing_count") != 0:
        raise RuntimeError("Frozen cohort is not the complete Paper-120 cohort")
    if manifest.get("selected_indices") != list(range(TARGET_COUNT)):
        raise RuntimeError("Frozen cohort target ordering mismatch")

    targets: dict[str, tuple[int, tuple[float, ...]]] = {}
    for position in range(TARGET_COUNT):
        path = cohort_root / f"target_{position:03d}.json"
        expected_file_hash = path.with_suffix(".sha256").read_text(
            encoding="ascii"
        ).strip()
        _require_hash(path, expected_file_hash)
        record = _load_json(path)
        if record.get("target_index") != position:
            raise RuntimeError(f"Frozen target position mismatch: {position}")
        content = {
            key: value
            for key, value in record.items()
            if key not in {"content_hash", "target_record_id"}
        }
        content_hash = _stable_hash("frozen-target-content-v2", content)
        record_id = _stable_hash(
            "frozen-target-record-v2", {**content, "content_hash": content_hash}
        )
        if record.get("content_hash") != content_hash or record.get(
            "target_record_id"
        ) != record_id:
            raise RuntimeError(f"Frozen target content identity mismatch: {position}")
        manifest_record = manifest["records"].get(str(position))
        if manifest_record != {
            "content_hash": content_hash,
            "target_record_id": record_id,
        }:
            raise RuntimeError(f"Cohort manifest target binding mismatch: {position}")
        if record_id in targets:
            raise RuntimeError(f"Duplicate Paper-120 target identity: {record_id}")
        targets[record_id] = (
            position,
            _finite_vector(record.get("theta_star"), f"target {position} theta_star"),
        )
    return targets


def _load_budget_recommendations(
    repo: Path,
    budget: int,
    targets: dict[str, tuple[int, tuple[float, ...]]],
) -> dict[str, dict[int, tuple[float, ...]]]:
    evidence = EVIDENCE[budget]
    manifest_path = (
        repo / f"results/paper/graph_scalar_primary_b{budget}_v1/manifest.json"
    )
    compact_path = repo / str(evidence["compact_relative_path"])
    _require_hash(manifest_path, str(evidence["manifest_sha256"]))
    _require_hash(compact_path, str(evidence["compact_sha256"]))
    manifest = _load_json(manifest_path)
    compact = _load_json(compact_path)

    if manifest.get("evidence_identity") != evidence["evidence_identity"]:
        raise RuntimeError(f"Authoritative B{budget} evidence identity mismatch")
    if manifest.get("budget") != budget or manifest.get("targets") != TARGET_COUNT:
        raise RuntimeError(f"Authoritative B{budget} manifest scope mismatch")
    if manifest.get("methods") != list(METHODS):
        raise RuntimeError(f"Authoritative B{budget} method ordering mismatch")
    if manifest.get("integrity", {}).get("compact_export_sha256") != evidence[
        "compact_sha256"
    ]:
        raise RuntimeError(f"B{budget} manifest does not bind the compact export")
    if compact.get("collection_identity") != evidence["collection_identity"]:
        raise RuntimeError(f"B{budget} compact collection identity mismatch")
    if compact.get("scientific_calls_added_by_export") != 0:
        raise RuntimeError(f"B{budget} compact export reports scientific calls")

    records = compact.get("records")
    if not isinstance(records, list) or len(records) != TARGET_COUNT * len(METHODS):
        raise RuntimeError(f"Expected 360 B{budget} recommendation records")
    recommendations: dict[str, dict[int, tuple[float, ...]]] = {
        method: {} for method in METHODS
    }
    record_ids: set[str] = set()
    for record in records:
        method = record.get("method")
        target_id = record.get("target_id")
        record_id = record.get("record_identity")
        if method not in METHODS or target_id not in targets:
            raise RuntimeError(f"Unexpected B{budget} method/target record")
        if not isinstance(record_id, str) or record_id in record_ids:
            raise RuntimeError(f"Missing or duplicate B{budget} record identity")
        record_ids.add(record_id)
        if (
            record.get("budget") != budget
            or record.get("campaign_id") != evidence["campaign_id"]
            or record.get("status") != "completed"
        ):
            raise RuntimeError(f"Incomplete or wrong-campaign B{budget} record")
        decision = record.get("decision")
        if (
            not isinstance(decision, dict)
            or decision.get("status") != "recommended"
            or decision.get("method") != method
            or decision.get("budget") != budget
        ):
            raise RuntimeError(f"Invalid B{budget} {method} recommendation")
        position = targets[target_id][0]
        if position in recommendations[method]:
            raise RuntimeError(f"Duplicate B{budget} {method} target {position}")
        recommendations[method][position] = _finite_vector(
            decision.get("recommended_theta"),
            f"B{budget} {method} target {position} recommendation",
        )

    expected_positions = set(range(TARGET_COUNT))
    for method in METHODS:
        if set(recommendations[method]) != expected_positions:
            raise RuntimeError(f"B{budget} {method} target coverage mismatch")
    return recommendations


def load_parameter_errors(repo: Path) -> dict[int, dict[str, list[float]]]:
    targets = _load_frozen_targets(repo)
    truth_by_position = {
        position: theta for position, theta in targets.values()
    }
    errors: dict[int, dict[str, list[float]]] = {}
    for budget in BUDGETS:
        recommendations = _load_budget_recommendations(repo, budget, targets)
        errors[budget] = {}
        for method in METHODS:
            cell = []
            for position in range(TARGET_COUNT):
                recommended = recommendations[method][position]
                truth = truth_by_position[position]
                error = math.sqrt(
                    sum(
                        (recommended[index] - truth[index]) ** 2
                        for index in range(PARAMETER_COUNT)
                    )
                )
                if not math.isfinite(error) or error < 0.0:
                    raise RuntimeError(
                        f"Invalid B{budget} {method} parameter error at {position}"
                    )
                cell.append(error)
            mean = statistics.fmean(cell)
            median = statistics.median(cell)
            expected_mean, expected_median = EXPECTED_SUMMARIES[budget][method]
            if not math.isclose(mean, expected_mean, rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError(f"B{budget} {method} mean reproduction mismatch")
            if not math.isclose(
                median, expected_median, rel_tol=0.0, abs_tol=1e-15
            ):
                raise RuntimeError(f"B{budget} {method} median reproduction mismatch")
            errors[budget][method] = cell
    return errors


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


def make_figure(errors: dict[int, dict[str, list[float]]]) -> mpl.figure.Figure:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    maximum = max(
        value
        for budget in BUDGETS
        for method in METHODS
        for value in errors[budget][method]
    )

    for axis, budget in zip(axes, BUDGETS, strict=True):
        positions = list(range(1, len(METHODS) + 1))
        cells = [errors[budget][method] for method in METHODS]
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
                errors[budget][method],
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
        axis.set_ylim(0.0, maximum * 1.08)
        axis.grid(axis="y", which="major", alpha=0.2, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=4)

    axes[0].set_ylabel(r"Parameter error $E_\theta$")
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
    errors: dict[int, dict[str, list[float]]], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "paper120_primary_b40_b100_parameter_boxplot.pdf"
    png_path = output_dir / "paper120_primary_b40_b100_parameter_boxplot.png"
    with tempfile.TemporaryDirectory(prefix="mutaco-primary-parameter-") as temp_name:
        temp = Path(temp_name)
        first_pdf, second_pdf = temp / "first.pdf", temp / "second.pdf"
        first_png, second_png = temp / "first.png", temp / "second.png"
        for pdf, png in ((first_pdf, first_png), (second_pdf, second_png)):
            fig = make_figure(errors)
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
        / "paper120_primary_parameter_boxplot_v1"
    )
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    errors = load_parameter_errors(repo)
    for budget in BUDGETS:
        for method in METHODS:
            cell = errors[budget][method]
            print(
                f"B{budget}_{METHOD_STYLES[method][0]}_n={len(cell)} "
                f"mean={statistics.fmean(cell):.12f} "
                f"median={statistics.median(cell):.12f}"
            )
    pdf_path, png_path = render_deterministically(errors, output_dir)
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
    _stable_hash,
    _finite_vector,
    _load_frozen_targets,
    _load_budget_recommendations,
    load_parameter_errors,
    configure_plotting,
    _target_jitter,
    make_figure,
    _save_figure,
    render_deterministically,
    main,
):
    _legacy_export.__module__ = "experiments.paper.exploration.plot_paper120_primary_parameter_boxplot_v1"
del _legacy_export


def _cli() -> None:
    """Describe the offline renderer without starting a render for --help."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()


if __name__ == "__main__":
    _cli()
