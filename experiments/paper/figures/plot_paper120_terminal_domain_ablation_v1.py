"""Plot the frozen Paper-120 B40 terminal-domain held-out ablation.

The figure compares the evaluated-point terminal recommendation (B') with the
continuous terminal recommendation (C') using only the completed, paired v2
held-out evidence.  It performs no simulation, GP fitting, recommendation
generation, or scientific-evidence mutation.
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
POLICIES = ("B_prime", "C_prime")
POLICY_STYLES = {
    "B_prime": (r"B$'$", "#777777"),
    "C_prime": (r"C$'$", "#0072B2"),
}

COHORT_SHA256 = (
    "359fc0bb59fc9a8a52f5da684e7a7eb352466bdc0d0a54fb8efa94131ce423bb"
)
REFIT_MANIFEST_SHA256 = (
    "85efb048ce24549cb145f324e6c27bdebc120e93a3b8d9fd6088bb752790dd08"
)
REFIT_ROWS_SHA256 = (
    "b88cd087fa7b2c1053ea749ef5750c01cc54b4b11a64b35a276df39f18bdc101"
)
REFIT_PACKAGE_IDENTITY = (
    "terminal-domain-ablation-b40-refit-v1-"
    "57b9b6308f16e628c5a7b3d5c4b7bc8da66c56f555e78d4e3b4978f691829a7e"
)
DECLARATION_SHA256 = (
    "7e2f4e7afe5a3f9cd8f070408980ba8964d8685c3bf94203cb6a6ff812c2d11f"
)
DECLARATION_IDENTITY = (
    "terminal-domain-heldout-declaration-v2-"
    "8becf876b147ebd3799dd40e68e18c5c71e14d411bcd00ae12e42bfa3779efa0"
)
EVIDENCE_MANIFEST_SHA256 = (
    "213a58b76df542dcb2f332063abcc3846cbe6238918fd1f36c13c30724e80fad"
)
EVIDENCE_PACKAGE_IDENTITY = (
    "terminal-domain-heldout-evidence-package-v2-"
    "dc9df5a726dfc00fe05cebcbb5a94ce6963d98f1783f842986c92c0a7588142e"
)
ANALYSIS_REPORT_SHA256 = (
    "19e126c932c134934713273dbff0495f646c6dccafa8595da373283348930f66"
)
ANALYSIS_IDENTITY = (
    "terminal-domain-heldout-analysis-v2-"
    "9545e519fa4d5a2560bf6fdb35051ad5ce52a10706df7b534277b3b3cbcfd671"
)
PAIRED_RESULTS_SHA256 = (
    "be8f7f48e1fa98ad154ab71631984114ea5f2a9f1c96ecc3962cf8420998b416"
)
CAMPAIGN_ID = "paper120_b40_terminal_domain_bprime_cprime_heldout_v2"
CONTEXT_NAMESPACE = "terminal_domain_paired_heldout_assessment"
ASSESSMENT_ROOT_SEED = 92021841
CANONICAL_INTERVAL = (-0.014645077700721048, -0.006026733225736714)
EXPECTED_SUMMARIES = {
    "B_prime_mean": 0.03596357617714898,
    "B_prime_median": 0.0279502446842713,
    "C_prime_mean": 0.025730973025557283,
    "C_prime_median": 0.02244389447350841,
    "delta_mean": -0.010232603151591697,
    "delta_median": -0.007135361108703362,
    "relative_mean_change": -0.2845268529800278,
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


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and strictly positive")
    return number


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be finite")
    return number


def load_frozen_evidence(
    repo: Path,
) -> tuple[dict[str, list[float]], list[float], tuple[float, float]]:
    cohort_path = repo / "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
    refit_root = repo / "results/paper/terminal_domain_ablation_b40_refit_v1"
    evidence_root = (
        repo / "results/paper/terminal_domain_heldout_assessment_b40_v2"
    )
    refit_manifest_path = refit_root / "manifest.json"
    refit_rows_path = refit_root / "rows.jsonl"
    declaration_path = evidence_root / "declaration.json"
    evidence_manifest_path = evidence_root / "manifest.json"
    analysis_report_path = evidence_root / "analysis_report.json"
    paired_results_path = evidence_root / "paired_results.jsonl"

    for path, expected in (
        (cohort_path, COHORT_SHA256),
        (refit_manifest_path, REFIT_MANIFEST_SHA256),
        (refit_rows_path, REFIT_ROWS_SHA256),
        (declaration_path, DECLARATION_SHA256),
        (evidence_manifest_path, EVIDENCE_MANIFEST_SHA256),
        (analysis_report_path, ANALYSIS_REPORT_SHA256),
        (paired_results_path, PAIRED_RESULTS_SHA256),
    ):
        _require_hash(path, expected)

    cohort = _load_json(cohort_path)
    refit_manifest = _load_json(refit_manifest_path)
    declaration = _load_json(declaration_path)
    evidence_manifest = _load_json(evidence_manifest_path)
    report = _load_json(analysis_report_path)
    rows = _load_jsonl(paired_results_path)

    if (
        cohort.get("completed_count") != TARGET_COUNT
        or cohort.get("missing_count") != 0
        or cohort.get("selected_indices") != list(range(TARGET_COUNT))
    ):
        raise RuntimeError("Frozen Paper-120 cohort scope or ordering mismatch")
    target_ids = [
        cohort["records"][str(position)]["target_record_id"]
        for position in range(TARGET_COUNT)
    ]
    if len(set(target_ids)) != TARGET_COUNT:
        raise RuntimeError("Frozen Paper-120 target identities are not unique")

    if (
        refit_manifest.get("package_identity") != REFIT_PACKAGE_IDENTITY
        or refit_manifest.get("completed_targets") != TARGET_COUNT
        or refit_manifest.get("simulator_calls") != 0
        or refit_manifest.get("heldout_assessment_calls") != 0
    ):
        raise RuntimeError("Frozen B'/C' refit recommendation package mismatch")
    recommendation_source = declaration.get("recommendation_source", {})
    if (
        declaration.get("declaration_identity") != DECLARATION_IDENTITY
        or declaration.get("campaign_id") != CAMPAIGN_ID
        or declaration.get("cohort", {}).get("sha256") != COHORT_SHA256
        or declaration.get("cohort", {}).get("targets") != TARGET_COUNT
        or recommendation_source.get("authority")
        != "sole_authoritative_B_prime_C_prime_recommendation_source"
        or recommendation_source.get("package_identity") != REFIT_PACKAGE_IDENTITY
        or recommendation_source.get("manifest_sha256") != REFIT_MANIFEST_SHA256
        or recommendation_source.get("rows_sha256") != REFIT_ROWS_SHA256
    ):
        raise RuntimeError("Frozen held-out declaration provenance mismatch")

    if (
        evidence_manifest.get("package_identity") != EVIDENCE_PACKAGE_IDENTITY
        or evidence_manifest.get("analysis_identity") != ANALYSIS_IDENTITY
        or evidence_manifest.get("declaration_identity") != DECLARATION_IDENTITY
        or evidence_manifest.get("analysis_report_sha256")
        != ANALYSIS_REPORT_SHA256
        or evidence_manifest.get("paired_results_sha256") != PAIRED_RESULTS_SHA256
        or evidence_manifest.get("verdict")
        != "TERMINAL_DOMAIN_PHYSICAL_BENEFIT_SUPPORTED"
    ):
        raise RuntimeError("Frozen held-out evidence package mismatch")
    if (
        report.get("analysis_identity") != ANALYSIS_IDENTITY
        or report.get("declaration_identity") != DECLARATION_IDENTITY
        or report.get("complete_pairs") != TARGET_COUNT
        or report.get("declared_targets") != TARGET_COUNT
        or report.get("confirmatory_complete") is not True
        or report.get("verdict") != "TERMINAL_DOMAIN_PHYSICAL_BENEFIT_SUPPORTED"
    ):
        raise RuntimeError("Frozen held-out analysis identity mismatch")
    if len(rows) != TARGET_COUNT:
        raise RuntimeError("Expected exactly 120 paired held-out rows")

    declaration_contexts = declaration.get("assessment_contexts", {}).get(
        "mapping", {}
    )
    if set(declaration_contexts) != set(target_ids):
        raise RuntimeError("Declaration context mapping does not match Paper-120")

    values = {policy: [] for policy in POLICIES}
    deltas: list[float] = []
    pair_ids: set[str] = set()
    call_ids: set[str] = set()
    outcome_ids: set[str] = set()
    for position, row in enumerate(rows):
        target_id = target_ids[position]
        context = row.get("assessment_context")
        if (
            row.get("schema_version") != "mutaco-terminal-domain-heldout-pair-v2"
            or row.get("target_position") != position
            or row.get("target_record_identity") != target_id
            or context != declaration_contexts[target_id]
            or context.get("family_definition", {}).get("campaign_id")
            != CAMPAIGN_ID
            or context.get("family_definition", {}).get("target_record_identity")
            != target_id
            or context.get("coordinate", {}).get("namespace") != CONTEXT_NAMESPACE
            or context.get("root_seed") != ASSESSMENT_ROOT_SEED
            or context.get("replicate") != 0
        ):
            raise RuntimeError(f"Target/context alignment mismatch at position {position}")

        pair_id = row.get("pair_identity")
        if not isinstance(pair_id, str) or pair_id in pair_ids:
            raise RuntimeError(f"Missing or duplicate pair identity at {position}")
        pair_ids.add(pair_id)
        for policy in POLICIES:
            policy_record = row.get(policy, {})
            value = _positive(
                policy_record.get("heldout_physical_mae"),
                f"{policy} target {position} held-out MAE",
            )
            call_id = policy_record.get("call_identity")
            outcome_id = policy_record.get("outcome_identity")
            if (
                not isinstance(call_id, str)
                or call_id in call_ids
                or not isinstance(outcome_id, str)
                or outcome_id in outcome_ids
            ):
                raise RuntimeError(f"Duplicate call/outcome identity at {position}")
            call_ids.add(call_id)
            outcome_ids.add(outcome_id)
            values[policy].append(value)

        delta = _finite(
            row.get("physical_C_prime_minus_B_prime"),
            f"target {position} paired effect",
        )
        if delta != values["C_prime"][-1] - values["B_prime"][-1]:
            raise RuntimeError(f"Stored paired effect mismatch at position {position}")
        deltas.append(delta)

    observed = {
        "B_prime_mean": statistics.fmean(values["B_prime"]),
        "B_prime_median": statistics.median(values["B_prime"]),
        "C_prime_mean": statistics.fmean(values["C_prime"]),
        "C_prime_median": statistics.median(values["C_prime"]),
        "delta_mean": statistics.fmean(deltas),
        "delta_median": statistics.median(deltas),
        "relative_mean_change": (
            statistics.fmean(values["C_prime"])
            - statistics.fmean(values["B_prime"])
        )
        / statistics.fmean(values["B_prime"]),
    }
    if observed != EXPECTED_SUMMARIES:
        raise RuntimeError(f"Canonical summary reproduction mismatch: {observed}")
    if (
        report.get("B_prime_heldout_physical_mae", {}).get("mean")
        != observed["B_prime_mean"]
        or report.get("B_prime_heldout_physical_mae", {}).get("median")
        != observed["B_prime_median"]
        or report.get("C_prime_heldout_physical_mae", {}).get("mean")
        != observed["C_prime_mean"]
        or report.get("C_prime_heldout_physical_mae", {}).get("median")
        != observed["C_prime_median"]
        or report.get("physical_C_prime_minus_B_prime", {}).get("mean")
        != observed["delta_mean"]
        or report.get("physical_C_prime_minus_B_prime", {}).get("median")
        != observed["delta_median"]
        or report.get("population_relative_physical_mae_change")
        != observed["relative_mean_change"]
        or report.get("counts")
        != {"B_prime_better": 45, "C_prime_better": 75, "tie_exact_float": 0}
    ):
        raise RuntimeError("Analysis report does not reproduce paired evidence")

    stored_interval = tuple(report.get("paired_mean_bootstrap_95_percentile_interval", ()))
    if stored_interval != CANONICAL_INTERVAL:
        raise RuntimeError("Stored canonical paired interval mismatch")
    return values, deltas, CANONICAL_INTERVAL


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


def _target_jitter(position: int, width: float = 0.27) -> float:
    rank = (position * 73) % TARGET_COUNT
    return -width / 2.0 + width * rank / (TARGET_COUNT - 1)


def make_figure(
    values: dict[str, list[float]],
    deltas: list[float],
    interval: tuple[float, float],
) -> mpl.figure.Figure:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))

    left = axes[0]
    positions = [1, 2]
    cells = [values[policy] for policy in POLICIES]
    boxes = left.boxplot(
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
    for policy_index, policy in enumerate(POLICIES):
        _, color = POLICY_STYLES[policy]
        box = boxes["boxes"][policy_index]
        box.set_edgecolor(color)
        box.set_linewidth(1.6)
        box.set_facecolor(to_rgba(color, 0.18))
        for artist in (
            boxes["whiskers"][2 * policy_index],
            boxes["whiskers"][2 * policy_index + 1],
            boxes["caps"][2 * policy_index],
            boxes["caps"][2 * policy_index + 1],
        ):
            artist.set_color(color)
        left.scatter(
            [positions[policy_index] + _target_jitter(i) for i in range(TARGET_COUNT)],
            values[policy],
            s=13,
            color=color,
            alpha=0.48,
            linewidths=0,
            zorder=3,
        )
    left.set_xticks(positions, [POLICY_STYLES[policy][0] for policy in POLICIES])
    left.set_ylabel("Held-out response MAE")
    left.set_title(r"$B = 40$")
    left.set_ylim(0.0, max(max(cell) for cell in cells) * 1.08)

    right = axes[1]
    right.axhline(0.0, color="#555555", linewidth=1.1, linestyle="--", zorder=1)
    right.scatter(
        [1.0 + _target_jitter(i, width=0.34) for i in range(TARGET_COUNT)],
        deltas,
        s=14,
        color=POLICY_STYLES["C_prime"][1],
        alpha=0.48,
        linewidths=0,
        zorder=3,
    )
    mean_delta = statistics.fmean(deltas)
    right.errorbar(
        [1.0],
        [mean_delta],
        yerr=[[mean_delta - interval[0]], [interval[1] - mean_delta]],
        fmt="o",
        markersize=9,
        markerfacecolor=POLICY_STYLES["C_prime"][1],
        markeredgecolor="#202020",
        markeredgewidth=1.1,
        ecolor="#202020",
        elinewidth=1.7,
        capsize=5,
        capthick=1.5,
        zorder=5,
    )
    right.set_xticks([1.0], [r"C$'$ $-$ B$'$"])
    right.set_xlim(0.55, 1.45)
    margin = 0.08 * (max(deltas) - min(deltas))
    right.set_ylim(min(deltas) - margin, max(deltas) + margin)
    right.set_ylabel(r"$\Delta$ held-out MAE (C$'$ $-$ B$'$)")
    right.set_title("Paired target effect")

    for axis in axes:
        axis.grid(axis="y", which="major", alpha=0.2, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=4)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.88, wspace=0.45)
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
    values: dict[str, list[float]],
    deltas: list[float],
    interval: tuple[float, float],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "paper120_terminal_domain_ablation_b40.pdf"
    png_path = output_dir / "paper120_terminal_domain_ablation_b40.png"
    with tempfile.TemporaryDirectory(prefix="mutaco-terminal-domain-") as temp_name:
        temp = Path(temp_name)
        first_pdf, second_pdf = temp / "first.pdf", temp / "second.pdf"
        first_png, second_png = temp / "first.png", temp / "second.png"
        for pdf, png in ((first_pdf, first_png), (second_pdf, second_png)):
            fig = make_figure(values, deltas, interval)
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
        / "paper120_terminal_domain_ablation_v1"
    )
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    values, deltas, interval = load_frozen_evidence(repo)
    print(f"B_prime_n={len(values['B_prime'])}")
    print(f"B_prime_mean={statistics.fmean(values['B_prime']):.17g}")
    print(f"B_prime_median={statistics.median(values['B_prime']):.17g}")
    print(f"C_prime_n={len(values['C_prime'])}")
    print(f"C_prime_mean={statistics.fmean(values['C_prime']):.17g}")
    print(f"C_prime_median={statistics.median(values['C_prime']):.17g}")
    print(f"paired_delta_n={len(deltas)}")
    print(f"delta_mean={statistics.fmean(deltas):.17g}")
    print(f"delta_median={statistics.median(deltas):.17g}")
    print(f"canonical_interval={list(interval)}")
    print(f"C_prime_better={sum(delta < 0.0 for delta in deltas)}")
    print(f"B_prime_better={sum(delta > 0.0 for delta in deltas)}")
    print(f"ties={sum(delta == 0.0 for delta in deltas)}")
    relative_change = (
        statistics.fmean(values["C_prime"])
        - statistics.fmean(values["B_prime"])
    ) / statistics.fmean(values["B_prime"])
    print(f"relative_mean_change_percent={100.0 * relative_change:.12f}")
    pdf_path, png_path = render_deterministically(
        values, deltas, interval, output_dir
    )
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
    _finite,
    load_frozen_evidence,
    configure_plotting,
    _target_jitter,
    make_figure,
    _save_figure,
    render_deterministically,
    main,
):
    _legacy_export.__module__ = "experiments.paper.exploration.plot_paper120_terminal_domain_ablation_v1"
del _legacy_export


def _cli() -> None:
    """Describe the offline renderer without starting a render for --help."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()


if __name__ == "__main__":
    _cli()
