"""Reproduce the frozen Paper-120 MUTACO--NLE--NRE comparison."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mutaco import canonical_json_bytes, stable_hash
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.common.validation import PaperContractError


ROOT = Path(__file__).resolve().parents[5]
SPEC_PATH = Path(
    "experiments/paper/studies/graph_paper120/analysis/sbi_zoo_comparison_v1.json"
)
OUTPUT_PATH = Path("results/paper/sbi_zoo_comparison_v1")
FIGURE_PATH = Path("docs/paper/figures/sbi_zoo_comparison_v1")
SPEC_SCHEMA = "mutaco-paper120-sbi-zoo-analysis-spec-v1"
PACKAGE_SCHEMA = "mutaco-paper120-sbi-zoo-analysis-package-v1"
METHODS = ("mutaco", "nle", "nre")
BUDGETS = (40, 100)
EXPECTED_GRID_SIZE = 720
COLORS = {"mutaco": "#1f4e79", "nle": "#d97706", "nre": "#2f855a"}
LABELS = {"mutaco": "MUTACO", "nle": "NLE", "nre": "NRE"}


def _fail(message: str) -> None:
    raise PaperContractError(message)


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or file_hash(path) != expected:
        _fail(f"frozen input SHA-256 mismatch: {path}")


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 120 or not all(math.isfinite(value) for value in values):
        _fail("every summary requires 120 finite Paper-120 values")
    return {
        "n": 120,
        "mean": mean(values),
        "standard_deviation": stdev(values),
        "minimum": min(values),
        "q05": _quantile(values, 0.05),
        "q25": _quantile(values, 0.25),
        "median": median(values),
        "q75": _quantile(values, 0.75),
        "q95": _quantile(values, 0.95),
        "maximum": max(values),
    }


def _read_spec(root: Path) -> dict[str, Any]:
    path = root / SPEC_PATH
    document = read_json(path)
    body = {key: value for key, value in document.items() if key != "analysis_spec_identity"}
    if (
        document.get("schema_version") != SPEC_SCHEMA
        or document.get("analysis_spec_identity")
        != stable_hash("paper120-sbi-zoo-analysis-spec-v1", body)
        or tuple(document.get("methods", ())) != METHODS
        or tuple(document.get("budgets", ())) != BUDGETS
        or document.get("primary_metric") != "response_error_mae"
        or document.get("paired_difference")
        != "competitor_response_error_mae_minus_mutaco_response_error_mae"
        or document.get("cross_budget_relation")
        != "independent_campaigns_same_ordered_target_cohort_not_nested"
    ):
        _fail("SBI-zoo analysis specification is not canonical")
    _require_hash(resolve_recorded_path(root, document["cohort"]["path"]), document["cohort"]["sha256"])
    return document


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


def _raw_mutaco_path(archive: Path, position: int, budget: int) -> Path:
    return (
        archive
        / "workers"
        / f"worker-{position:03d}"
        / "attempts"
        / f"b{budget}"
        / f"target-{position:03d}"
        / "mutaco.json"
    )


def _load_mutaco(root: Path, specification: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = specification["mutaco_source"]
    manifest_path = resolve_recorded_path(root, source["manifest"])
    _require_hash(manifest_path, source["manifest_sha256"])
    manifest = read_json(manifest_path)
    if (
        manifest.get("evidence_identity") != source["evidence_identity"]
        or manifest.get("collection_identity") != source["collection_identity"]
        or manifest.get("methods") != ["mutaco", "sbi_npe", "sbi_snpe"]
        or manifest.get("call_accounting", {}).get("successful_calls") != 720
        or manifest.get("call_accounting", {}).get("failed_calls") != 0
        or manifest.get("call_accounting", {}).get("retry_calls") != 0
    ):
        _fail("authoritative repaired MUTACO assessment manifest is invalid")

    archive = resolve_recorded_path(root, source["raw_archive"])
    result: list[dict[str, Any]] = []
    for budget in BUDGETS:
        row_spec = source["analysis_rows"][str(budget)]
        compact_path = resolve_recorded_path(root, row_spec["path"])
        _require_hash(compact_path, row_spec["sha256"])
        compact = [row for row in _read_jsonl(compact_path) if row.get("method") == "mutaco"]
        compact.sort(key=lambda row: int(row["target_position"]))
        if [int(row["target_position"]) for row in compact] != list(range(120)):
            _fail(f"MUTACO B{budget} does not contain the ordered Paper-120 cohort")
        for row in compact:
            position = int(row["target_position"])
            raw_path = _raw_mutaco_path(archive, position, budget)
            sidecar = raw_path.with_suffix(".sha256")
            if not sidecar.is_file():
                _fail(f"MUTACO raw result sidecar is missing: {raw_path}")
            _require_hash(raw_path, sidecar.read_text(encoding="ascii").strip())
            raw = read_json(raw_path)
            compact_mae = float(
                row["errors"]["heldout_assessment_error"]["metric"]["component_mean_mae"]
            )
            if (
                raw.get("status") != "completed"
                or raw.get("method") != "mutaco"
                or raw.get("budget") != budget
                or raw.get("target_position") != position
                or raw.get("target_record_id") != row.get("target_id")
                or raw.get("assessment_context") != row["assessment"]["context"]
                or float(raw.get("response_error_mae")) != compact_mae
                or raw.get("call_accounting", {}).get("assessment_successful_calls") != 1
                or raw.get("call_accounting", {}).get("retry_calls") != 0
            ):
                _fail("raw and compact authoritative MUTACO evidence disagree")
            result.append(
                {
                    "target_position": position,
                    "target_id": str(raw["target_record_id"]),
                    "budget": budget,
                    "method": "mutaco",
                    "response_error_mae": compact_mae,
                    "parameter_error_l2": float(raw["parameter_error_l2"]),
                    "assessment_context": raw["assessment_context"],
                    "recommendation_identity": str(raw["recommendation_identity"]),
                    "result_identity": str(raw["result_identity"]),
                    "source": "repaired_mutaco_sbi_assessment_6251303",
                }
            )
    return result


def _load_p3(root: Path, specification: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = specification["p3_source"]
    path = resolve_recorded_path(root, source["manifest"])
    _require_hash(path, source["manifest_sha256"])
    document = read_json(path)
    if (
        document.get("status") != "PASS"
        or document.get("collection_identity") != source["collection_identity"]
        or document.get("p3_declaration_identity") != source["declaration_identity"]
        or document.get("p3_execution_source") != source["execution_source"]
        or document.get("record_count") != 480
        or document.get("errors") != []
        or document.get("physical_calls")
        != {
            "maximum_calls": 480,
            "attempted_calls": 480,
            "successful_calls": 480,
            "failed_calls": 0,
            "attempted_outcome_unknown": 0,
            "retry_calls": 0,
        }
    ):
        _fail("authoritative P3 collection is invalid")
    mapping = specification["p3_method_mapping"]
    result = []
    for record in document.get("records", ()):
        method = mapping.get(record.get("method"))
        values = (record.get("response_error_mae"), record.get("parameter_error_l2"))
        if (
            method not in {"nle", "nre"}
            or record.get("budget") not in BUDGETS
            or record.get("target_position") not in range(120)
            or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
        ):
            _fail("P3 result record is malformed")
        result.append(
            {
                "target_position": int(record["target_position"]),
                "target_id": str(record["target_record_id"]),
                "budget": int(record["budget"]),
                "method": method,
                "response_error_mae": float(record["response_error_mae"]),
                "parameter_error_l2": float(record["parameter_error_l2"]),
                "assessment_context": record["assessment_context"],
                "recommendation_identity": str(record["recommendation_identity"]),
                "result_identity": str(record["result_identity"]),
                "source": "sbi_zoo_p3_6265116",
            }
        )
    return result


def load_rows(root: Path, specification: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = _load_mutaco(root, specification) + _load_p3(root, specification)
    rows.sort(key=lambda row: (row["target_position"], row["budget"], METHODS.index(row["method"])))
    index = {(row["target_position"], row["budget"], row["method"]): row for row in rows}
    expected = {(position, budget, method) for position in range(120) for budget in BUDGETS for method in METHODS}
    if len(rows) != EXPECTED_GRID_SIZE or len(index) != EXPECTED_GRID_SIZE or set(index) != expected:
        _fail("analysis requires the exact 120 by 2 by 3 evidence grid")
    cohort = read_json(resolve_recorded_path(root, specification["cohort"]["path"]))["records"]
    for position in range(120):
        expected_id = cohort[str(position)]["target_record_id"]
        for budget in BUDGETS:
            group = [index[position, budget, method] for method in METHODS]
            if any(row["target_id"] != expected_id for row in group):
                _fail("target identity/order differs across methods")
            if any(row["assessment_context"] != group[0]["assessment_context"] for row in group[1:]):
                _fail("methods do not share the exact repaired held-out context")
    return tuple(rows)


def _group(rows: Iterable[Mapping[str, Any]], method: str, budget: int, field: str) -> list[float]:
    values = [float(row[field]) for row in rows if row["method"] == method and row["budget"] == budget]
    if len(values) != 120:
        _fail("analysis group is incomplete")
    return values


def analyze(rows: Sequence[Mapping[str, Any]], specification: Mapping[str, Any]) -> dict[str, Any]:
    index = {(row["target_position"], row["budget"], row["method"]): row for row in rows}
    response = {
        str(budget): {method: _summary(_group(rows, method, budget, "response_error_mae")) for method in METHODS}
        for budget in BUDGETS
    }
    parameter = {
        str(budget): {method: _summary(_group(rows, method, budget, "parameter_error_l2")) for method in METHODS}
        for budget in BUDGETS
    }
    paired: dict[str, dict[str, Any]] = {}
    for budget in BUDGETS:
        paired[str(budget)] = {}
        for method in ("nle", "nre"):
            differences = [
                index[position, budget, method]["response_error_mae"]
                - index[position, budget, "mutaco"]["response_error_mae"]
                for position in range(120)
            ]
            paired[str(budget)][method] = {
                "difference": "competitor_minus_mutaco",
                "summary": _summary(differences),
                "competitor_beats_mutaco": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "mutaco_beats_competitor": sum(value > 0 for value in differences),
                "fraction_competitor_beats_mutaco": sum(value < 0 for value in differences) / 120,
            }
    budget_change: dict[str, Any] = {}
    for method in METHODS:
        differences = [
            index[position, 100, method]["response_error_mae"]
            - index[position, 40, method]["response_error_mae"]
            for position in range(120)
        ]
        budget_change[method] = {
            "difference": "b100_minus_b40",
            "campaign_relation": specification["cross_budget_relation"],
            "summary": _summary(differences),
            "improved_at_b100": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "worse_at_b100": sum(value > 0 for value in differences),
            "fraction_improved_at_b100": sum(value < 0 for value in differences) / 120,
        }
    winners = {}
    for budget in BUDGETS:
        counts = {method: 0 for method in METHODS}
        ties = 0
        for position in range(120):
            values = {method: index[position, budget, method]["response_error_mae"] for method in METHODS}
            best = min(values.values())
            selected = [method for method, value in values.items() if value == best]
            if len(selected) == 1:
                counts[selected[0]] += 1
            else:
                ties += 1
        winners[str(budget)] = {"counts": counts, "ties": ties}
    outliers = []
    for budget in BUDGETS:
        for method in METHODS:
            group = [row for row in rows if row["budget"] == budget and row["method"] == method]
            ordered = sorted(group, key=lambda row: row["response_error_mae"], reverse=True)
            q25 = _quantile([row["response_error_mae"] for row in group], 0.25)
            q75 = _quantile([row["response_error_mae"] for row in group], 0.75)
            fence = q75 + 1.5 * (q75 - q25)
            top = {row["target_position"] for row in ordered[:5]}
            for row in ordered:
                if row["response_error_mae"] > fence or row["target_position"] in top:
                    outliers.append(
                        {
                            "budget": budget,
                            "method": method,
                            "target_position": row["target_position"],
                            "target_id": row["target_id"],
                            "response_error_mae": row["response_error_mae"],
                            "parameter_error_l2": row["parameter_error_l2"],
                            "above_tukey_upper_fence": row["response_error_mae"] > fence,
                            "top_five": row["target_position"] in top,
                        }
                    )
    body = {
        "schema_version": "mutaco-paper120-sbi-zoo-analysis-summary-v1",
        "analysis_spec_identity": specification["analysis_spec_identity"],
        "primary_metric": "response_error_mae",
        "target_count": 120,
        "row_count": EXPECTED_GRID_SIZE,
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "response_recovery": response,
        "parameter_recovery_separate": parameter,
        "paired_vs_mutaco": paired,
        "cross_budget_change": budget_change,
        "target_winners": winners,
        "outliers": outliers,
        "failed_evaluations": 0,
        "scientific_calls_added_by_analysis": 0,
    }
    return {**body, "analysis_identity": stable_hash("paper120-sbi-zoo-analysis-summary-v1", body)}


def _write_csv(path: Path, fieldnames: Sequence[str], records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _save_figure(figure: Any, base: Path) -> None:
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(base.with_suffix(".png"), bbox_inches="tight", metadata={"Software": "MUTACO canonical analysis"})
    plt.close(figure)


def _distribution_plot(rows: Sequence[Mapping[str, Any]], field: str, ylabel: str, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    for axis, budget in zip(axes, BUDGETS):
        values = [_group(rows, method, budget, field) for method in METHODS]
        boxes = axis.boxplot(values, tick_labels=[LABELS[m] for m in METHODS], widths=0.55, showfliers=False, patch_artist=True)
        for patch, method in zip(boxes["boxes"], METHODS):
            patch.set(facecolor=COLORS[method], alpha=0.28, edgecolor=COLORS[method])
        for method_index, (method, group) in enumerate(zip(METHODS, values), start=1):
            jitter = np.asarray([((position * 37) % 101 - 50) / 500.0 for position in range(120)])
            axis.scatter(np.full(120, method_index) + jitter, group, s=7, alpha=0.35, color=COLORS[method], linewidths=0)
        axis.set_yscale("log")
        axis.set_title(f"B{budget}")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(ylabel)
    figure.tight_layout()
    _save_figure(figure, output)


def _paired_plot(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    index = {(row["target_position"], row["budget"], row["method"]): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    for axis, budget in zip(axes, BUDGETS):
        values = [[index[p, budget, method]["response_error_mae"] - index[p, budget, "mutaco"]["response_error_mae"] for p in range(120)] for method in ("nle", "nre")]
        boxes = axis.boxplot(values, tick_labels=["NLE − MUTACO", "NRE − MUTACO"], widths=0.55, showfliers=False, patch_artist=True)
        for patch, method in zip(boxes["boxes"], ("nle", "nre")):
            patch.set(facecolor=COLORS[method], alpha=0.28, edgecolor=COLORS[method])
        for method_index, (method, group) in enumerate(zip(("nle", "nre"), values), start=1):
            jitter = np.asarray([((position * 37) % 101 - 50) / 500.0 for position in range(120)])
            axis.scatter(np.full(120, method_index) + jitter, group, s=7, alpha=0.35, color=COLORS[method], linewidths=0)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_yscale("symlog", linthresh=0.01)
        axis.set_title(f"B{budget}")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Competitor MAE − MUTACO MAE")
    figure.tight_layout()
    _save_figure(figure, output)


def _budget_plot(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    index = {(row["target_position"], row["budget"], row["method"]): row for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharex=True, sharey=True)
    limits = [row["response_error_mae"] for row in rows]
    lower, upper = min(limits) * 0.8, max(limits) * 1.25
    for axis, method in zip(axes, METHODS):
        b40 = [index[p, 40, method]["response_error_mae"] for p in range(120)]
        b100 = [index[p, 100, method]["response_error_mae"] for p in range(120)]
        axis.scatter(b40, b100, s=10, alpha=0.52, color=COLORS[method], linewidths=0)
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=0.8, linestyle="--")
        axis.set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper), title=LABELS[method])
        axis.grid(alpha=0.2)
        axis.set_xlabel("B40 response MAE")
    axes[0].set_ylabel("B100 response MAE")
    figure.tight_layout()
    _save_figure(figure, output)


def _text_artifacts(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output: Path) -> None:
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    row_fields = ["target_position", "target_id", "budget", "method", "response_error_mae", "parameter_error_l2", "recommendation_identity", "result_identity", "source"]
    _write_csv(output / "rows.csv", row_fields, ({key: row[key] for key in row_fields} for row in rows))
    summary_rows = []
    for endpoint, key in (("response", "response_recovery"), ("parameter", "parameter_recovery_separate")):
        for budget in BUDGETS:
            for method in METHODS:
                summary_rows.append({"endpoint": endpoint, "budget": budget, "method": method, **summary[key][str(budget)][method]})
    _write_csv(output / "summary.csv", list(summary_rows[0]), summary_rows)
    index = {(row["target_position"], row["budget"], row["method"]): row for row in rows}
    paired_rows = []
    for budget in BUDGETS:
        for position in range(120):
            for method in ("nle", "nre"):
                difference = index[position, budget, method]["response_error_mae"] - index[position, budget, "mutaco"]["response_error_mae"]
                paired_rows.append({"target_position": position, "target_id": index[position, budget, method]["target_id"], "budget": budget, "competitor": method, "competitor_minus_mutaco_response_error_mae": difference, "competitor_beats_mutaco": difference < 0})
    _write_csv(output / "paired_vs_mutaco.csv", list(paired_rows[0]), paired_rows)
    budget_rows = []
    for position in range(120):
        for method in METHODS:
            difference = index[position, 100, method]["response_error_mae"] - index[position, 40, method]["response_error_mae"]
            budget_rows.append({"target_position": position, "target_id": index[position, 40, method]["target_id"], "method": method, "b40_response_error_mae": index[position, 40, method]["response_error_mae"], "b100_response_error_mae": index[position, 100, method]["response_error_mae"], "b100_minus_b40_response_error_mae": difference, "improved_at_b100": difference < 0})
    _write_csv(output / "budget_change.csv", list(budget_rows[0]), budget_rows)
    _write_csv(output / "outliers.csv", list(summary["outliers"][0]), summary["outliers"])


def build(*, repository_root: Path = ROOT, check: bool = False) -> dict[str, Any]:
    root = repository_root.resolve()
    specification = _read_spec(root)
    rows = load_rows(root, specification)
    summary = analyze(rows, specification)
    output, figures = root / OUTPUT_PATH, root / FIGURE_PATH
    if check:
        manifest = read_json(output / "manifest.json")
        manifest_body = {key: value for key, value in manifest.items() if key != "package_identity"}
        if (
            manifest.get("package_identity")
            != stable_hash("paper120-sbi-zoo-analysis-package-v1", manifest_body)
            or manifest.get("analysis_identity") != summary["analysis_identity"]
            or (output / "summary.json").read_bytes() != canonical_json_bytes(summary) + b"\n"
        ):
            _fail("tracked analysis summary does not reproduce")
        for relative, expected in manifest.get("artifacts", {}).items():
            _require_hash(resolve_recorded_path(root, relative), expected)
        return {"status": "reproduced", "package_identity": manifest["package_identity"], "scientific_calls": 0}
    if output.exists() or figures.exists():
        _fail("analysis output paths already exist; use --check to reproduce")
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    _text_artifacts(rows, summary, output)
    _configure_plots()
    _distribution_plot(rows, "response_error_mae", "Held-out response MAE", figures / "response_error_distributions")
    _paired_plot(rows, figures / "paired_differences_vs_mutaco")
    _budget_plot(rows, figures / "b40_to_b100_response_error")
    _distribution_plot(rows, "parameter_error_l2", "Parameter error (L2)", figures / "parameter_error_distributions")
    artifact_paths = sorted([path for path in output.iterdir() if path.name != "manifest.json"] + list(figures.iterdir()))
    artifacts = {str(path.relative_to(root)).replace("\\", "/"): file_hash(path) for path in artifact_paths}
    body = {
        "schema_version": PACKAGE_SCHEMA,
        "status": "complete_frozen_evidence_analysis",
        "analysis_spec_identity": specification["analysis_spec_identity"],
        "analysis_identity": summary["analysis_identity"],
        "source_identities": {
            "mutaco": specification["mutaco_source"]["collection_identity"],
            "p3": specification["p3_source"]["collection_identity"],
        },
        "source_sha256": {
            "mutaco_manifest": specification["mutaco_source"]["manifest_sha256"],
            "p3_manifest": specification["p3_source"]["manifest_sha256"],
        },
        "target_count": 120,
        "row_count": EXPECTED_GRID_SIZE,
        "artifacts": artifacts,
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    manifest = {**body, "package_identity": stable_hash("paper120-sbi-zoo-analysis-package-v1", body)}
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return {"status": "materialized", "package_identity": manifest["package_identity"], "analysis_identity": summary["analysis_identity"], "output": str(OUTPUT_PATH), "figures": str(FIGURE_PATH), "scientific_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(build(repository_root=arguments.repository_root, check=arguments.check), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
