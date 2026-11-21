"""Validation helpers for tracked and pending paper report sources."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

from reproducibility.recorded_paths import resolve_recorded_path
from experiments.benchmarking.artifacts import file_hash, read_json


def _repository_path(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve()
    path = resolve_recorded_path(root, relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("paper report source lies outside repository") from exc
    return path


def validate_report_source(
    specification: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    """Validate accepted evidence or preserve an explicit pending boundary."""

    source = specification["source"]
    path = _repository_path(repository_root, str(source["path"]))
    if source["state"] == "accepted_tracked_evidence":
        if not path.is_file():
            raise ValueError(f"accepted report source does not exist: {path}")
        actual = file_hash(path)
        if actual != source["sha256"]:
            raise ValueError("accepted report source hash mismatch")
        claims = _compact_claims(path)
        if claims != specification["claims"]:
            raise ValueError("accepted report claims differ from compact evidence")
        status = "accepted_source_validated"
    else:
        actual = None
        claims = None
        status = "pending_canonical_consolidation"
    return {
        "schema_version": "mutaco-paper-report-source-validation-v1",
        "report_identity": specification["report_identity"],
        "status": status,
        "source_path": path.as_posix(),
        "source_sha256": actual,
        "claims": claims,
        "scientific_calls": 0,
        "report_written": False,
    }


def _method_label(method: str) -> str:
    return {
        "mutaco": "MUTACO",
        "sbi_npe": "NPE",
        "sbi_snpe": "SNPE",
        "mutaco_standardized": "MUTACO standardized",
        "mutaco_unstandardized": "MUTACO unstandardized",
    }.get(method, method)


def render_report(specification: Mapping[str, Any], repository_root: Path) -> str:
    """Render the complete deterministic Markdown report from accepted evidence."""

    validation = validate_report_source(specification, repository_root)
    if validation["status"] != "accepted_source_validated":
        raise ValueError("report materialization requires accepted tracked evidence")
    claims = validation["claims"]
    methods = tuple(str(value) for value in specification["methods"])
    target_count = int(claims["target_count"])
    budget = int(specification["inference_budget_per_method_target"])
    calls = claims["physical_calls"]
    inference_calls = target_count * len(methods) * budget
    assessment_calls = int(calls["successful_calls"]) - inference_calls
    lines = [
        f"# {specification['report_id'].replace('_', ' ')} report",
        "",
        "**Status:** accepted canonical repository evidence",
        "",
        f"**Classification:** {specification.get('classification', 'primary paper comparison')}",
        "",
        "**Figure status:** discussion pending; no figures generated",
        "",
        "## Experiment parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Frozen targets | {target_count} |",
        f"| Methods | {', '.join(_method_label(method) for method in methods)} |",
        f"| Inference budget | {budget} calls per method and target |",
        "| Assessment | one paired held-out call per method and target |",
        "| Assessment randomness | shared target-addressed disorder, initial state, and thermal noise |",
        f"| Successful full-cohort calls | {int(calls['successful_calls']):,} |",
        "",
        "## Method results",
        "",
        "All errors are lower-is-better. Values are means over all targets; medians are shown in parentheses.",
        "",
        "| Method | Response MAE | Response L2 | Parameter L2 |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        values = claims["methods"][method]
        cells = []
        for metric in ("response_error_mae", "response_error_l2", "parameter_error_l2"):
            cells.append(f"{values[metric]['mean']:.7f} ({values[metric]['median']:.7f})")
        lines.append(f"| {_method_label(method)} | {' | '.join(cells)} |")
    lines.extend(["", "## Paired response-MAE results", ""])
    for pair, values in claims["paired_response_mae"].items():
        left, right = pair.split("__", 1)
        line = (
            f"- {_method_label(left)} minus {_method_label(right)}: mean "
            f"`{values['mean_left_minus_right']:.7f}`; "
            f"{values['left_wins']} wins, {values['ties']} ties, "
            f"{values['left_losses']} losses."
        )
        interval = values.get("paired_mean_bootstrap_95_percentile_interval")
        if interval is not None:
            line = line[:-1] + (
                f"; seeded 95% percentile bootstrap interval "
                f"`[{interval[0]:.7f}, {interval[1]:.7f}]`."
            )
        lines.append(line)
    source = specification["source"]
    manifest = read_json(_repository_path(repository_root, str(source["path"])))
    lines.extend([
        "",
        "## Call accounting",
        "",
        f"The comparison contains {int(calls['successful_calls']):,} successful physical calls: "
        f"{inference_calls:,} inference/training calls and {assessment_calls:,} paired assessments. "
        f"There were {int(calls['failed_calls'])} failed calls and {int(calls['retry_calls'])} retries. "
        "Export, transfer, consolidation, analysis, verification, and reporting add zero scientific calls.",
        "",
        "## Provenance",
        "",
        f"The accepted evidence manifest is `{source['path']}` at SHA-256 `{source['sha256']}`.",
    ])
    if manifest.get("paired_records_sha256"):
        lines.append(f"The paired-record SHA-256 is `{manifest['paired_records_sha256']}`.")
    if manifest.get("bundle_identity"):
        lines.append(f"The canonical bundle identity is `{manifest['bundle_identity']}`.")
    lines.extend([
        "",
        "## Figures pending",
        "",
        "No panels, plot types, captions, or rendering choices have been approved.",
        "",
    ])
    return "\n".join(lines)


def build_report(
    specification: Mapping[str, Any], repository_root: Path, *,
    confirm_authorized: bool, check: bool = False,
) -> dict[str, Any]:
    """Materialize or check a deterministic accepted-evidence report."""

    content = render_report(specification, repository_root)
    output = _repository_path(repository_root, str(specification["output"]))
    matches = output.is_file() and output.read_text(encoding="utf-8") == content
    if check:
        if not matches:
            raise ValueError("tracked report differs from deterministic materialization")
        written = False
        status = "report_reproduced"
    else:
        if not confirm_authorized:
            raise ValueError("report build requires explicit runtime authorization")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8", newline="\n")
        written = True
        status = "report_materialized"
    return {
        "schema_version": "mutaco-paper-report-materialization-v1",
        "status": status,
        "report_identity": specification["report_identity"],
        "output": output.as_posix(),
        "output_sha256": file_hash(output),
        "report_written": written,
        "scientific_calls": 0,
    }


def _compact_claims(manifest_path: Path) -> dict[str, Any]:
    """Recompute report claims from one compact three-method bundle."""

    manifest = read_json(manifest_path)
    records_path = manifest_path.parent / "paired_records.jsonl"
    if not records_path.is_file():
        raise ValueError("compact report source lacks paired_records.jsonl")
    if manifest.get("paired_records_sha256") != file_hash(records_path):
        raise ValueError("compact report paired-record hash mismatch")
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != manifest.get("target_count"):
        raise ValueError("compact report target count differs from manifest")
    methods = tuple(str(value) for value in manifest.get("methods", ()))
    if not methods:
        raise ValueError("compact report manifest has no methods")
    paired_summary: dict[tuple[str, str], Mapping[str, Any]] = {}
    summary_path = manifest_path.parent / "population_summary.json"
    if manifest.get("population_summary_sha256"):
        if (
            not summary_path.is_file()
            or file_hash(summary_path) != manifest["population_summary_sha256"]
        ):
            raise ValueError("compact report population-summary hash mismatch")
        summary = read_json(summary_path)
        paired_summary = {
            (str(item["left"]), str(item["right"])): item["metrics"][
                "response_error_mae"
            ]
            for item in summary.get("paired", ())
        }
    metrics = ("response_error_mae", "response_error_l2", "parameter_error_l2")
    method_claims: dict[str, Any] = {}
    for method in methods:
        values_by_metric = {
            metric: [float(row["methods"][method][metric]) for row in rows]
            for metric in metrics
        }
        method_claims[method] = {
            metric: {
                "mean": mean(values),
                "median": median(values),
            }
            for metric, values in values_by_metric.items()
        }
    paired = {}
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            differences = [
                float(row["methods"][left]["response_error_mae"])
                - float(row["methods"][right]["response_error_mae"])
                for row in rows
            ]
            pair_claim = {
                "mean_left_minus_right": mean(differences),
                "left_wins": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "left_losses": sum(value > 0 for value in differences),
            }
            if (left, right) in paired_summary:
                pair_claim["paired_mean_bootstrap_95_percentile_interval"] = list(
                    paired_summary[(left, right)][
                        "paired_mean_bootstrap_95_percentile_interval"
                    ]
                )
            paired[f"{left}__{right}"] = pair_claim
    calls = {field: 0 for field in ("attempted_calls", "successful_calls", "failed_calls", "retry_calls")}
    for row in rows:
        for method in methods:
            for category in row["methods"][method]["call_ledger"].values():
                for field in calls:
                    calls[field] += int(category[field])
    return {
        "target_count": len(rows),
        "methods": method_claims,
        "paired_response_mae": paired,
        "physical_calls": calls,
    }
