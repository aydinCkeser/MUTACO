"""Read, select, summarize, export and plot stored campaign rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .artifacts import read_json, write_csv, write_pretty_json
from .plotting import generate_distribution_plots
from .statistics import METRIC_FIELDS, aggregate_rows


ROW_FIELDS = (
    "target_index",
    "method",
    "status",
    "target_error_l1",
    "target_error_l2",
    "design_error_l2",
    "latent_error_l1",
    "latent_error_l2",
    "runtime_seconds",
    "failure_explanation",
)


@dataclass(frozen=True)
class BenchmarkResultSet:
    """Documented iterable view over persisted benchmark rows."""

    root: Path
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def select(
        self,
        *,
        targets: Sequence[int] | None = None,
        methods: Sequence[str] | None = None,
        views: Sequence[str] | None = None,
        regimes: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> "BenchmarkResultSet":
        filters = {
            "target_index": set(targets) if targets is not None else None,
            "method": set(methods) if methods is not None else None,
            "view": set(views) if views is not None else None,
            "regime": set(regimes) if regimes is not None else None,
            "status": set(statuses) if statuses is not None else None,
        }
        selected = tuple(
            row
            for row in self.rows
            if all(allowed is None or row.get(field) in allowed for field, allowed in filters.items())
        )
        return BenchmarkResultSet(self.root, selected, self.manifest)


def load_results(path: str | Path) -> BenchmarkResultSet:
    root = Path(path).resolve()
    manifest = read_json(root / "manifest.json")
    rows = read_json(root / "metrics.json")["rows"]
    if not isinstance(rows, list):
        raise ValueError("metrics rows must be a list")
    return BenchmarkResultSet(root, tuple(rows), manifest)


def summarize_results(
    results: BenchmarkResultSet | Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = METRIC_FIELDS,
    group_by: Sequence[str] = ("method",),
) -> dict[str, Any]:
    rows = results.rows if isinstance(results, BenchmarkResultSet) else tuple(results)
    return aggregate_rows(rows, metrics=metrics, group_by=group_by)


def export_tables(
    results: BenchmarkResultSet,
    summary: Mapping[str, Any],
    *,
    output: str | Path,
) -> tuple[Path, ...]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = destination / "records.csv"
    extra_fields = tuple(
        field for field in (
            "regime", "budget", "source_regime", "master_target_identity",
            "target_construction_calls", "inference_calls", "assessment_calls",
            "gp_fits", "posterior_samples", "derived_zero_call_view",
        )
        if any(field in row for row in results.rows)
    )
    write_csv(metrics, results.rows, ROW_FIELDS + extra_fields)
    groups = destination / "summary_groups.json"
    write_pretty_json(groups, summary)
    return metrics, groups


def plot_results(
    results: BenchmarkResultSet,
    summary: Mapping[str, Any],
    *,
    output: str | Path,
    style: str | Path,
    formats: Sequence[str] = ("pdf", "svg"),
) -> dict[str, Any]:
    groups = summary.get("groups", [])
    if len(groups) != 1:
        raise ValueError("current distribution plot requires one selected summary group")
    legacy_summary = groups[0]
    return generate_distribution_plots(
        summary=legacy_summary,
        rows=list(results.rows),
        output=Path(output),
        style=Path(style),
        formats=tuple(formats),
    )

# Preserve the historical serialization address of public views and functions.
for _view in (BenchmarkResultSet, load_results, summarize_results, export_tables, plot_results):
    _view.__module__ = "experiments.benchmarking.api"
del _view
