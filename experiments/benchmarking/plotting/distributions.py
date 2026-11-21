"""Matplotlib distribution rendering from persisted benchmark records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mutaco import stable_hash


def _box(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "med": summary["median"],
        "q1": summary["q1"],
        "q3": summary["q3"],
        "whislo": summary["whisker_low"],
        "whishi": summary["whisker_high"],
        "fliers": summary["outliers"],
    }


def generate_distribution_plots(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
    style: Path,
    formats: Sequence[str] = ("pdf", "svg"),
    show_points: bool = True,
) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    output.mkdir(parents=True, exist_ok=True)
    specs = (
        (
            "primary_error_boxplots",
            ("target_error_l1", "target_error_l2", "design_error_l2"),
            ("Held-out target L1", "Held-out target L2", "Design L2"),
        ),
        (
            "latent_response_boxplots",
            ("latent_error_l1", "latent_error_l2"),
            ("Latent response L1", "Latent response L2"),
        ),
    )
    written = []
    with plt.style.context(str(style)):
        for stem, metrics, labels in specs:
            available = tuple(
                (metric, label)
                for metric, label in zip(metrics, labels)
                if summary["metrics"].get(metric, {}).get("count", 0) > 0
            )
            if not available:
                continue
            figure, axes = plt.subplots(1, len(available), squeeze=False)
            for axis, (metric, label) in zip(axes[0], available):
                axis.bxp([_box(summary["metrics"][metric], label)], showfliers=True)
                if show_points:
                    values = [
                        row[metric]
                        for row in completed
                        if row.get(metric) is not None
                    ]
                    offsets = [
                        ((index % 9) - 4) * 0.012 for index in range(len(values))
                    ]
                    axis.scatter(
                        [1.0 + offset for offset in offsets],
                        values,
                        s=9,
                        alpha=0.45,
                        color="#2368a2",
                        zorder=3,
                    )
                axis.set_title(label)
                axis.set_ylabel(
                    "physical parameter error"
                    if metric == "design_error_l2"
                    else "physical response error"
                )
                axis.text(
                    0.5,
                    -0.13,
                    f"n={summary['completed_count']}/{summary['target_count']}; "
                    f"failed={summary['failed_count']}",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                )
            figure.tight_layout()
            for fmt in formats:
                path = output / f"{stem}.{fmt}"
                figure.savefig(path, format=fmt)
                written.append(path.as_posix())
            plt.close(figure)
    return {
        "schema_version": "mutaco-benchmark-distribution-plot-v1",
        "style_path": style.as_posix(),
        "formats": tuple(formats),
        "files": written,
        "plot_identity": stable_hash(
            "plot-v1",
            {
                "summary": summary,
                "style": style.read_text(encoding="utf-8"),
                "formats": tuple(formats),
            },
        ),
        "matplotlib_version": matplotlib.__version__,
    }
