"""Presentation-only Fig. 7 renderer; no simulator or scientific calculations."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/paper/oscillator_physics_multi_target_sensitivity/outputs/paper120_maximin12_job6353452/plotting_data.json"
DATA_SHA256 = "dc02c234b792548872f684102bd4ca66646274de03e6413a919271fea00e059e"
PDF = ROOT / "docs/paper/latex/figures/fig07_local_sensitivity.pdf"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    assert sha256(DATA) == DATA_SHA256, "Frozen plotting data changed"
    data = json.loads(DATA.read_text(encoding="utf-8"))
    norms = np.asarray(data["scaled_column_norms"], dtype=float)
    targets = data["target_order"]
    assert len(targets) == 12
    assert norms.shape == (len(targets), 4)
    mpl.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage[T1]{fontenc}",
        "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
        "axes.linewidth": 1.0, "xtick.major.width": 1.0,
        "ytick.major.width": 1.0, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, axis = plt.subplots(figsize=(5.9, 4.6))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.97)
    labels = data["parameter_math_labels"]
    box_fill = "#E6E6E6"
    point_color = "#333333"
    median_color = "#111111"
    outline_color = "#4D4D4D"
    positions = np.arange(4)
    # Identical per-parameter input and boxplot statistic convention to archive.
    axis.boxplot([norms[:, i] for i in range(4)], positions=positions,
                 widths=0.48, patch_artist=True, showfliers=False,
                 medianprops={"color": median_color, "linewidth": 2.0},
                 whiskerprops={"color": outline_color, "linewidth": 1.25},
                 capprops={"color": outline_color, "linewidth": 1.25},
                 boxprops={"edgecolor": outline_color, "linewidth": 1.6,
                           "facecolor": box_fill})
    for rank, offset in enumerate(np.linspace(-0.12, 0.12, len(targets))):
        line, = axis.plot(positions + offset, norms[rank], color="#777777",
                          alpha=0.22, linewidth=0.8, zorder=1)
        points = axis.scatter(positions + offset, norms[rank], s=13,
                              color=point_color, linewidths=0, zorder=3)
        np.testing.assert_array_equal(line.get_ydata(), data["scaled_column_norms"][rank])
        np.testing.assert_array_equal(points.get_offsets()[:, 1], data["scaled_column_norms"][rank])
    axis.set_yscale("log")
    axis.set_xticks(positions, labels)
    axis.set_ylabel(r"Local sensitivity norm $\|S_{\cdot a}\|_2$")
    axis.tick_params(direction="out", length=4)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    axis.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    assert axis.get_yscale() == "log"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == labels
    assert all(spine.get_visible() for spine in axis.spines.values())
    assert all(spine.get_linewidth() == 1.0 for spine in axis.spines.values())
    fixed = datetime(1970, 1, 1, tzinfo=timezone.utc)
    fig.savefig(PDF, metadata={"Creator": "MUTACO presentation-only frozen-data plotting",
                              "CreationDate": fixed, "ModDate": fixed})
    plt.close(fig)
    assert sha256(DATA) == DATA_SHA256
    print("All 12 plotted target arrays, parameter order, and log scale checked exactly")
    print(f"PDF_SHA256={sha256(PDF)}")


# Preserve historical global/pickle references across the compatibility import.
for _legacy_export in (
    sha256,
    main,
):
    _legacy_export.__module__ = "experiments.paper.exploration.plot_paper120_local_sensitivity_v6"
del _legacy_export


def _cli() -> None:
    """Describe the offline renderer without starting a render for --help."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()


if __name__ == "__main__":
    _cli()
