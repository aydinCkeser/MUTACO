"""Render graph-forward PDF/SVG figures from public result data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_forward_plots(result: dict[str, Any], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    response = np.asarray(result["response"], dtype=float)
    forces = np.asarray(result["configuration"]["forces"], dtype=float)
    half = response.size // 2
    omega = response[:half].reshape(len(forces), 2)
    synchronization = response[half:].reshape(len(forces), 2)
    written: list[Path] = []
    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        for gauge_index, label in enumerate(("gauge 0", "gauge π")):
            axes[0].plot(forces, omega[:, gauge_index], marker="o", label=label)
            axes[1].plot(forces, synchronization[:, gauge_index], marker="o", label=label)
        axes[0].set(title="Mean angular response", xlabel="Force", ylabel="ω")
        axes[1].set(title="Synchronization magnitude", xlabel="Force", ylabel="|Z|", ylim=(0, 1.05))
        for axis in axes:
            axis.legend()
        figure.suptitle("Approved overdamped scientific graph forward")
        figure.tight_layout()
        for suffix in ("pdf", "svg"):
            path = output / f"forward_response.{suffix}"
            figure.savefig(path)
            written.append(path)
        plt.close(figure)

        sensitivity = result["burn_in_sensitivity"]
        figure, axis = plt.subplots(figsize=(6.2, 3.8))
        labels = [f"{item['burn_fraction']:.0%}" for item in sensitivity]
        relative = [item["relative_l2_from_10_percent"] for item in sensitivity]
        axis.bar(labels, relative, color="#2d6a9f")
        axis.set(
            title="Fixed-length burn-in sensitivity",
            xlabel="Discarded trajectory fraction",
            ylabel="Relative L2 vs approved 10%",
        )
        figure.tight_layout()
        for suffix in ("pdf", "svg"):
            path = output / f"burn_in_sensitivity.{suffix}"
            figure.savefig(path)
            written.append(path)
        plt.close(figure)
    return written
