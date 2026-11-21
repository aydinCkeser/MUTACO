r"""Plot one Paper-120 B40 target and its MUTACO/NPE reconstruction.

Run from the repository root with the pinned local environment::

    .\infrastructure\local\run_mutaco.ps1 -- experiments/paper/exploration/plot_paper120_target.py --target 0

The frozen target curve is read from disk.  Reconstructing MUTACO and NPE
uses exactly two local forward calls at their stored recommendations, with the
same paired disorder/thermal-noise coordinate used by the B40 study.  There is
no inference or training here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:  # Support the simple ``python path/to/script.py`` form.
    sys.path.insert(0, str(ROOT))

# Paper-120 was evaluated with eight BLAS/Torch worker threads.  Set these
# before importing Torch when this file is run as a script (including PyCharm).
PAPER120_INTRAOP_THREADS = 8
PAPER120_INTEROP_THREADS = 1
if __name__ == "__main__":
    for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_name] = str(PAPER120_INTRAOP_THREADS)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.graph_inversion.paper_v3_callbacks import GraphPopulationCallbacks
from experiments.paper.protocol.population_v3 import load_population_declaration_v3
from experiments.paper.protocol.protocol_v3 import paired_assessment_context

# ---------------------------------------------------------------------------
# Easy-to-edit choices.  Command-line --target/--gauge/--output override these.
# ---------------------------------------------------------------------------
TARGET_INDEX = 0                 # Any integer from 0 through 119.
DISPLAY_GAUGE = 0                # 0 displays Phi=0; 1 displays Phi=pi.
FORCES = (0.0, 1.25, 2.5, 3.75, 5.0)
FIGURE_SIZE = (9.0, 3.8)
DPI = 180
FORMATS = ("png", "svg", "pdf")
OUTPUT_ROOT = ROOT / "experiments/paper/exploration/outputs"
REPLAY_L1_TOLERANCE = 1e-3     # Allows harmless float32 runtime drift only.

COLORS = {"target": "black", "mutaco": "#0072B2", "sbi_npe": "#E69F00"}
MARKERS = {"target": "o", "mutaco": "s", "sbi_npe": "^"}
LABELS = {"target": "Target", "mutaco": "MUTACO", "sbi_npe": "NPE"}

# Canonical B40 inputs.  Change these only when intentionally plotting another
# accepted campaign, because they determine both recommendations and randomness.
COHORT = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3"
RECORDS = ROOT / "results/paper/graph_paper120/paper120/paired_records.jsonl"
DECLARATION = ROOT / "results/paper/graph_paper120/paper120/executed_declaration.json"
METHOD_CONFIGS = {
    "mutaco": ROOT / "results/paper/graph_paper120/paper120/executed_mutaco_configuration.json",
    "sbi_npe": ROOT / "results/paper/graph_paper120/paper120/executed_npe_configuration.json",
}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_target_and_methods(target_index: int) -> tuple[dict, dict]:
    """Load and identity-check one target plus its two B40 recommendations."""

    if not 0 <= target_index < 120:
        raise ValueError("target index must be between 0 and 119")
    target = _read_json(COHORT / f"target_{target_index:03d}.json")
    row = None
    with RECORDS.open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate["target_position"] == target_index:
                row = candidate
                break
    if row is None or row["target_record_identity"] != target["target_record_id"]:
        raise ValueError("B40 result and frozen target identities do not match")
    methods = row["methods"]
    if set(("mutaco", "sbi_npe")) - methods.keys():
        raise ValueError("selected target lacks completed MUTACO/NPE records")
    return target, methods


def _configure_historical_threads(target: Mapping) -> None:
    """Reproduce the frozen Paper-120 CPU reduction layout."""

    provenance = target.get("execution_provenance", {})
    intraop = provenance.get("torch_intraop_threads")
    interop = provenance.get("torch_interop_threads")
    if (intraop, interop) != (PAPER120_INTRAOP_THREADS, PAPER120_INTEROP_THREADS):
        raise ValueError(f"unexpected frozen target thread layout: {intraop}/{interop}")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(intraop)
    torch.set_num_threads(intraop)
    torch.set_num_interop_threads(interop)


def reconstruct(
    target: Mapping,
    methods: Mapping,
    *,
    evaluate: Callable[[str, Sequence[float], Mapping], Sequence[float]] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Evaluate both recommendations at one shared historical coordinate."""

    declaration = load_population_declaration_v3(_read_json(DECLARATION))
    context = paired_assessment_context(declaration.comparison, target["target_record_id"])
    if evaluate is None:
        _configure_historical_threads(target)
        callbacks = GraphPopulationCallbacks(
            method_configurations=METHOD_CONFIGS,
            output=OUTPUT_ROOT,  # evaluate_assessment itself writes nothing.
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        def evaluate(method: str, theta: Sequence[float], shared: Mapping) -> Sequence[float]:
            del method  # Both methods intentionally use the identical simulator/context.
            return callbacks.evaluate_assessment(target, theta, shared).response

    responses = {"target": tuple(float(x) for x in target["target_response"])}
    for method in ("mutaco", "sbi_npe"):
        response = tuple(float(x) for x in evaluate(method, methods[method]["recommended_theta"], context))
        if len(response) != 20:
            raise ValueError(f"{method} reconstruction must contain 20 response values")
        observed_l1 = sum(abs(a - b) for a, b in zip(response, responses["target"], strict=True))
        expected_l1 = float(methods[method]["response_error_l1_sum"])
        drift = abs(observed_l1 - expected_l1)
        if drift > REPLAY_L1_TOLERANCE:
            raise ValueError(
                f"{method} replay differs from stored B40 evidence: "
                f"L1={observed_l1:.8g}, expected={expected_l1:.8g}, drift={drift:.3g}"
            )
        print(
            f"{LABELS[method]} replay check: local L1={observed_l1:.6f}, "
            f"stored L1={expected_l1:.6f}, drift={drift:.2g} (accepted)"
        )
        responses[method] = response
    return responses


def _observable(response: Sequence[float], observable: str, gauge: int) -> list[float]:
    """Extract one five-force curve from [omega(10), synchronization(10)]."""

    start = 0 if observable == "omega" else 10
    return [float(response[start + 2 * force + gauge]) for force in range(5)]


def _axes(title: str) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    figure.suptitle(title)
    for axis, ylabel in zip(axes, (r"$\langle\bar{\omega}\rangle$", r"$\langle M\rangle$"), strict=True):
        axis.set(xlabel=r"Force $F$", ylabel=ylabel, xticks=FORCES)
        axis.grid(alpha=0.25)
    axes[1].set_ylim(0.0, 1.05)
    return figure, axes


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        options = {"dpi": DPI} if extension == "png" else {}
        if extension == "pdf":
            options["metadata"] = {"CreationDate": None, "ModDate": None}
        elif extension == "svg":
            options["metadata"] = {"Date": None}
        figure.savefig(output / f"{stem}.{extension}", **options)
    plt.close(figure)


def plot_target(target: Mapping, gauge: int, output: Path) -> None:
    """Plot the selected frozen target response by itself."""

    figure, axes = _axes(f"Paper-120 target {target['target_index']} (gauge {gauge})")
    for axis, observable in zip(axes, ("omega", "synchronization"), strict=True):
        axis.plot(
            FORCES,
            _observable(target["target_response"], observable, gauge),
            color=COLORS["target"], marker=MARKERS["target"], linewidth=1.8,
        )
    _save(figure, output, "forward")


def plot_reconstruction(
    target: Mapping,
    methods: Mapping,
    responses: Mapping[str, Sequence[float]],
    gauge: int,
    output: Path,
) -> None:
    """Overlay the target and the paired MUTACO/NPE held-out responses."""

    figure, axes = _axes(f"Paper-120 target {target['target_index']} reconstruction (gauge {gauge})")
    for axis, observable in zip(axes, ("omega", "synchronization"), strict=True):
        for curve in ("target", "mutaco", "sbi_npe"):
            axis.plot(
                FORCES,
                _observable(responses[curve], observable, gauge),
                label=LABELS[curve], color=COLORS[curve], marker=MARKERS[curve],
                linewidth=1.8 if curve == "target" else 1.4,
                linestyle="-" if curve == "target" else "--",
            )
    axes[0].legend()
    _save(figure, output, "reconstruction")

    # A small text summary keeps the plotted recommendations easy to inspect.
    for method in ("mutaco", "sbi_npe"):
        theta = ", ".join(f"{value:.4f}" for value in methods[method]["recommended_theta"])
        print(f"{LABELS[method]} theta=({theta}), stored L1={methods[method]['response_error_l1_sum']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET_INDEX, help="Paper-120 target index (0..119)")
    parser.add_argument("--gauge", type=int, choices=(0, 1), default=DISPLAY_GAUGE)
    parser.add_argument("--output", type=Path, help="default: exploration/outputs/target_NNN")
    args = parser.parse_args()

    target, methods = load_target_and_methods(args.target)
    output = args.output or OUTPUT_ROOT / f"target_{args.target:03d}"
    responses = reconstruct(target, methods)  # Exactly two local physical calls.
    plot_target(target, args.gauge, output)
    plot_reconstruction(target, methods, responses, args.gauge, output)
    print(f"Wrote forward and reconstruction figures to {output.resolve()}")


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.exploration.plot_paper120_target")


if __name__ == "__main__":
    main()
