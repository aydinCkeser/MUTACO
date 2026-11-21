"""Run the bounded Prompt 5C calibration visibly from PyCharm."""

from __future__ import annotations

import argparse
from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.benchmarking.calibration import run_calibration


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments/analytic_target_matching/configs/benchmark_campaigns/prompt5c_calibration.json"
DEFAULT_OUTPUT = ROOT / "runs/benchmarks/analytic/prompt5c_calibration"


def execute(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
    device: str | None = None,
) -> dict:
    # 1. Load and strictly validate the fixed three-target calibration policy.
    # 2. Run every method in a fresh, parent-supervised process with progress.
    # 3. Persist readable evidence and measured Prompt 5D cost projections.
    return run_calibration(config_path, output=output, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args = parser.parse_args()
    execute(config_path=args.config, output=args.output, device=args.device)


if __name__ == "__main__":
    main()
