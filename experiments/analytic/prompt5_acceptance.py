"""Compatibility entry point for the completed named Prompt 5 milestone.

New work should use :mod:`experiments.analytic_target_matching.run_suite`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analytic_target_matching.run_suite import (  # noqa: E402
    execute_combined_config as execute,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "analytic_target_matching" / "configs" / "prompt5_vertical_slice.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "prompt5_vertical_slice"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    arguments = parser.parse_args()
    execute(
        config_path=arguments.config,
        output=arguments.output,
        requested_device=arguments.device,
    )


if __name__ == "__main__":
    main()
