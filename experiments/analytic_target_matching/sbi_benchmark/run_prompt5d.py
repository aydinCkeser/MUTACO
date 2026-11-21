"""Run the frozen Prompt 5D campaign with visible progress from PyCharm."""

from __future__ import annotations

import argparse
from pathlib import Path

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.benchmarking.prompt5d import run_prompt5d
from experiments.benchmarking.prompt5d_postprocess import postprocess_prompt5d

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments/analytic_target_matching/configs/benchmark_campaigns/prompt5d_campaign_corrected.json"
DEFAULT_OUTPUT = ROOT / "runs/benchmarks/analytic/prompt5d_campaign_corrected"
DEFAULT_MASTER_TARGET_SOURCE = (
    ROOT / "runs/benchmarks/analytic/prompt5d_campaign/master_targets"
)


def execute(*, config: Path = DEFAULT_CONFIG, output: Path = DEFAULT_OUTPUT,
            device: str | None = None, smoke: bool = False,
            restart: bool = True,
            master_target_source: Path | None = DEFAULT_MASTER_TARGET_SOURCE) -> dict:
    # 1. Validate the frozen declaration and construct/recover master targets.
    # 2. Run bounded method workers against those identical target observations.
    # 3. Persist reusable numerical evidence; plotting never reruns inference.
    result = run_prompt5d(
        config, output=output, device=device, smoke=smoke, restart=restart,
        master_target_source=master_target_source,
    )
    if not smoke:
        postprocess_prompt5d(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument(
        "--master-target-source", type=Path, default=DEFAULT_MASTER_TARGET_SOURCE
    )
    parser.add_argument("--smoke", action="store_true", help="mechanics only; no claim")
    parser.add_argument("--fresh", action="store_true", help="disable safe record recovery")
    args = parser.parse_args()
    execute(
        config=args.config, output=args.output, device=args.device,
        smoke=args.smoke, restart=not args.fresh,
        master_target_source=args.master_target_source,
    )


if __name__ == "__main__":
    main()
