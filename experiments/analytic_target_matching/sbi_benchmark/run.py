"""Run the declared analytic benchmark with visible progress."""

from __future__ import annotations

import argparse
from pathlib import Path

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.benchmarking import load_campaign, run_campaign


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments/analytic_target_matching/configs/benchmark_campaigns/prompt5b_sbi_smoke.json"
DEFAULT_OUTPUT = ROOT / "runs/benchmarks/analytic/prompt5b_sbi_smoke"


def execute(
    *,
    config_path: Path,
    output: Path,
    device: str | None = None,
    restart: bool = False,
) -> dict:
    # 1. Read and validate the complete scientific campaign declaration.
    campaign = load_campaign(config_path, device=device)

    # 2. Run the selected simulator, MUTACO reference, and optional SBI methods.
    execution = run_campaign(campaign, output=output, restart=restart)

    # 3. Complete numerical evidence is now reusable without rerunning science.
    print(
        f"[benchmark] completed campaign={execution.manifest['campaign_id']} "
        f"output={execution.output}"
    )
    return execution.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    execute(
        config_path=args.config,
        output=args.output,
        device=args.device,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
