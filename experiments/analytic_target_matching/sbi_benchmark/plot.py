"""Regenerate plots for one selected method from persisted results only."""

from __future__ import annotations

import argparse
from pathlib import Path

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.benchmarking import load_results, plot_results, summarize_results


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "runs/benchmarks/analytic/prompt5b_sbi_smoke"
DEFAULT_STYLE = ROOT / "experiments/analytic_target_matching/noisy_cohort/styles/noisy_cohort_default.mplstyle"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--method", default="mutaco")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg"), default=("pdf", "svg"))
    args = parser.parse_args()
    results = load_results(args.input).select(methods=(args.method,))
    summary = summarize_results(results)
    manifest = plot_results(
        results,
        summary,
        output=args.output or args.input / "plots" / args.method,
        style=args.style,
        formats=args.formats,
    )
    print(f"[benchmark plot] wrote {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
