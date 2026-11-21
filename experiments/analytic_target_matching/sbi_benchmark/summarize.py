"""Recompute and export summaries from a completed campaign without simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.benchmarking import export_tables, load_results, summarize_results


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "runs/benchmarks/analytic/prompt5b_sbi_smoke"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()
    results = load_results(args.input)
    if args.methods:
        results = results.select(methods=args.methods)
    summary = summarize_results(results, group_by=("method", "view", "regime"))
    written = export_tables(
        results, summary, output=args.output or args.input / "tables"
    )
    print(f"[benchmark summarize] wrote {len(written)} artifacts")


if __name__ == "__main__":
    main()
