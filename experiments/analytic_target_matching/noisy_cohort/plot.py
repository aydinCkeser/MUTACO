"""Regenerate noisy-cohort PDF/SVG figures from saved metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.analytic_target_matching.noisy_cohort._artifacts import write_json
from experiments.analytic_target_matching.noisy_cohort._plotting import generate_plots


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "runs/analytic_target_matching/noisy_cohort_100"
DEFAULT_STYLE = ROOT / "experiments/analytic_target_matching/noisy_cohort/styles/noisy_cohort_default.mplstyle"


def execute(*, input_path: Path, output: Path, style: Path, formats: tuple[str, ...]) -> dict:
    summary = json.loads((input_path / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((input_path / "metrics.json").read_text(encoding="utf-8"))["rows"]
    result = generate_plots(
        summary=summary, rows=rows, output=output, style=style, formats=formats
    )
    write_json(output / "plot_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg"), default=("pdf", "svg"))
    args = parser.parse_args()
    execute(
        input_path=args.input,
        output=args.output or args.input / "plots",
        style=args.style,
        formats=tuple(args.formats),
    )


if __name__ == "__main__":
    main()
