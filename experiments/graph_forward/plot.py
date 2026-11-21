"""Replot an existing validated Prompt 7 graph-forward artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import load_forward_artifacts
from .plotting import render_forward_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    loaded = load_forward_artifacts(args.input)
    output = args.output or args.input / "replotted"
    written = render_forward_plots(loaded["result"], output)
    print(f"[graph-forward replot] wrote {len(written)} files to {output}")


if __name__ == "__main__":
    main()
