"""Materialize figure-ready summaries from persisted validated records only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper import load_paper_records, summarize_paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize_paired(load_paper_records(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
