"""Combine independent TRUBA thread benchmark records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.input / f"threads_{n}.json").read_text()) for n in (1, 4, 8, 20, 40)]
    baseline = rows[0]["response"]
    for row in rows:
        delta = max(abs(a - b) for a, b in zip(baseline, row["response"], strict=True))
        row["max_abs_response_difference_from_one_thread"] = delta
    best = min(rows, key=lambda row: row["median_seconds"])
    summary = {
        "schema_version": "mutaco-truba-thread-summary-v1",
        "classification": "engineering_no_claim",
        "rows": rows,
        "fastest_observed_threads": best["threads_requested"],
        "fastest_observed_median_seconds": best["median_seconds"],
        "cross_thread_exact_response": all(row["response"] == baseline for row in rows),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
