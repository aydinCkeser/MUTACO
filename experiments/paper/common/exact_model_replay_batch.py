"""Batch exact-state replay over an immutable raw-trajectory archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from exact_model_replay import SCHEMA, _write_once, replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=(40, 100), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="qualified")
    parser.add_argument("--exclude-position", type=int, action="append", default=[])
    arguments = parser.parse_args()

    raw_paths = sorted(arguments.raw_root.resolve().glob("worker-*/raw/target-*/mutaco.json"))
    if arguments.budget == 100 and not raw_paths:
        raw_paths = sorted(
            arguments.raw_root.resolve().glob("*/worker-*/raw/target-*/mutaco.json")
        )
    selected: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for raw_path in raw_paths:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        position = int(raw["target_index"])
        if position in seen:
            raise SystemExit(f"duplicate target position in archive: {position}")
        seen.add(position)
        if position not in arguments.exclude_position:
            selected.append((position, raw_path))
    expected = 120 - len(set(arguments.exclude_position))
    if len(seen) != 120 or len(selected) != expected:
        raise SystemExit(
            f"archive coverage mismatch: total={len(seen)}, selected={len(selected)}, expected={expected}"
        )

    started = perf_counter()
    passed = 0
    failed = 0
    for ordinal, (position, raw_path) in enumerate(sorted(selected), start=1):
        output = (
            arguments.output_dir.resolve()
            / f"{arguments.output_prefix}-b{arguments.budget}-target-{position:03d}.json"
        )
        try:
            document = replay(
                raw_path,
                raw_artifact=f"b{arguments.budget}/target-{position:03d}/mutaco.json",
                expected_budget=arguments.budget,
                source_commit=arguments.source_commit,
            )
        except Exception as exc:
            document = {
                "schema_version": SCHEMA,
                "source_commit": arguments.source_commit,
                "raw_artifact": f"b{arguments.budget}/target-{position:03d}/mutaco.json",
                "target_position": position,
                "budget": arguments.budget,
                "qualification_passed": False,
                "failure_type": type(exc).__name__,
                "failure_explanation": str(exc),
                "simulator_constructed": False,
                "simulator_calls": 0,
                "terminal_recommendation_executed": False,
            }
        _write_once(output, document)
        if document.get("qualification_passed"):
            passed += 1
        else:
            failed += 1
        print(
            json.dumps(
                {
                    "budget": arguments.budget,
                    "ordinal": ordinal,
                    "total": len(selected),
                    "target_position": position,
                    "qualification_passed": bool(document.get("qualification_passed")),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "budget": arguments.budget,
        "selected": len(selected),
        "passed": passed,
        "failed": failed,
        "elapsed_seconds": perf_counter() - started,
        "simulator_calls": 0,
        "terminal_recommendations": 0,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
