"""Validate disjoint source, scheduler-log, and qualification-output paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class QualificationSchedulerLayout:
    source_root: Path
    log_root: Path
    output_root: Path

    def validate(self) -> dict[str, str | bool]:
        source = self.source_root.resolve()
        logs = self.log_root.resolve()
        output = self.output_root.resolve()
        if not source.is_dir() or not (source / ".git").exists():
            raise ValueError("source root must be an existing Git checkout")
        pairs = (
            ("scheduler logs", logs, "source", source),
            ("qualification output", output, "source", source),
            ("scheduler logs", logs, "qualification output", output),
        )
        for left_name, left, right_name, right in pairs:
            if _contains(left, right) or _contains(right, left):
                raise ValueError(f"{left_name} and {right_name} paths overlap")
        return {
            "schema_version": "mutaco-truba-qualification-layout-v1",
            "source_root": source.as_posix(),
            "scheduler_log_root": logs.as_posix(),
            "qualification_output_root": output.as_posix(),
            "paths_pairwise_disjoint": True,
            "scientific_calls": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(QualificationSchedulerLayout(
        args.source_root, args.log_root, args.output_root
    ).validate(), indent=2))


if __name__ == "__main__":
    main()
