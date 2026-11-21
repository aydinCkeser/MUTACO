"""Small deterministic artifact helpers for paper figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.benchmarking.artifacts import file_hash


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")


def write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_bundle_text(root: Path) -> None:
    """Canonicalize generated text for stable hashes on every platform."""

    text_suffixes = {".csv", ".json", ".log", ".md", ".sha256", ".svg"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        content = "\n".join(line.rstrip() for line in lines).rstrip() + "\n"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)


def inventory(root: Path, *, exclude: Sequence[str] = ()) -> list[dict[str, Any]]:
    excluded = set(exclude)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.figures.common.figure_artifacts")
