"""Read-only inventory for append-only JSON/hash evidence trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_inventory(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError("inventory root must be a directory")
    json_paths = tuple(sorted(resolved.rglob("*.json")))
    sidecars = tuple(sorted(resolved.rglob("*.sha256")))
    entries = []
    mismatches = []
    missing = []
    paired_sidecars = set()
    for path in json_paths:
        relative = path.relative_to(resolved).as_posix()
        sidecar = path.with_suffix(".sha256")
        actual = _file_hash(path)
        if not sidecar.is_file():
            missing.append(relative)
        else:
            paired_sidecars.add(sidecar.resolve())
            expected = sidecar.read_text(encoding="ascii").strip()
            if expected != actual:
                mismatches.append(relative)
        entries.append({"path": relative, "sha256": actual, "size_bytes": path.stat().st_size})
    orphan_sidecars = [
        path.relative_to(resolved).as_posix()
        for path in sidecars
        if path.resolve() not in paired_sidecars
    ]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False)
    categories = {
        "worker_start": sum(path.name == "worker_start.json" for path in json_paths),
        "worker_terminal": sum(path.name == "worker_terminal.json" for path in json_paths),
        "target_execution": sum(path.name == "target_execution.json" for path in json_paths),
        "decision_freeze": sum(path.name == "decision_freeze.json" for path in json_paths),
        "mutaco": sum(path.name == "mutaco.json" for path in json_paths),
        "scalar_qlognei": sum(path.name == "scalar_qlognei.json" for path in json_paths),
        "sobol": sum(path.name == "sobol.json" for path in json_paths),
        "collection": sum(path.parent.name == "collection" for path in json_paths),
    }
    return {
        "schema_version": "mutaco-json-hash-pair-inventory-v1",
        "json_files": len(json_paths),
        "sha256_sidecars": len(sidecars),
        "hash_mismatches": mismatches,
        "missing_sidecars": missing,
        "orphan_sidecars": orphan_sidecars,
        "json_bytes": sum(path.stat().st_size for path in json_paths),
        "categories": categories,
        "inventory_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "scientific_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_inventory(args.root), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
