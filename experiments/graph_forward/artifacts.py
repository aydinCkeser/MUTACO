"""Public validated artifact I/O for the graph forward experiment."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from mutaco.atomic_io import atomic_write_text


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, document: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("graph artifact CSV requires at least one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return True


def load_forward_artifacts(directory: str | Path) -> dict[str, Any]:
    """Load and validate public numerical artifacts without private internals."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise ValueError("graph forward artifact directory is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "prompt7-graph-forward-manifest-v1":
        raise ValueError("unsupported graph forward manifest schema")
    if result.get("schema_version") != "prompt7-graph-forward-result-v1":
        raise ValueError("unsupported graph forward result schema")
    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"graph forward artifact hash mismatch: {relative}")
    if len(result["response"]) != 20 or len(result["response_names"]) != 20:
        raise ValueError("graph forward response contract is not M=20")
    if not _finite_tree(result):
        raise ValueError("graph forward result contains nonfinite values")
    return {"manifest": manifest, "result": result}
