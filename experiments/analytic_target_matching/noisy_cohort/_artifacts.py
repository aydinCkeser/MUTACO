"""Canonical campaign artifacts and completed-target recovery."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from mutaco.atomic_io import atomic_write_text


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, document: Any) -> None:
    """Write deterministic, indented human-readable JSON atomically."""

    atomic_write_text(
        path,
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_target(directory: Path, result: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    documents = {
        "result": result,
        "scientific_history": result.get("scientific_projection"),
        "assessment": {
            key: result.get(key)
            for key in (
                "assessment_response",
                "assessment_seed",
                "target_error_l1",
                "target_error_l2",
                "design_error_l2",
                "latent_error_l1",
                "latent_error_l2",
            )
        },
    }
    for name, document in documents.items():
        path = directory / f"{name}.json"
        write_json(path, document)
        (directory / f"{name}.sha256").write_text(file_hash(path), encoding="ascii")


def recover_target(directory: Path, campaign_id: str) -> dict[str, Any] | None:
    names = ("result", "scientific_history", "assessment")
    expected = tuple(
        path
        for name in names
        for path in (directory / f"{name}.json", directory / f"{name}.sha256")
    )
    if not any(path.exists() for path in expected):
        return None
    if not all(path.is_file() for path in expected):
        raise ValueError(f"incomplete target artifact: {directory}")
    for name in names:
        path = directory / f"{name}.json"
        recorded = (directory / f"{name}.sha256").read_text(encoding="ascii").strip()
        if file_hash(path) != recorded:
            raise ValueError(f"target artifact hash mismatch ({name}): {directory}")
    result_path = directory / "result.json"
    result = read_json(result_path)
    if result["campaign_id"] != campaign_id:
        raise ValueError(f"target campaign identity mismatch: {directory}")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "target_index", "status", "target_error_l1", "target_error_l2",
        "design_error_l2", "latent_error_l1", "latent_error_l2",
        "runtime_seconds", "failure_explanation",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
