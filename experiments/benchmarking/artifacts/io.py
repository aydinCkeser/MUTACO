"""Canonical evidence and complementary human-readable artifact I/O."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mutaco import canonical_json_bytes
from mutaco.atomic_io import atomic_write_bytes


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    atomic_write_bytes(path, payload)


def write_canonical_json(path: Path, document: Any) -> None:
    _atomic_bytes(path, canonical_json_bytes(document) + b"\n")


def write_pretty_json(path: Path, document: Any) -> None:
    payload = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


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
        write_pretty_json(path, document)
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
    result = read_json(directory / "result.json")
    if result["campaign_id"] != campaign_id:
        raise ValueError(f"target campaign identity mismatch: {directory}")
    return result
