"""Atomic append-only persistence for independently executable methods."""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from typing import Any, Mapping

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json

from .validation import PaperContractError, assert_record_compatible
from .protocol_v3 import validate_paper_record_schema


def record_path(root: Path, target_identity: str, method: str) -> Path:
    directory = "target-" + sha256(target_identity.encode("utf-8")).hexdigest()[:16]
    return root / "records" / directory / f"{method}.json"


def load_method_record(path: Path, expected: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    sidecar = path.with_suffix(".sha256")
    if not path.exists() and not sidecar.exists():
        return None
    if not path.is_file() or not sidecar.is_file():
        raise PaperContractError(f"incomplete paper method record: {path}")
    if file_hash(path) != sidecar.read_text(encoding="ascii").strip():
        raise PaperContractError(f"paper method record hash mismatch: {path}")
    record = read_json(path)
    content = {key: value for key, value in record.items() if key != "record_content_hash"}
    if record.get("record_content_hash") != stable_hash("paper-method-record-v1", content):
        raise PaperContractError(f"paper method content hash mismatch: {path}")
    validate_paper_record_schema(record)
    if expected is not None:
        assert_record_compatible(record, expected)
    return record


def append_method_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    content = dict(record)
    validate_paper_record_schema(content)
    content["record_content_hash"] = stable_hash("paper-method-record-v1", content)
    existing = load_method_record(path)
    if existing is not None:
        if existing != content:
            raise PaperContractError(f"append-only record already exists with different content: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pretty_json(path, content)
    path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")
    return content

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.method_append")
