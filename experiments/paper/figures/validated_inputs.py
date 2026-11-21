"""Read-only validation for immutable Paper-100 figure sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json


EXPECTED_RECEIPT_SHA256 = (
    "81cdc49deb8c9fcbc20de68b478214967642630e7d6e0dbb341fa654c36dc799"
)


def load_validated_target_ledger(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    repository_root = Path(__file__).resolve().parents[3]

    def source_key(path: Path) -> str:
        return path.resolve().relative_to(repository_root).as_posix()

    ledger = root / "target_ledger"
    records, hashes = [], {}
    for path in sorted(ledger.glob("target_*.json")):
        sidecar = path.with_suffix(".sha256")
        if not sidecar.is_file() or file_hash(path) != sidecar.read_text(
            encoding="ascii"
        ).strip():
            raise ValueError(f"target ledger hash mismatch: {path}")
        record = read_json(path)
        identity_content = {
            key: value
            for key, value in record.items()
            if key not in {"target_record_id", "content_hash"}
        }
        content_hash = stable_hash("target-record-content-v1", identity_content)
        record_id = stable_hash(
            "target-record-v1", {"content_hash": content_hash, **identity_content}
        )
        if record.get("content_hash") != content_hash or record.get(
            "target_record_id"
        ) != record_id:
            raise ValueError(f"target ledger identity mismatch: {path}")
        records.append(record)
        hashes[source_key(path)] = file_hash(path)
    if len(records) != 100 or sorted(row["target_index"] for row in records) != list(
        range(100)
    ):
        raise ValueError("Paper-100 target ledger must contain indices 0..99")
    receipt = root / "executions" / "receipt-5c0a8e41a082bf12.json"
    if file_hash(receipt) != EXPECTED_RECEIPT_SHA256:
        raise ValueError("accepted Paper-100 receipt hash mismatch")
    hashes[source_key(receipt)] = file_hash(receipt)
    return records, hashes


def m20_gauge_zero_indices() -> dict[str, tuple[int, ...]]:
    omega = tuple(2 * force_index for force_index in range(5))
    synchronization = tuple(10 + 2 * force_index for force_index in range(5))
    return {"omega": omega, "synchronization": synchronization}

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.figures.common.validated_inputs")
