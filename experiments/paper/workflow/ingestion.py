"""Read-only source binding interfaces for future paper consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BoundIngestionSource:
    source_id: str
    kind: str
    path: Path


def bind_ingestion_sources(
    specification: Mapping[str, Any], bindings: Sequence[str]
) -> tuple[BoundIngestionSource, ...]:
    """Bind ``source_id=path`` values to a validated ingestion specification."""

    declared = {
        str(item["source_id"]): str(item["kind"])
        for item in specification["source_fragments"]
    }
    parsed: dict[str, Path] = {}
    for binding in bindings:
        source_id, separator, raw_path = binding.partition("=")
        if not separator or not source_id or not raw_path:
            raise ValueError("ingestion binding must use source_id=path")
        if source_id in parsed:
            raise ValueError(f"duplicate ingestion binding: {source_id}")
        parsed[source_id] = Path(raw_path).resolve()
    if set(parsed) != set(declared):
        raise ValueError("ingestion bindings must exactly match declared source ids")
    return tuple(
        BoundIngestionSource(source_id, declared[source_id], parsed[source_id])
        for source_id in declared
    )


def validate_ingestion_bindings(
    specification: Mapping[str, Any], bindings: Sequence[str]
) -> dict[str, Any]:
    """Inspect source shapes without copying or consolidating artifacts."""

    sources = bind_ingestion_sources(specification, bindings)
    documents = []
    for source in sources:
        if not source.path.is_dir():
            raise ValueError(f"ingestion source is not a directory: {source.path}")
        if source.kind == "compact_bundle":
            required = (source.path / "manifest.json", source.path / "paired_records.jsonl")
            if not all(path.is_file() for path in required):
                raise ValueError(f"compact source lacks manifest or paired records: {source.path}")
        elif source.kind == "worker_population":
            if not any(path.is_dir() for path in source.path.glob("worker-*")):
                raise ValueError(f"worker source contains no worker directories: {source.path}")
        documents.append({
            "source_id": source.source_id,
            "kind": source.kind,
            "path": source.path.as_posix(),
        })
    return {
        "schema_version": "mutaco-paper-ingestion-validation-v1",
        "ingestion_identity": specification["ingestion_identity"],
        "status": "validated",
        "sources": documents,
        "scientific_calls": 0,
        "files_copied": 0,
        "consolidation_performed": False,
    }
