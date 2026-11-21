"""Minimal strict schemas used before full run serialization exists."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from mutaco.identity import canonical_json_bytes


ARTIFACT_SCHEMAS: dict[str, str] = {
    "config.json": "mutaco-config-v1",
    "provenance.json": "mutaco-provenance-v1",
    "status.json": "mutaco-status-v1",
    "events.jsonl": "mutaco-event-v1",
    "outcome.json": "mutaco-outcome-v1",
    "observations.json": "mutaco-observations-v1",
    "recommendation.json": "mutaco-recommendation-v1",
    "accounting.json": "mutaco-accounting-v1",
    "timings.json": "mutaco-timings-v1",
    "profile.json": "mutaco-profile-v1",
    "manifest.json": "mutaco-manifest-v1",
}


class ArtifactValidationError(ValueError):
    """An artifact document does not satisfy its declared foundation schema."""


def make_artifact_document(filename: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if filename not in ARTIFACT_SCHEMAS:
        raise ArtifactValidationError(f"unsupported artifact filename: {filename}")
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("artifact payload must be a mapping")
    document = {
        "schema_version": ARTIFACT_SCHEMAS[filename],
        "artifact_type": filename,
        **dict(payload),
    }
    validate_artifact_document(filename, document)
    return document


def validate_artifact_document(filename: str, document: Mapping[str, Any]) -> None:
    if filename not in ARTIFACT_SCHEMAS:
        raise ArtifactValidationError(f"unsupported artifact filename: {filename}")
    if not isinstance(document, Mapping):
        raise ArtifactValidationError("artifact document must be a mapping")
    if document.get("schema_version") != ARTIFACT_SCHEMAS[filename]:
        raise ArtifactValidationError(f"{filename} has an unsupported schema version")
    if document.get("artifact_type") != filename:
        raise ArtifactValidationError(f"{filename} has a mismatched artifact_type")
    canonical_json_bytes(document)


def build_manifest(artifacts: Mapping[str, bytes]) -> dict[str, Any]:
    """Hash immutable finalized artifacts; the manifest cannot hash itself."""

    if "manifest.json" in artifacts:
        raise ArtifactValidationError("manifest.json cannot include itself")
    entries: list[dict[str, Any]] = []
    for filename in sorted(artifacts):
        raw = artifacts[filename]
        if not isinstance(raw, bytes):
            raise ArtifactValidationError("manifest inputs must be finalized bytes")
        entries.append(
            {
                "path": filename,
                "sha256": sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    return make_artifact_document("manifest.json", {"artifacts": entries})


# Trusted historical pickles refer to this original Python module name.
for _legacy_export in (
    ArtifactValidationError,
    make_artifact_document,
    validate_artifact_document,
    build_manifest,
):
    _legacy_export.__module__ = "mutaco.artifacts.schemas"
del _legacy_export
