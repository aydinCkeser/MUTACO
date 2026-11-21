from __future__ import annotations

import pytest

from mutaco.artifacts import (
    ARTIFACT_SCHEMAS,
    ArtifactValidationError,
    build_manifest,
    make_artifact_document,
    validate_artifact_document,
)


def test_every_declared_artifact_has_a_valid_foundation_document() -> None:
    for filename in ARTIFACT_SCHEMAS:
        document = make_artifact_document(filename, {"status": "foundation"})
        validate_artifact_document(filename, document)
        assert document["artifact_type"] == filename


def test_wrong_schema_and_unknown_artifact_fail_closed() -> None:
    with pytest.raises(ArtifactValidationError, match="unsupported"):
        make_artifact_document("unknown.json", {})
    with pytest.raises(ArtifactValidationError, match="schema"):
        validate_artifact_document(
            "status.json",
            {"schema_version": "wrong", "artifact_type": "status.json"},
        )


def test_manifest_hashes_finalized_bytes_in_stable_path_order() -> None:
    manifest = build_manifest({"z.json": b"z", "a.json": b"alpha"})
    entries = manifest["artifacts"]
    assert [entry["path"] for entry in entries] == ["a.json", "z.json"]
    assert entries[0]["size_bytes"] == 5
    assert len(entries[0]["sha256"]) == 64
    with pytest.raises(ArtifactValidationError, match="cannot include itself"):
        build_manifest({"manifest.json": b"recursive"})
