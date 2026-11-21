"""Versioned run-artifact foundations."""

import sys as _sys

from mutaco.file_management import artifact_documents_and_manifests as schemas
from mutaco.file_management.artifact_documents_and_manifests import (
    ARTIFACT_SCHEMAS,
    ArtifactValidationError,
    build_manifest,
    make_artifact_document,
    validate_artifact_document,
)

_sys.modules[__name__ + ".schemas"] = schemas

__all__ = [
    "ARTIFACT_SCHEMAS",
    "ArtifactValidationError",
    "build_manifest",
    "make_artifact_document",
    "validate_artifact_document",
]
