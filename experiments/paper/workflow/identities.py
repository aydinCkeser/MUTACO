"""Content identities for paper workflow documents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from mutaco import stable_hash


IDENTITY_FIELDS = {
    "mutaco-paper-study-spec-v1": ("study_identity", "paper-study-spec-v1"),
    "mutaco-paper-execution-spec-v1": (
        "execution_identity",
        "paper-execution-spec-v1",
    ),
    "mutaco-paper-ingestion-spec-v1": (
        "ingestion_identity",
        "paper-ingestion-spec-v1",
    ),
    "mutaco-paper-analysis-spec-v1": (
        "analysis_identity",
        "paper-analysis-spec-v1",
    ),
    "mutaco-paper-report-spec-v1": ("report_identity", "paper-report-spec-v1"),
    "mutaco-paper-figure-spec-v1": ("figure_identity", "paper-figure-spec-v1"),
    "mutaco-paper-exploration-spec-v1": (
        "exploration_identity",
        "paper-exploration-spec-v1",
    ),
}


def document_identity(document: Mapping[str, Any]) -> str:
    """Return the canonical identity of one recognized workflow document."""

    schema = str(document.get("schema_version", ""))
    try:
        identity_field, domain = IDENTITY_FIELDS[schema]
    except KeyError as exc:
        raise ValueError(f"unsupported paper workflow schema: {schema!r}") from exc
    body = deepcopy(dict(document))
    body.pop(identity_field, None)
    return stable_hash(domain, body)


def assert_document_identity(document: Mapping[str, Any]) -> str:
    """Validate a persisted identity and return it."""

    schema = str(document.get("schema_version", ""))
    try:
        identity_field, _ = IDENTITY_FIELDS[schema]
    except KeyError as exc:
        raise ValueError(f"unsupported paper workflow schema: {schema!r}") from exc
    expected = document_identity(document)
    if document.get(identity_field) != expected:
        raise ValueError(f"{identity_field} does not match canonical document content")
    return expected
