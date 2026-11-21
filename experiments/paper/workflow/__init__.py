"""Approval-aware orchestration foundations for paper experiments."""

from .identities import document_identity
from .consolidation import build_consolidated_bundle, verify_consolidated_bundle
from .ingestion import bind_ingestion_sources, validate_ingestion_bindings
from .promotion import verify_promotion
from .reporting import build_report, render_report, validate_report_source
from .schemas import load_and_validate, validate_document
from .transport import export_compact_population, fetch_compact_export

__all__ = [
    "bind_ingestion_sources",
    "build_consolidated_bundle",
    "build_report",
    "document_identity",
    "export_compact_population",
    "fetch_compact_export",
    "load_and_validate",
    "render_report",
    "validate_document",
    "validate_ingestion_bindings",
    "validate_report_source",
    "verify_consolidated_bundle",
    "verify_promotion",
]
