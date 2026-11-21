"""Zero-call verification gate for canonical Paper Lab promotion state."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any, Mapping

from reproducibility.recorded_paths import resolve_recorded_path
from experiments.benchmarking.artifacts import read_json

from .reporting import build_report, validate_report_source


def _has_bound_entry(value: Any, source_path: str, bundle_identity: str) -> bool:
    if isinstance(value, Mapping):
        direct = tuple(value.values())
        if source_path in direct and bundle_identity in direct:
            return True
        return any(
            _has_bound_entry(item, source_path, bundle_identity)
            for item in direct
        )
    if isinstance(value, list):
        return any(
            _has_bound_entry(item, source_path, bundle_identity) for item in value
        )
    return False


def verify_promotion(
    specification: Mapping[str, Any], repository_root: Path, *,
    state_path: Path, registry_path: Path,
) -> dict[str, Any]:
    """Require evidence, report, state, and experiment registry to agree."""

    validation = validate_report_source(specification, repository_root)
    build_report(specification, repository_root, confirm_authorized=False, check=True)
    source_path = str(specification["source"]["path"])
    manifest = read_json(resolve_recorded_path(repository_root, source_path))
    bundle_identity = str(manifest.get("bundle_identity", ""))
    if not bundle_identity:
        raise ValueError("promotion requires canonical evidence with a bundle identity")
    state = tomllib.loads(state_path.read_text(encoding="utf-8"))
    registry = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    for name, document in (("project state", state), ("experiment registry", registry)):
        if not _has_bound_entry(document, source_path, bundle_identity):
            raise ValueError(f"{name} is not aligned with canonical evidence")
    return {
        "schema_version": "mutaco-paper-promotion-verification-v1",
        "status": "promotion_verified",
        "report_identity": specification["report_identity"],
        "bundle_identity": bundle_identity,
        "source_sha256": validation["source_sha256"],
        "scientific_calls": 0,
        "files_written": 0,
    }
