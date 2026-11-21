"""Computation-free consolidation of v3 methods with compact paper evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mutaco import stable_hash
from experiments.benchmarking.artifacts import file_hash, write_pretty_json

from .protocol_v3 import PaperComparisonConfigV3, paired_assessment_context
from .validation import PaperContractError


_COMPACT_FIELDS = (
    "status",
    "recommended_theta",
    "response_error_l1_sum",
    "response_error_mae",
    "response_error_l2",
    "parameter_error_l2",
    "call_ledger",
    "method_configuration_identity",
    "record_content_hash",
)


def compact_method_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one validated full v3 record onto the compact evidence surface."""

    method = str(record.get("method", ""))
    if not method:
        raise PaperContractError("compact method record is missing its method")
    decision = record.get("decision")
    if not isinstance(decision, Mapping):
        raise PaperContractError("appended v3 record is missing its frozen decision")
    missing = [field for field in _COMPACT_FIELDS if field not in record]
    if missing:
        raise PaperContractError(f"appended v3 record lacks compact fields: {missing}")
    return {
        **{field: deepcopy(record[field]) for field in _COMPACT_FIELDS},
        "decision_policy_identifier": decision.get("decision_policy_identifier"),
        "timing_events": deepcopy(record.get("population_timing_events", [])),
        "assessment_context": deepcopy(record.get("assessment_context")),
        "assessment_randomness_evidence": deepcopy(
            record.get("assessment_randomness_evidence")
        ),
    }


def join_compact_method_records(
    existing_rows: Sequence[Mapping[str, Any]],
    appended_records: Sequence[Mapping[str, Any]],
    *,
    comparison: PaperComparisonConfigV3 | None = None,
) -> tuple[dict[str, Any], ...]:
    """Join missing method records by target identity without mutation or calls."""

    rows = [deepcopy(dict(row)) for row in existing_rows]
    by_target: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = str(row.get("target_record_identity", ""))
        methods = row.get("methods")
        if not target or not isinstance(methods, Mapping) or target in by_target:
            raise PaperContractError("existing compact rows have invalid target identities")
        by_target[target] = row
    seen = set()
    paired_evidence: dict[str, Mapping[str, Any]] = {}
    for record in appended_records:
        target = str(record.get("target_record_identity", ""))
        if target not in by_target:
            raise PaperContractError("SNPE append target is absent from compact evidence")
        method = str(record.get("method", ""))
        slot = (target, method)
        if slot in seen:
            raise PaperContractError("duplicate appended target-method record")
        methods = by_target[target]["methods"]
        if method in methods:
            raise PaperContractError(f"compact evidence already contains {method}")
        if comparison is not None:
            if record.get("paper_comparison_configuration_identity") != comparison.identity:
                raise PaperContractError("appended comparison identity is incompatible")
            if record.get("simulator_configuration_identity") != comparison.simulator_configuration_identity:
                raise PaperContractError("appended simulator configuration is incompatible")
            expected_context = paired_assessment_context(comparison, target)
            if record.get("assessment_context") != expected_context:
                raise PaperContractError("appended assessment family is incompatible")
        evidence = record.get("assessment_randomness_evidence")
        if evidence is not None:
            if not isinstance(evidence, Mapping):
                raise PaperContractError("appended assessment randomness evidence is malformed")
            previous = paired_evidence.get(target)
            if previous is not None and (
                evidence.get("coordinate_family_identities")
                != previous.get("coordinate_family_identities")
                or evidence.get("namespace_seed_manifest_hashes")
                != previous.get("namespace_seed_manifest_hashes")
            ):
                raise PaperContractError("paired assessment randomness namespaces differ")
            paired_evidence[target] = evidence
        methods[method] = compact_method_record(record)
        seen.add(slot)
    return tuple(rows)


def write_consolidated_compact_evidence(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_identities: Sequence[str],
) -> dict[str, Any]:
    """Write a new immutable compact bundle; source evidence is never modified."""

    if output.exists():
        raise PaperContractError("consolidated output must be a new append-only path")
    targets = [str(row.get("target_record_identity", "")) for row in rows]
    if not targets or len(set(targets)) != len(targets) or any(not target for target in targets):
        raise PaperContractError("consolidated compact targets must be nonempty and unique")
    sources = tuple(str(value) for value in source_identities)
    if not sources or any(not value for value in sources):
        raise PaperContractError("consolidation requires source evidence identities")
    output.mkdir(parents=True)
    records_path = output / "paired_records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    methods = sorted({method for row in rows for method in row["methods"]})
    body = {
        "schema_version": "mutaco-paper-v3-consolidated-compact-evidence-v1",
        "source_evidence_identities": list(sources),
        "target_count": len(rows),
        "methods": methods,
        "paired_records_sha256": file_hash(records_path),
        "scientific_calls_added_by_consolidation": 0,
    }
    manifest = {
        **body,
        "consolidation_identity": stable_hash(
            "paper-v3-consolidated-compact-evidence-v1", body
        ),
    }
    write_pretty_json(output / "manifest.json", manifest)
    return manifest

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.compact_append")
