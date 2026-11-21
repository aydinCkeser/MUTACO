"""Generic zero-call consolidation of validated Paper-v3 worker populations."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.paper.analysis.aggregate import PaperRecordSet, summarize_population_v3
from experiments.paper.protocol.compact_append import compact_method_record
from experiments.paper.protocol.method_append import load_method_record
from experiments.paper.common.validation import PaperContractError

from .ingestion import BoundIngestionSource, bind_ingestion_sources


_CALL_FIELDS = ("attempted_calls", "successful_calls", "failed_calls", "retry_calls")
_DATA_FILES = (
    "paired_records.jsonl",
    "population_collection.json",
    "population_summary.json",
)


def _load_hashed(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise PaperContractError(f"missing hash-paired artifact: {path}")
    if file_hash(path) != sidecar.read_text(encoding="ascii").strip():
        raise PaperContractError(f"artifact hash mismatch: {path}")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise PaperContractError(f"artifact is not a JSON object: {path}")
    return dict(value)


def _repository_path(root: Path, relative: str) -> Path:
    repository = root.resolve()
    path = resolve_recorded_path(repository, relative).resolve()
    try:
        path.relative_to(repository)
    except ValueError as exc:
        raise PaperContractError(f"paper path escapes repository: {path}") from exc
    return path


def _cohort_targets(manifest_path: Path, expected_sha256: str) -> tuple[str, ...]:
    if file_hash(manifest_path) != expected_sha256:
        raise PaperContractError("cohort manifest hash mismatch")
    manifest = read_json(manifest_path)
    count = manifest.get("completed_count")
    records = manifest.get("records")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise PaperContractError("cohort completed_count is invalid")
    if not isinstance(records, Mapping) or set(records) != {str(i) for i in range(count)}:
        raise PaperContractError("cohort record map is incomplete")
    targets = tuple(str(records[str(i)]["target_record_id"]) for i in range(count))
    if len(set(targets)) != count:
        raise PaperContractError("cohort target identities are not unique")
    return targets


def _source_fragment(
    specification: Mapping[str, Any], source_id: str
) -> Mapping[str, Any]:
    return next(
        item for item in specification["source_fragments"]
        if item["source_id"] == source_id
    )


def _validate_source_collection(
    source: BoundIngestionSource, fragment: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if source.kind != "worker_population":
        raise PaperContractError("canonical population build requires worker_population sources")
    collection_path = source.path / "collection" / "population_collection.json"
    collection = _load_hashed(collection_path)
    expected = {
        "source_commit": fragment["source_commit"],
        "declaration_identity": fragment["declaration_identity"],
        "declared_target_count": fragment["target_count"],
        "declared_slot_count": fragment["record_count"],
        "collected_slot_count": fragment["record_count"],
        "status": "completed",
        "scientific_calls_added_by_collection": 0,
    }
    mismatches = {
        key: {"expected": value, "observed": collection.get(key)}
        for key, value in expected.items()
        if collection.get(key) != value
    }
    if mismatches:
        raise PaperContractError(
            f"source collection metadata mismatch for {source.source_id}: {mismatches}"
        )
    if collection.get("errors") != []:
        raise PaperContractError(f"source collection contains errors: {source.source_id}")
    calls = collection.get("physical_calls")
    if not isinstance(calls, Mapping) or calls.get("successful_calls") != fragment["successful_calls"]:
        raise PaperContractError(f"source collection call ledger mismatch: {source.source_id}")
    if calls.get("attempted_calls") != calls.get("successful_calls") or calls.get("failed_calls") != 0:
        raise PaperContractError(f"source collection is not failure-free: {source.source_id}")
    actual_collection_hash = file_hash(collection_path)
    if actual_collection_hash != fragment["collection_sha256"]:
        raise PaperContractError(f"source collection identity mismatch: {source.source_id}")

    records = []
    for path in sorted(source.path.glob("worker-*/records/*/*.json")):
        record = load_method_record(path)
        assert record is not None
        records.append(record)
    if len(records) != fragment["record_count"]:
        raise PaperContractError(f"source record count mismatch: {source.source_id}")
    return collection, tuple(records)


def _call_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {field: 0 for field in _CALL_FIELDS}
    for record in records:
        ledger = record.get("call_ledger")
        if not isinstance(ledger, Mapping):
            raise PaperContractError("method record has no call ledger")
        for category in ledger.values():
            if not isinstance(category, Mapping):
                raise PaperContractError("method call-ledger category is malformed")
            for field in totals:
                totals[field] += int(category.get(field, 0))
    return totals


def _validate_assessment_pairing(
    records_by_slot: Mapping[tuple[str, str], Mapping[str, Any]],
    targets: Sequence[str], methods: Sequence[str],
) -> dict[str, Any]:
    fields = (
        "coordinate_family_identities",
        "namespace_seed_manifest_hashes",
        "full_seed_manifest_hash",
        "root_evaluation_seed",
    )
    target_evidence = []
    for index, target in enumerate(targets):
        rows = [records_by_slot[(target, method)] for method in methods]
        contexts = [row.get("assessment_context") for row in rows]
        if any(context != contexts[0] for context in contexts[1:]):
            raise PaperContractError(f"assessment contexts differ within target {index}")
        evidence = [row.get("assessment_randomness_evidence") for row in rows]
        if not all(isinstance(item, Mapping) for item in evidence):
            raise PaperContractError(f"assessment randomness evidence is missing for target {index}")
        for field in fields:
            values = [item.get(field) for item in evidence]
            if any(value != values[0] for value in values[1:]):
                raise PaperContractError(
                    f"assessment randomness field {field} differs within target {index}"
                )
        target_evidence.append({
            "target_index": index,
            "target_record_identity": target,
            "root_evaluation_seed": evidence[0]["root_evaluation_seed"],
            "full_seed_manifest_hash": evidence[0]["full_seed_manifest_hash"],
        })
    return {
        "contexts_shared_across_methods_per_target": True,
        "coordinate_families_shared_across_methods_per_target": True,
        "namespace_manifests_shared_across_methods_per_target": True,
        "target_evidence": target_evidence,
    }


def _validated_population(
    specification: Mapping[str, Any], sources: Sequence[BoundIngestionSource],
    *, repository_root: Path,
) -> tuple[
    tuple[str, ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...],
    dict[str, str], dict[str, int], dict[str, Any],
]:
    cohort_path = _repository_path(repository_root, str(specification["cohort_manifest"]))
    targets = _cohort_targets(cohort_path, str(specification["cohort_sha256"]))
    if len(targets) != specification["expected_target_count"]:
        raise PaperContractError("cohort target count differs from ingestion specification")
    methods = tuple(str(value) for value in specification["methods"])
    all_records = []
    source_documents = []
    for source in sources:
        fragment = _source_fragment(specification, source.source_id)
        collection, records = _validate_source_collection(source, fragment)
        all_records.extend(records)
        source_documents.append({
            "source_id": source.source_id,
            "source_evidence_identity": fragment["source_evidence_identity"],
            "source_commit": collection["source_commit"],
            "declaration_identity": collection["declaration_identity"],
            "collection_sha256": fragment["collection_sha256"],
            "target_count": collection["declared_target_count"],
            "record_count": collection["collected_slot_count"],
            "successful_calls": collection["physical_calls"]["successful_calls"],
        })
    slots = {
        (str(record["target_record_identity"]), str(record["method"])): record
        for record in all_records
    }
    expected_slots = {(target, method) for target in targets for method in methods}
    if len(slots) != len(all_records) or set(slots) != expected_slots:
        raise PaperContractError("consolidated target-method coverage is incomplete or duplicated")
    if any(record.get("status") != "completed" for record in all_records):
        raise PaperContractError("canonical consolidation requires every method record to complete")
    config_identities = {}
    for method in methods:
        identities = {str(slots[(target, method)]["method_configuration_identity"]) for target in targets}
        if len(identities) != 1:
            raise PaperContractError(f"method configuration identity differs for {method}")
        config_identities[method] = identities.pop()
    totals = _call_totals(all_records)
    expected_calls = {field: int(specification["expected_physical_calls"][field]) for field in _CALL_FIELDS}
    if totals != expected_calls:
        raise PaperContractError(f"consolidated physical-call ledger mismatch: {totals}")
    per_method_budget = int(specification["inference_budget_per_method_target"])
    for target, method in expected_slots:
        ledger = slots[(target, method)]["call_ledger"]
        inference = ledger["method_inference_or_training"]
        assessment = ledger["paired_assessment_and_metric"]
        if inference["successful_calls"] != per_method_budget or assessment["successful_calls"] != 1:
            raise PaperContractError("record budget or assessment accounting is incompatible")
    assessment = _validate_assessment_pairing(slots, targets, methods)
    ordered_records = tuple(slots[(target, method)] for target in targets for method in methods)
    rows = tuple({
        "target_position": index,
        "target_record_identity": target,
        "methods": {
            method: compact_method_record(slots[(target, method)]) for method in methods
        },
    } for index, target in enumerate(targets))
    return (
        targets, ordered_records, rows, config_identities, totals,
        {"sources": source_documents, "assessment": assessment},
    )


def _write_hash_sidecar(path: Path) -> None:
    path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


def build_consolidated_bundle(
    specification: Mapping[str, Any], bindings: Sequence[str], *,
    repository_root: Path, output: Path, confirm_authorized: bool,
) -> dict[str, Any]:
    """Build one new canonical compact bundle without simulator calls."""

    if not confirm_authorized:
        raise PaperContractError("consolidation requires explicit runtime authorization")
    sources = bind_ingestion_sources(specification, bindings)
    output_path = output.resolve()
    repository = repository_root.resolve()
    try:
        output_path.relative_to(repository)
    except ValueError as exc:
        raise PaperContractError("canonical consolidation output lies outside repository") from exc
    if output_path.exists():
        raise PaperContractError("canonical consolidation output must be a new append-only path")
    targets, records, rows, config_ids, calls, audit = _validated_population(
        specification, sources, repository_root=repository
    )
    summary = summarize_population_v3(
        PaperRecordSet(output_path, records),
        declared_target_identities=targets,
        bootstrap_seed=int(specification["bootstrap_seed"]),
        bootstrap_resamples=int(specification["bootstrap_resamples"]),
    )
    collection = {
        "schema_version": "mutaco-paper-canonical-population-collection-v1",
        "status": "completed",
        "ingestion_identity": specification["ingestion_identity"],
        "cohort_sha256": specification["cohort_sha256"],
        "target_count": len(targets),
        "methods": list(specification["methods"]),
        "record_count": len(records),
        "method_configuration_identities": config_ids,
        "physical_calls": calls,
        "assessment_audit": audit["assessment"],
        "sources": audit["sources"],
        "scientific_calls_added_by_consolidation": 0,
    }

    output_path.mkdir(parents=True)
    records_path = output_path / "paired_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8", newline="\n",
    )
    write_pretty_json(output_path / "population_collection.json", collection)
    write_pretty_json(output_path / "population_summary.json", summary)
    for name in _DATA_FILES:
        _write_hash_sidecar(output_path / name)
    inventory = {
        "schema_version": "mutaco-paper-canonical-artifact-inventory-v1",
        "scientific_calls": 0,
        "files": [
            {"path": name, "size_bytes": (output_path / name).stat().st_size,
             "sha256": file_hash(output_path / name)}
            for name in _DATA_FILES
        ],
    }
    write_pretty_json(output_path / "artifact_inventory.json", inventory)
    _write_hash_sidecar(output_path / "artifact_inventory.json")
    body = {
        "schema_version": "mutaco-paper-canonical-population-evidence-v1",
        "bundle_id": specification["bundle_id"],
        "ingestion_identity": specification["ingestion_identity"],
        "cohort_identity": specification["cohort_identity"],
        "cohort_sha256": specification["cohort_sha256"],
        "target_count": len(targets),
        "methods": list(specification["methods"]),
        "method_configuration_identities": config_ids,
        "source_evidence": audit["sources"],
        "physical_calls": calls,
        "bootstrap_seed": specification["bootstrap_seed"],
        "bootstrap_resamples": specification["bootstrap_resamples"],
        "paired_records_sha256": file_hash(records_path),
        "population_collection_sha256": file_hash(output_path / "population_collection.json"),
        "population_summary_sha256": file_hash(output_path / "population_summary.json"),
        "artifact_inventory_sha256": file_hash(output_path / "artifact_inventory.json"),
        "assessment_coordinates_shared_across_methods_per_target": True,
        "scientific_calls_added_by_consolidation": 0,
    }
    manifest = {
        **body,
        "bundle_identity": stable_hash("paper-canonical-population-evidence-v1", body),
    }
    write_pretty_json(output_path / "manifest.json", manifest)
    _write_hash_sidecar(output_path / "manifest.json")
    return manifest


def verify_consolidated_bundle(path: Path) -> dict[str, Any]:
    """Independently reload hashes, coverage, accounting, and pairing."""

    root = path.resolve()
    manifest = _load_hashed(root / "manifest.json")
    if manifest.get("schema_version") != "mutaco-paper-canonical-population-evidence-v1":
        raise PaperContractError("unsupported canonical population bundle")
    identity_body = {key: value for key, value in manifest.items() if key != "bundle_identity"}
    if manifest.get("bundle_identity") != stable_hash(
        "paper-canonical-population-evidence-v1", identity_body
    ):
        raise PaperContractError("canonical bundle identity mismatch")
    hash_fields = {
        "paired_records.jsonl": "paired_records_sha256",
        "population_collection.json": "population_collection_sha256",
        "population_summary.json": "population_summary_sha256",
        "artifact_inventory.json": "artifact_inventory_sha256",
    }
    for name, field in hash_fields.items():
        artifact = root / name
        if not artifact.is_file() or file_hash(artifact) != manifest.get(field):
            raise PaperContractError(f"canonical artifact hash mismatch: {name}")
        sidecar = artifact.with_suffix(".sha256")
        if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != file_hash(artifact):
            raise PaperContractError(f"canonical artifact sidecar mismatch: {name}")
    inventory = read_json(root / "artifact_inventory.json")
    expected_inventory = [
        {"path": name, "size_bytes": (root / name).stat().st_size,
         "sha256": file_hash(root / name)}
        for name in _DATA_FILES
    ]
    if inventory != {
        "schema_version": "mutaco-paper-canonical-artifact-inventory-v1",
        "scientific_calls": 0,
        "files": expected_inventory,
    }:
        raise PaperContractError("canonical artifact inventory content mismatch")
    rows = [
        json.loads(line) for line in (root / "paired_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    methods = tuple(str(value) for value in manifest["methods"])
    if len(rows) != manifest["target_count"] or [row["target_position"] for row in rows] != list(range(len(rows))):
        raise PaperContractError("canonical compact target ordering is invalid")
    if any(set(row["methods"]) != set(methods) for row in rows):
        raise PaperContractError("canonical compact method coverage is invalid")
    if len({row["target_record_identity"] for row in rows}) != len(rows):
        raise PaperContractError("canonical compact target identities are duplicated")
    for row in rows:
        for method in methods:
            record = row["methods"][method]
            if record.get("status") != "completed":
                raise PaperContractError("canonical compact record is not completed")
            if record.get("method_configuration_identity") != manifest[
                "method_configuration_identities"
            ][method]:
                raise PaperContractError("canonical method configuration identity mismatch")
    compact_records = [row["methods"][method] for row in rows for method in methods]
    calls = _call_totals(compact_records)
    if calls != manifest["physical_calls"]:
        raise PaperContractError("canonical compact call ledger does not match manifest")
    collection = read_json(root / "population_collection.json")
    if (
        collection.get("status") != "completed"
        or collection.get("target_count") != len(rows)
        or collection.get("record_count") != len(compact_records)
        or collection.get("physical_calls") != calls
        or collection.get("scientific_calls_added_by_consolidation") != 0
    ):
        raise PaperContractError("canonical population collection content mismatch")
    summary = read_json(root / "population_summary.json")
    if (
        summary.get("declared_target_count") != len(rows)
        or summary.get("scientific_calls") != 0
        or summary.get("bootstrap", {}).get("seed") != manifest["bootstrap_seed"]
        or summary.get("bootstrap", {}).get("resamples") != manifest["bootstrap_resamples"]
    ):
        raise PaperContractError("canonical population summary metadata mismatch")
    slots = {
        (row["target_record_identity"], method): {
            **row["methods"][method],
            "assessment_context": row["methods"][method].get("assessment_context"),
            "assessment_randomness_evidence": row["methods"][method].get("assessment_randomness_evidence"),
        }
        for row in rows for method in methods
    }
    _validate_assessment_pairing(
        slots, [row["target_record_identity"] for row in rows], methods
    )
    return {
        "schema_version": "mutaco-paper-canonical-population-verification-v1",
        "status": "verified",
        "bundle_identity": manifest["bundle_identity"],
        "target_count": len(rows),
        "record_count": len(compact_records),
        "methods": list(methods),
        "physical_calls": calls,
        "assessment_coordinates_shared_across_methods_per_target": True,
        "scientific_calls": 0,
    }
