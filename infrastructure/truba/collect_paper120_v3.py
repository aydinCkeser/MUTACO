"""Strict computation-free collection of unified Paper-v3 worker shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.paper.analysis.aggregate import PaperRecordSet, summarize_population_v3
from experiments.paper.protocol.method_append import load_method_record, record_path
from experiments.paper.common.population_v3 import (
    load_population_declaration_v3,
    population_call_preview,
    shard_indices,
)
from experiments.paper.common.validation import PaperContractError
from infrastructure.truba.paper120_v3 import BOOTSTRAP_SEED, WORKER_COUNT


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


def _write_hashed(path: Path, document: Mapping[str, Any]) -> None:
    write_pretty_json(path, dict(document))
    path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


def _coverage_errors(
    rows: list[Mapping[str, Any]], targets: tuple[str, ...], methods: tuple[str, ...]
) -> list[dict[str, str]]:
    expected = {(target, method) for target in targets for method in methods}
    observed = [(str(row["target_record_identity"]), str(row["method"])) for row in rows]
    errors = []
    if set(observed) != expected or len(observed) != len(expected):
        errors.append({
            "failure_type": "PopulationCoverageError",
            "explanation": "global target-method coverage is incomplete or duplicated",
        })
    return errors


def collect(
    *, root: Path, declaration_path: Path, workers_root: Path,
    output: Path, expected_commit: str, workers: int = WORKER_COUNT,
) -> dict[str, Any]:
    """Validate all shards and write a summary only for a valid population."""

    if output.exists():
        raise PaperContractError("append-only Paper-v3 collection output already exists")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise PaperContractError("Paper-v3 collection requires a positive worker count")
    repository_root = root.resolve()
    declaration_file = declaration_path.resolve()
    try:
        declaration_file.relative_to(repository_root)
    except ValueError as exc:
        raise PaperContractError("Paper-v3 declaration is outside the pinned checkout") from exc
    declaration_sha256 = file_hash(declaration_file)
    declaration = load_population_declaration_v3(read_json(declaration_file))
    methods = tuple(method.method for method in declaration.methods)
    target_count = len(declaration.target_record_identities)
    shards = tuple(
        shard_indices(target_count, rank=rank, workers=workers)
        for rank in range(workers)
    )
    if any(not positions for positions in shards):
        raise PaperContractError("Paper-v3 worker count cannot exceed target count")
    global_call_cap = population_call_preview(target_count, declaration.methods)[
        "maximum_total_calls"
    ]
    rows: list[dict[str, Any]] = []
    worker_documents: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_calls = {key: 0 for key in ("maximum_calls", "attempted_calls", "successful_calls", "failed_calls")}
    for rank, positions in enumerate(shards):
        worker = workers_root / f"worker-{rank:02d}"
        worker_call_cap = population_call_preview(len(positions), declaration.methods)[
            "maximum_total_calls"
        ]
        worker_slot_count = len(positions) * len(methods)
        try:
            start = _load_hashed(worker / "worker_start.json")
            terminal = _load_hashed(worker / "worker_terminal.json")
            attempt = _load_hashed(worker / "population_execution_attempt.json")
            local_collection = _load_hashed(worker / "population_collection.json")
            preview = start.get("preview", {})
            if start.get("source_commit") != expected_commit or terminal.get("source_commit") != expected_commit:
                raise PaperContractError("worker source commit mismatch")
            if start.get("declaration_sha256") != declaration_sha256 or terminal.get("declaration_sha256") != declaration_sha256:
                raise PaperContractError("worker declaration hash mismatch")
            if preview.get("rank") != rank or preview.get("workers") != workers:
                raise PaperContractError("worker rank geometry mismatch")
            if preview.get("target_positions") != list(positions):
                raise PaperContractError("worker target positions mismatch")
            if preview.get("maximum_calls_argument") != worker_call_cap:
                raise PaperContractError("worker call-cap argument mismatch")
            if terminal.get("status") != "completed" or attempt.get("status") != "completed":
                raise PaperContractError("worker did not reach an operationally completed terminal state")
            calls = attempt.get("physical_calls")
            if not isinstance(calls, Mapping) or calls.get("maximum_calls") != worker_call_cap:
                raise PaperContractError("worker physical-call ledger is invalid")
            if calls.get("attempted_calls", -1) > worker_call_cap:
                raise PaperContractError("worker exceeded its physical-call cap")
            if calls.get("attempted_calls") != calls.get("successful_calls", 0) + calls.get("failed_calls", 0):
                raise PaperContractError("worker physical-call ledger does not balance")
            if terminal.get("physical_calls") != calls:
                raise PaperContractError("worker terminal and execution call ledgers differ")
            if not local_collection.get("complete") or local_collection.get("declared_slot_count") != worker_slot_count:
                raise PaperContractError("worker target-method collection is incomplete")
            expected_targets = tuple(declaration.target_record_identities[index] for index in positions)
            worker_rows = []
            for target in expected_targets:
                for method in methods:
                    record = load_method_record(record_path(worker, target, method))
                    if record is None:
                        raise PaperContractError("worker is missing a declared method record")
                    worker_rows.append(record)
            if len({(row["target_record_identity"], row["method"]) for row in worker_rows}) != worker_slot_count:
                raise PaperContractError("worker contains duplicate target-method slots")
            rows.extend(worker_rows)
            for key in total_calls:
                total_calls[key] += int(calls[key])
            worker_documents.append({
                "rank": rank,
                "target_positions": list(positions),
                "terminal_status": terminal["status"],
                "completed_slots": sum(row["status"] == "completed" for row in worker_rows),
                "failed_slots": sum(row["status"] == "failed" for row in worker_rows),
                "physical_calls": dict(calls),
            })
        except Exception as exc:
            errors.append({"rank": rank, "failure_type": type(exc).__name__, "explanation": str(exc)})

    expected_slots = {(target, method) for target in declaration.target_record_identities for method in methods}
    observed_slots = {(row["target_record_identity"], row["method"]) for row in rows}
    errors.extend(_coverage_errors(rows, declaration.target_record_identities, methods))
    if total_calls["maximum_calls"] != global_call_cap:
        errors.append({
            "failure_type": "GlobalCallCapError",
            "explanation": "collected worker maximum-call ledgers differ from the declaration",
        })

    output.mkdir(parents=True)
    validation = {
        "schema_version": "mutaco-paper-v3-global-collection-v1",
        "status": "failed" if errors else "completed",
        "source_commit": expected_commit,
        "declaration_identity": declaration.document()["declaration_identity"],
        "declaration_sha256": declaration_sha256,
        "declared_target_count": target_count,
        "declared_method_count": len(methods),
        "declared_slot_count": len(expected_slots),
        "collected_slot_count": len(observed_slots),
        "worker_count": workers,
        "workers": worker_documents,
        "physical_calls": total_calls,
        "errors": errors,
        "scientific_calls_added_by_collection": 0,
    }
    _write_hashed(output / "population_collection.json", validation)
    if errors:
        raise PaperContractError("Paper-v3 collection failed; preserved population_collection.json")

    summary = summarize_population_v3(
        PaperRecordSet(workers_root, tuple(rows)),
        declared_target_identities=declaration.target_record_identities,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=10_000,
    )
    _write_hashed(output / "population_summary.json", summary)
    inventory = {
        "schema_version": "mutaco-paper-v3-global-inventory-v1",
        "scientific_calls": 0,
        "files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        ],
    }
    _write_hashed(output / "artifact_inventory.json", inventory)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    collect(
        root=args.root,
        declaration_path=args.declaration,
        workers_root=args.workers_root,
        output=args.output,
        expected_commit=args.expected_commit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
