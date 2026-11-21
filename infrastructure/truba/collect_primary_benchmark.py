"""Zero-call collectors and compact exporters for primary Q1/Q2 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mutaco import stable_hash
from experiments.paper.common.population_v3 import shard_indices
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark.analysis import (
    endpoint_results,
    paired_endpoint_differences,
    runtime_results,
    trajectory_results,
)
from experiments.paper.primary_benchmark.protocol import (
    PRIMARY_DECLARATION_SCHEMA,
    PrimaryPopulationDeclaration,
    load_primary_population_declaration,
)
from experiments.paper.primary_benchmark.scalar_protocol import (
    SCALAR_PRIMARY_DECLARATION_SCHEMA,
    ScalarPrimaryPopulationDeclaration,
    load_scalar_primary_population_declaration,
)
from experiments.paper.primary_benchmark.assessment_rescue import (
    load_primary_assessment_rescue_declaration,
)
from experiments.paper.primary_benchmark.qualification import (
    NEIQualificationDeclaration,
    load_nei_qualification_declaration,
    select_nei_numerical_tier,
)
from experiments.paper.primary_benchmark.qualification_execution import (
    append_hashed_json,
    load_hashed_json,
)


def collect_q1(
    *,
    declaration: NEIQualificationDeclaration,
    workers_root: Path,
    output: Path,
    workers: int,
    expected_commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise PaperContractError("Q1 collection output must be append-only")
    identity = declaration.document()["declaration_identity"]
    observations: list[Mapping[str, Any]] = []
    records = []
    calls = {"maximum_calls": 0, "attempted_calls": 0, "successful_calls": 0, "failed_calls": 0}
    errors = []
    for rank, positions in enumerate(shard_indices(5, rank=i, workers=workers) for i in range(workers)):
        root = workers_root / f"worker-{rank:02d}"
        try:
            start = load_hashed_json(root / "worker_start.json")
            terminal = load_hashed_json(root / "worker_terminal.json")
            if start.get("source_commit") != expected_commit or terminal.get("source_commit") != expected_commit:
                raise PaperContractError("Q1 worker source commit mismatch")
            if start.get("declaration_identity") != identity or terminal.get("declaration_identity") != identity:
                raise PaperContractError("Q1 worker declaration mismatch")
            if start.get("preview", {}).get("target_positions") != list(positions):
                raise PaperContractError("Q1 worker shard mismatch")
            if terminal.get("status") != "completed":
                raise PaperContractError("Q1 worker is incomplete")
            ledger = terminal.get("physical_calls", {})
            for key in calls:
                calls[key] += int(ledger[key])
            for position in positions:
                target = declaration.target_record_identities[position]
                for budget in (40, 100):
                    record = load_hashed_json(
                        root / "records" / f"target-{position:02d}-b{budget}.json"
                    )
                    if record.get("target_record_identity") != target or record.get("budget") != budget:
                        raise PaperContractError("Q1 trajectory slot mismatch")
                    records.append(record)
                    observations.extend(record["result"]["tier_observations"])
        except Exception as exc:
            errors.append({"rank": rank, "failure_type": type(exc).__name__, "explanation": str(exc)})
    if len(records) != 10 or calls["attempted_calls"] != 700:
        errors.append({"failure_type": "Q1CoverageError", "explanation": "Q1 requires ten trajectories and exactly 700 calls"})
    body = {
        "schema_version": "mutaco-nei-q1-global-collection-v1",
        "declaration_identity": identity,
        "source_commit": expected_commit,
        "record_count": len(records),
        "physical_calls": calls,
        "errors": errors,
        "status": "failed" if errors else "completed",
        "scientific_calls_added_by_collection": 0,
    }
    collection = {**body, "collection_identity": stable_hash("nei-q1-global-collection-v1", body)}
    append_hashed_json(output / "collection.json", collection)
    if errors:
        raise PaperContractError("Q1 collection failed; evidence was preserved")
    decision = select_nei_numerical_tier(declaration, observations)
    append_hashed_json(output / "qualification_result.json", decision)
    return decision


PrimaryCollectionDeclaration = (
    PrimaryPopulationDeclaration | ScalarPrimaryPopulationDeclaration
)


def load_primary_collection_declaration(
    document: Mapping[str, Any],
) -> PrimaryCollectionDeclaration:
    """Dispatch a primary declaration without weakening either schema."""

    schema = document.get("schema_version")
    if schema == PRIMARY_DECLARATION_SCHEMA:
        return load_primary_population_declaration(document)
    if schema == SCALAR_PRIMARY_DECLARATION_SCHEMA:
        return load_scalar_primary_population_declaration(document)
    raise PaperContractError("primary collection declaration schema is unsupported")


def collect_primary(
    *,
    declaration: PrimaryCollectionDeclaration,
    workers_root: Path,
    output: Path,
    workers: int,
    expected_commit: str,
) -> dict[str, Any]:
    """Collect Q2 or production records and emit deterministic compact tables."""

    if output.exists():
        raise PaperContractError("primary collection output must be append-only")
    identity = declaration.document()["declaration_identity"]
    records: list[Mapping[str, Any]] = []
    calls = {"maximum_calls": 0, "attempted_calls": 0, "successful_calls": 0, "failed_calls": 0}
    errors = []
    shards = tuple(
        shard_indices(len(declaration.target_record_identities), rank=rank, workers=workers)
        for rank in range(workers)
    )
    for rank, positions in enumerate(shards):
        root = workers_root / f"worker-{rank:02d}"
        try:
            start = load_hashed_json(root / "worker_start.json")
            terminal = load_hashed_json(root / "worker_terminal.json")
            if start.get("source_commit") != expected_commit or terminal.get("source_commit") != expected_commit:
                raise PaperContractError("primary worker source commit mismatch")
            if start.get("declaration_identity") != identity or terminal.get("declaration_identity") != identity:
                raise PaperContractError("primary worker declaration mismatch")
            if start.get("preview", {}).get("target_positions") != list(positions):
                raise PaperContractError("primary worker shard mismatch")
            if terminal.get("status") != "completed":
                raise PaperContractError("primary worker is incomplete")
            for key in calls:
                calls[key] += int(terminal["physical_calls"][key])
            for position in positions:
                root_target = root / "targets" / f"target-{position:03d}"
                execution = load_hashed_json(root_target / "target_execution.json")
                if execution.get("target_id") != declaration.target_record_identities[position]:
                    raise PaperContractError("primary collected target mismatch")
                records.extend(execution["records"])
        except Exception as exc:
            errors.append({"rank": rank, "failure_type": type(exc).__name__, "explanation": str(exc)})
    expected_records = len(declaration.target_record_identities) * 3
    expected_calls = declaration.document()["call_preview"]["maximum_total_calls"]
    identities = [str(row.get("record_identity", "")) for row in records]
    if len(records) != expected_records or len(set(identities)) != expected_records:
        errors.append({"failure_type": "PrimaryCoverageError", "explanation": "primary record coverage is incomplete or duplicated"})
    if calls["attempted_calls"] != expected_calls:
        errors.append({"failure_type": "PrimaryCallCapError", "explanation": "primary global call ledger differs from declaration"})
    collection_schema = str(getattr(
        declaration,
        "collection_schema",
        "mutaco-primary-global-collection-v1",
    ))
    collection_domain = str(getattr(
        declaration,
        "collection_identity_domain",
        "primary-global-collection-v1",
    ))
    body = {
        "schema_version": collection_schema,
        "declaration_identity": identity,
        "source_commit": expected_commit,
        "record_count": len(records),
        "physical_calls": calls,
        "errors": errors,
        "status": "failed" if errors else "completed",
        "scientific_calls_added_by_collection": 0,
    }
    collection = {
        **body,
        "collection_identity": stable_hash(collection_domain, body),
    }
    append_hashed_json(output / "collection.json", collection)
    if errors:
        raise PaperContractError("primary collection failed; evidence was preserved")
    compact = {
        "schema_version": str(getattr(
            declaration,
            "compact_export_schema",
            "mutaco-primary-compact-export-v1",
        )),
        "collection_identity": collection["collection_identity"],
        "records": list(records),
        "endpoint_results": list(endpoint_results(records)),
        "trajectory_results": list(trajectory_results(records)),
        "paired_endpoint_differences": list(paired_endpoint_differences(records)),
        "runtime_results": list(runtime_results(records)),
        "excluded": ["monte_carlo_sample_tensors", "gp_model_tensors"],
        "scientific_calls_added_by_export": 0,
    }
    append_hashed_json(output / "compact_export.json", compact)
    return collection


def collect_assessment_rescue(
    *,
    declaration: Mapping[str, Any],
    workers_root: Path,
    output: Path,
    expected_commit: str,
) -> dict[str, Any]:
    """Collect the one-worker, three-record assessment-only rescue."""

    if output.exists():
        raise PaperContractError("assessment rescue collection must be append-only")
    root = workers_root / "worker-00"
    terminal = load_hashed_json(root / "rescue_terminal.json")
    result = load_hashed_json(root / "rescue_result.json")
    identity = declaration["declaration_identity"]
    if (
        terminal.get("source_commit") != expected_commit
        or result.get("source_commit") != expected_commit
        or terminal.get("declaration_identity") != identity
        or result.get("declaration_identity") != identity
        or terminal.get("decision_freeze_identity")
        != declaration["decision_freeze_identity"]
        or result.get("decision_freeze_identity")
        != declaration["decision_freeze_identity"]
        or terminal.get("status") != "completed"
        or result.get("status") != "completed"
        or terminal.get("physical_calls")
        != {
            "maximum_calls": 3,
            "attempted_calls": 3,
            "successful_calls": 3,
            "failed_calls": 0,
        }
    ):
        raise PaperContractError("assessment rescue worker evidence is invalid")
    records = result.get("records")
    if (
        not isinstance(records, list)
        or tuple(record.get("method") for record in records) != ("mutaco", "nei_bo", "sobol")
        or len({record.get("record_identity") for record in records}) != 3
    ):
        raise PaperContractError("assessment rescue record coverage is invalid")
    body = {
        "schema_version": "mutaco-primary-assessment-rescue-collection-v1",
        "source_commit": expected_commit,
        "declaration_identity": identity,
        "source_attempt_job_id": declaration["source_attempt_job_id"],
        "decision_freeze_identity": declaration["decision_freeze_identity"],
        "record_count": 3,
        "records": records,
        "physical_calls": terminal["physical_calls"],
        "search_rerun": False,
        "status": "completed",
        "scientific_calls_added_by_collection": 0,
    }
    collection = {
        **body,
        "collection_identity": stable_hash(
            "primary-assessment-rescue-collection-v1", body
        ),
    }
    append_hashed_json(output / "collection.json", collection)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    q1 = subcommands.add_parser("q1")
    q1.add_argument("--declaration", type=Path, required=True)
    q1.add_argument("--workers-root", type=Path, required=True)
    q1.add_argument("--output", type=Path, required=True)
    q1.add_argument("--workers", type=int, required=True)
    q1.add_argument("--expected-commit", required=True)
    primary = subcommands.add_parser("primary")
    primary.add_argument("--declaration", type=Path, required=True)
    primary.add_argument("--workers-root", type=Path, required=True)
    primary.add_argument("--output", type=Path, required=True)
    primary.add_argument("--workers", type=int, required=True)
    primary.add_argument("--expected-commit", required=True)
    rescue = subcommands.add_parser("assessment-rescue")
    rescue.add_argument("--declaration", type=Path, required=True)
    rescue.add_argument("--workers-root", type=Path, required=True)
    rescue.add_argument("--output", type=Path, required=True)
    rescue.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if args.command == "q1":
        declaration = load_nei_qualification_declaration(
            json.loads(args.declaration.read_text(encoding="utf-8"))
        )
        result = collect_q1(
            declaration=declaration,
            workers_root=args.workers_root,
            output=args.output,
            workers=args.workers,
            expected_commit=args.expected_commit,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "primary":
        declaration = load_primary_collection_declaration(
            json.loads(args.declaration.read_text(encoding="utf-8"))
        )
        result = collect_primary(
            declaration=declaration,
            workers_root=args.workers_root,
            output=args.output,
            workers=args.workers,
            expected_commit=args.expected_commit,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "assessment-rescue":
        declaration = load_primary_assessment_rescue_declaration(
            json.loads(args.declaration.read_text(encoding="utf-8"))
        )
        result = collect_assessment_rescue(
            declaration=declaration,
            workers_root=args.workers_root,
            output=args.output,
            expected_commit=args.expected_commit,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
