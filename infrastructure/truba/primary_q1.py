"""Computation-free validation of the primary Q1 TRUBA preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mutaco import stable_hash
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.protocol.cohort_v2 import validate_cohort_declaration
from experiments.paper.common.validation import PaperContractError


def validate_preparation(path: Path, *, workers: int, maximum_calls: int) -> dict[str, Any]:
    plan = read_json(path)
    if plan.get("schema_version") != "mutaco-nei-q1-preparation-v1":
        raise PaperContractError("unsupported Q1 preparation schema")
    if plan.get("state") != "targets_unconstructed_execution_unauthorized":
        raise PaperContractError("Q1 preparation unexpectedly claims execution readiness")
    body = {key: value for key, value in plan.items() if key != "preparation_identity"}
    if plan.get("preparation_identity") != stable_hash("nei-q1-preparation-v1", body):
        raise PaperContractError("Q1 preparation identity mismatch")
    theta_path = Path(__file__).resolve().parents[2] / str(plan["theta_declaration"])
    if file_hash(theta_path) != plan.get("theta_declaration_sha256"):
        raise PaperContractError("Q1 theta declaration hash mismatch")
    theta = validate_cohort_declaration(read_json(theta_path))
    if theta.get("declaration_hash") != plan.get("theta_declaration_identity"):
        raise PaperContractError("Q1 theta declaration identity mismatch")
    identities = [row["target_identity"] for row in theta["records"]]
    if identities != plan.get("theta_target_identities") or len(identities) != 5:
        raise PaperContractError("Q1 theta target identities mismatch")
    graph_path = Path(__file__).resolve().parents[2] / str(
        plan["graph_qualification_configuration"]
    )
    if file_hash(graph_path) != plan.get("graph_qualification_configuration_sha256"):
        raise PaperContractError("Q1 graph qualification configuration hash mismatch")
    if workers < 1 or workers > 5 or maximum_calls < 1:
        raise PaperContractError("Q1 scheduler geometry is invalid")
    largest_shard = (5 + workers - 1) // workers
    required_cap = largest_shard * 140
    if maximum_calls != required_cap:
        raise PaperContractError("Q1 per-worker cap must exactly match its largest shard")
    return {
        "schema_version": "mutaco-primary-q1-truba-preview-v1",
        "preparation_identity": plan["preparation_identity"],
        "target_count": 5,
        "workers": workers,
        "maximum_calls_per_worker": maximum_calls,
        "global_search_call_cap": 700,
        "target_construction_calls_required": 5,
        "target_construction_authorized": False,
        "qualification_authorized": False,
        "job_submitted": False,
        "scientific_calls_executed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--maximum-calls", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_preparation(
        args.preparation, workers=args.workers, maximum_calls=args.maximum_calls
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
