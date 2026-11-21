"""Materialize the post-construction Q1 declaration without scientific calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark.qualification import (
    NEIQualificationDeclaration,
    NEINumericalTier,
)


def materialize(preparation: Path, targets_root: Path, output_root: str) -> dict:
    plan = read_json(preparation)
    records = []
    for index in range(5):
        path = targets_root / f"target_{index:03d}.json"
        sidecar = path.with_suffix(".sha256")
        if not path.is_file() or not sidecar.is_file() or file_hash(path) != sidecar.read_text(encoding="ascii").strip():
            raise PaperContractError("Q1 frozen response ledger is incomplete or corrupt")
        record = read_json(path)
        if record.get("target_identity") != plan["theta_target_identities"][index]:
            raise PaperContractError("Q1 frozen response theta identity mismatch")
        if record.get("declaration_hash") != plan["theta_declaration_identity"]:
            raise PaperContractError("Q1 frozen response declaration mismatch")
        records.append(record)
    raw_tiers = plan["qualification"]["tiers_lowest_effort_first"]
    reference = plan["qualification"]["reference_optimizer"]
    seeds = plan["qualification"]["campaign_seeds"]
    declaration = NEIQualificationDeclaration(
        campaign_id="graph_primary_nei_q1_v1",
        study_identity=plan["study_identity"],
        target_ledger_hash=plan["theta_declaration_identity"],
        target_record_identities=tuple(record["target_record_id"] for record in records),
        tiers=tuple(NEINumericalTier(
            item["tier_id"], item["n_mc"], item["raw_candidates"],
            item["local_starts"], item["optimization_steps"], item["learning_rate"],
        ) for item in raw_tiers),
        snapshot_positions_b40=tuple(plan["qualification"]["snapshot_positions"]["40"]),
        snapshot_positions_b100=tuple(plan["qualification"]["snapshot_positions"]["100"]),
        output_root=output_root,
        trajectory_tier_id=plan["qualification"]["trajectory_tier_id"],
        reference_optimizer=NEINumericalTier(
            reference["tier_id"], reference["n_mc"], reference["raw_candidates"],
            reference["local_starts"], reference["optimization_steps"],
            reference["learning_rate"],
        ),
        design_campaign_seed=seeds["design"],
        inference_campaign_seed=seeds["inference"],
        terminal_campaign_seed=seeds["terminal"],
        reference_campaign_seed=seeds["reference"],
    )
    return declaration.document()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    document = materialize(args.preparation, args.targets_root, args.output_root)
    print(json.dumps({
        "declaration_identity": document["declaration_identity"],
        "maximum_search_calls": document["call_preview"]["maximum_search_calls"],
        "scientific_calls_executed": 0,
    }, indent=2))
    if not args.write:
        return
    if not args.confirm_approved:
        raise SystemExit("Q1 declaration writing requires --confirm-approved")
    if args.output.exists() or args.output.with_suffix(".sha256").exists():
        raise SystemExit("Q1 declaration output is append-only")
    write_pretty_json(args.output, document)
    args.output.with_suffix(".sha256").write_text(file_hash(args.output), encoding="ascii")


if __name__ == "__main__":
    main()
