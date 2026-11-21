"""Freeze the three-call assessment rescue declaration for one failed smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark import (
    PRIMARY_METHODS,
    load_primary_population_declaration,
)


def materialize(
    *,
    primary_declaration: Path,
    decision_freeze: Path,
    configuration: Path,
    output_root: str,
    source_attempt_job_id: int,
    source_commit: str,
) -> dict[str, Any]:
    primary = load_primary_population_declaration(read_json(primary_declaration))
    freeze = read_json(decision_freeze)
    sidecar = decision_freeze.with_suffix(".sha256")
    freeze_sha = file_hash(decision_freeze)
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != freeze_sha:
        raise PaperContractError("assessment rescue freeze hash mismatch")
    decisions = freeze.get("decisions")
    if not isinstance(decisions, list) or tuple(
        item.get("method") for item in decisions if isinstance(item, dict)
    ) != PRIMARY_METHODS:
        raise PaperContractError("assessment rescue freeze method coverage mismatch")
    body = {
        "schema_version": "mutaco-primary-assessment-rescue-declaration-v1",
        "state": "declared_unexecuted_submission_unauthorized",
        "classification": "prequalification_engineering_rescue_not_q1_or_primary_performance_evidence",
        "campaign_id": primary.campaign_id,
        "primary_declaration_identity": primary.document()["declaration_identity"],
        "source_attempt_job_id": source_attempt_job_id,
        "source_attempt_commit": source_commit,
        "decision_freeze_identity": freeze["freeze_identity"],
        "decision_freeze_sha256": freeze_sha,
        "target_id": primary.target_record_identities[0],
        "budget": primary.budget,
        "assessment_replicate": 0,
        "methods": [
            {
                "method": decision["method"],
                "recommended_theta": decision["recommended_theta"],
            }
            for decision in decisions
        ],
        "configuration_sha256": file_hash(configuration),
        "output_root": output_root,
        "search_calls": 0,
        "maximum_assessment_calls": 3,
        "automatic_retry": False,
        "scientific_submission_authorized": False,
        "primary_benchmark_evidence_eligible": False,
    }
    return {
        **body,
        "declaration_identity": stable_hash(
            "primary-assessment-rescue-declaration-v1", body
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-declaration", type=Path, required=True)
    parser.add_argument("--decision-freeze", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-attempt-job-id", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = materialize(
        primary_declaration=args.primary_declaration,
        decision_freeze=args.decision_freeze,
        configuration=args.configuration,
        output_root=args.output_root,
        source_attempt_job_id=args.source_attempt_job_id,
        source_commit=args.source_commit,
    )
    write_pretty_json(args.output, document)
    args.output.with_suffix(".sha256").write_text(file_hash(args.output), encoding="ascii")
    print(document["declaration_identity"])


if __name__ == "__main__":
    main()
