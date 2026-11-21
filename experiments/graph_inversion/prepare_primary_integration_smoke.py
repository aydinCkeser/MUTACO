"""Materialize the approval-gated prequalification primary integration smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mutaco import stable_hash
from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)
from experiments.graph_inversion.primary_integration_smoke_driver import (
    SMOKE_NEI_TIER_ID,
    primary_smoke_method_identities,
)
from experiments.paper.primary_benchmark import (
    PRIMARY_METHODS,
    PrimaryComparisonConfig,
    PrimaryMethodDeclaration,
    PrimaryPopulationDeclaration,
)


def materialize(
    *, q1_declaration: Path, configuration: Path, output_root: str
) -> tuple[dict, dict]:
    q1 = read_json(q1_declaration)
    target_id = str(q1["target_record_identities"][0])
    simulator = GraphScientificD4Adapter().build_target_simulator(
        read_json(configuration), device=torch.device("cpu"), dtype=torch.float32
    )
    identities = primary_smoke_method_identities(configuration)
    methods = tuple(
        PrimaryMethodDeclaration(
            method=method,
            configuration_identity=identities[method],
            search_policy_identifier={
                "mutaco": "canonical_registered_mutaco_profile",
                "nei_bo": "composite_mc_nei_physical_mae_v1",
                "sobol": "single_scrambled_sobol_sequence_per_target_budget_v1",
            }[method],
            terminal_policy_identifier=(
                "evaluated_best_observed_physical_mae_v1"
                if method == "sobol"
                else "continuous_posterior_mean_physical_l1_mae"
            ),
        )
        for method in PRIMARY_METHODS
    )
    declaration = PrimaryPopulationDeclaration(
        campaign_id="graph_primary_prequalification_b40_integration_smoke_v1",
        comparison=PrimaryComparisonConfig(
            simulator_definition_identity=simulator.capabilities.simulator_identity,
            simulator_configuration_identity=file_hash(configuration),
            assessment_fidelity="sparse-m20-five-force-scientific",
            design_campaign_seed=92020921,
            inference_campaign_seed=92020922,
            terminal_campaign_seed=92020923,
            assessment_campaign_seed=92020924,
        ),
        study_identity=str(q1["study_identity"]),
        target_ledger_hash=str(q1["target_ledger_hash"]),
        target_record_identities=(target_id,),
        budget=40,
        methods=methods,
        output_root=output_root,
    ).document()
    plan_body = {
        "schema_version": "mutaco-primary-prequalification-smoke-plan-v1",
        "state": "declared_unexecuted_submission_unauthorized",
        "classification": "engineering_integration_smoke_ineligible_for_q1_tier_selection_and_primary_performance",
        "declaration_identity": declaration["declaration_identity"],
        "target_source": "first_frozen_q1_offcohort_target",
        "target_record_identity": target_id,
        "budget": 40,
        "methods": list(PRIMARY_METHODS),
        "provisional_nei_tier_id": SMOKE_NEI_TIER_ID,
        "production_nei_tier_selected": False,
        "maximum_search_calls": 120,
        "maximum_assessment_calls": 3,
        "maximum_total_calls": 123,
        "historical_or_future_q1_evidence_eligible": False,
        "primary_benchmark_evidence_eligible": False,
        "scientific_calls_executed": 0,
        "scientific_submission_authorized": False,
        "automatic_retry": False,
    }
    plan = {
        **plan_body,
        "plan_identity": stable_hash(
            "primary-prequalification-smoke-plan-v1", plan_body
        ),
    }
    return declaration, plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-declaration", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--declaration-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    declaration, plan = materialize(
        q1_declaration=args.q1_declaration,
        configuration=args.configuration,
        output_root=args.output_root,
    )
    print(json.dumps({
        "declaration_identity": declaration["declaration_identity"],
        "plan_identity": plan["plan_identity"],
        "maximum_total_calls": 123,
        "scientific_calls_executed": 0,
        "job_submitted": False,
    }, indent=2, sort_keys=True))
    if not args.write:
        return
    if not args.confirm_approved:
        raise SystemExit("smoke declaration writing requires --confirm-approved")
    for path, document in (
        (args.declaration_output, declaration),
        (args.plan_output, plan),
    ):
        if path.exists() or path.with_suffix(".sha256").exists():
            raise SystemExit("smoke declaration outputs are append-only")
        write_pretty_json(path, document)
        path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


if __name__ == "__main__":
    main()
