"""Materialize a zero-call scalar-primary B40 or B100 prequalification smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mutaco import stable_hash
from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.benchmarking.methods.scalar_qlognei import (
    SCALAR_POSTERIOR_MEAN_LOSS_TERMINAL,
    SCALAR_QLOGNEI_POLICY,
    scalar_qlognei_settings_from_document,
)
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)
from experiments.graph_inversion.primary_integration_smoke_driver import (
    scalar_primary_method_identities,
)
from experiments.paper.primary_benchmark import (
    SCALAR_PRIMARY_METHODS,
    PrimaryMethodDeclaration,
    ScalarPrimaryComparisonConfig,
    ScalarPrimaryPopulationDeclaration,
)


PLAN_SCHEMA = "mutaco-scalar-primary-prequalification-smoke-plan-v1"
PLAN_IDENTITY_DOMAIN = "scalar-primary-prequalification-smoke-plan-v1"


def materialize_scalar_smoke(
    *,
    q1_declaration: Path,
    study: Path,
    configuration: Path,
    settings: Path,
    output_root: str,
    budget: int = 40,
) -> tuple[dict, dict]:
    if budget not in {40, 100}:
        raise ValueError("scalar primary smoke budget must be 40 or 100")
    q1 = read_json(q1_declaration)
    study_document = read_json(study)
    settings_document = read_json(settings)
    qlognei_settings = scalar_qlognei_settings_from_document(settings_document)
    target_id = str(q1["target_record_identities"][0])
    simulator = GraphScientificD4Adapter().build_target_simulator(
        read_json(configuration), device=torch.device("cpu"), dtype=torch.float32
    )
    identities = scalar_primary_method_identities(
        configuration, budget=budget, qlognei_settings=qlognei_settings
    )
    methods = tuple(
        PrimaryMethodDeclaration(
            method=method,
            configuration_identity=identities[method],
            search_policy_identifier={
                "mutaco": "canonical_registered_mutaco_profile",
                "scalar_qlognei": SCALAR_QLOGNEI_POLICY,
                "sobol": "single_scrambled_sobol_sequence_per_target_budget_v1",
            }[method],
            terminal_policy_identifier={
                "mutaco": "continuous_posterior_mean_physical_l1_mae",
                "scalar_qlognei": SCALAR_POSTERIOR_MEAN_LOSS_TERMINAL,
                "sobol": "evaluated_best_observed_physical_mae_v1",
            }[method],
        )
        for method in SCALAR_PRIMARY_METHODS
    )
    declaration = ScalarPrimaryPopulationDeclaration(
        campaign_id=f"graph_scalar_primary_prequalification_b{budget}_smoke_v1",
        comparison=ScalarPrimaryComparisonConfig(
            simulator_definition_identity=simulator.capabilities.simulator_identity,
            simulator_configuration_identity=file_hash(configuration),
            assessment_fidelity="sparse-m20-five-force-scientific",
            design_campaign_seed=92021121 if budget == 40 else 92021221,
            inference_campaign_seed=92021122 if budget == 40 else 92021222,
            terminal_campaign_seed=92021123 if budget == 40 else 92021223,
            assessment_campaign_seed=92021124 if budget == 40 else 92021224,
        ),
        study_identity=str(study_document["study_identity"]),
        target_ledger_hash=str(q1["target_ledger_hash"]),
        target_record_identities=(target_id,),
        budget=budget,
        methods=methods,
        output_root=output_root,
    ).document()
    plan_body = {
        "schema_version": PLAN_SCHEMA,
        "state": "declared_unexecuted_submission_unauthorized",
        "classification": (
            "engineering_integration_smoke_ineligible_for_primary_performance"
        ),
        "declaration_identity": declaration["declaration_identity"],
        "study_identity": study_document["study_identity"],
        "target_source": "first_frozen_q1_offcohort_target",
        "target_record_identity": target_id,
        "budget": budget,
        "methods": list(SCALAR_PRIMARY_METHODS),
        "settings_identity": settings_document["settings_identity"],
        "maximum_search_calls": 3 * budget,
        "maximum_assessment_calls": 3,
        "maximum_total_calls": 3 * (budget + 1),
        "primary_benchmark_evidence_eligible": False,
        "scientific_calls_executed": 0,
        "scientific_submission_authorized": False,
        "automatic_retry": False,
    }
    plan = {
        **plan_body,
        "plan_identity": stable_hash(PLAN_IDENTITY_DOMAIN, plan_body),
    }
    return declaration, plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-declaration", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--budget", type=int, choices=(40, 100), default=40)
    parser.add_argument("--declaration-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    declaration, plan = materialize_scalar_smoke(
        q1_declaration=args.q1_declaration,
        study=args.study,
        configuration=args.configuration,
        settings=args.settings,
        output_root=args.output_root,
        budget=args.budget,
    )
    print(json.dumps({"declaration": declaration, "plan": plan}, indent=2))
    if not args.write:
        return
    if not args.confirm_approved:
        raise SystemExit("writing scalar smoke artifacts requires --confirm-approved")
    for path, document in (
        (args.declaration_output, declaration),
        (args.plan_output, plan),
    ):
        if path.exists() or path.with_suffix(".sha256").exists():
            raise SystemExit("scalar smoke declaration outputs are append-only")
        write_pretty_json(path, document)
        path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


if __name__ == "__main__":
    main()
