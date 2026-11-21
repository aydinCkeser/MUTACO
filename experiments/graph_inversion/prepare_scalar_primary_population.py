"""Materialize zero-call scalar-primary Paper-120 population declarations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

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
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark import (
    SCALAR_PRIMARY_METHODS,
    PrimaryMethodDeclaration,
    ScalarPrimaryComparisonConfig,
    ScalarPrimaryPopulationDeclaration,
    scalar_primary_call_preview,
)


PLAN_SCHEMA = "mutaco-scalar-primary-population-plan-v1"
PLAN_IDENTITY_DOMAIN = "scalar-primary-population-plan-v1"
TARGET_COUNT = 120
WORKER_COUNT = 120


def _canonical_targets(cohort_root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    manifest_path = cohort_root / "manifest.json"
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != file_hash(
        manifest_path
    ):
        raise PaperContractError("scalar population cohort manifest hash mismatch")
    manifest = read_json(manifest_path)
    if (
        manifest.get("completed_count") != TARGET_COUNT
        or manifest.get("missing_count") != 0
        or manifest.get("selected_indices") != list(range(TARGET_COUNT))
    ):
        raise PaperContractError("scalar population cohort is not the complete Paper-120 ledger")
    records = manifest.get("records")
    if not isinstance(records, Mapping) or set(records) != {
        str(index) for index in range(TARGET_COUNT)
    }:
        raise PaperContractError("scalar population cohort record map is incomplete")
    identities = tuple(
        str(records[str(index)]["target_record_id"]) for index in range(TARGET_COUNT)
    )
    if len(set(identities)) != TARGET_COUNT:
        raise PaperContractError("scalar population target identities are not unique")
    for index, target_id in enumerate(identities):
        path = cohort_root / f"target_{index:03d}.json"
        target_sidecar = path.with_suffix(".sha256")
        if (
            not target_sidecar.is_file()
            or target_sidecar.read_text(encoding="ascii").strip() != file_hash(path)
        ):
            raise PaperContractError(f"scalar population target hash mismatch: {path}")
        target = read_json(path)
        if (
            target.get("target_index") != index
            or target.get("target_record_id") != target_id
            or target.get("declaration_hash") != manifest.get("declaration_hash")
        ):
            raise PaperContractError("scalar population target differs from cohort manifest")
    return manifest, identities


def materialize_scalar_population(
    *,
    cohort_root: Path,
    study: Path,
    configuration: Path,
    settings: Path,
    output_root: str,
    budget: int,
    execution_revision: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if budget not in {40, 100}:
        raise PaperContractError("scalar population budget must be 40 or 100")
    if execution_revision not in {1, 2, 3, 4}:
        raise PaperContractError(
            "scalar population execution revision must be 1, 2, 3, or 4"
        )
    manifest, target_ids = _canonical_targets(cohort_root)
    manifest_path = cohort_root / "manifest.json"
    study_document = read_json(study)
    settings_document = read_json(settings)
    qlognei_settings = scalar_qlognei_settings_from_document(settings_document)
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
    seed_base = 92021320 if budget == 40 else 92021420
    declaration = ScalarPrimaryPopulationDeclaration(
        campaign_id=f"graph_paper120_scalar_primary_b{budget}_v1",
        comparison=ScalarPrimaryComparisonConfig(
            simulator_definition_identity=simulator.capabilities.simulator_identity,
            simulator_configuration_identity=file_hash(configuration),
            assessment_fidelity="canonical_sparse_m20_one_evaluation_v1",
            design_campaign_seed=seed_base + 1,
            inference_campaign_seed=seed_base + 2,
            terminal_campaign_seed=seed_base + 3,
            assessment_campaign_seed=seed_base + 4,
        ),
        study_identity=str(study_document["study_identity"]),
        target_ledger_hash=file_hash(manifest_path),
        target_record_identities=target_ids,
        budget=budget,
        methods=methods,
        output_root=output_root,
    ).document()
    call_preview = scalar_primary_call_preview(TARGET_COUNT, budget)
    scheduler = (
        {
            "kind": "slurm_array",
            "array_indices": "0-119",
            "maximum_concurrent_tasks": 120,
            "nodes_per_task": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 20,
            "exclusive_node_per_task": True,
            "worker_packing": False,
        }
        if execution_revision == 1
        else {
            "kind": "slurm_array_segments",
            "logical_worker_ranks": "0-119",
            "cluster_max_array_size": 100,
            "physical_arrays": [
                {
                    "task_indices": "0-99",
                    "rank_offset": 0,
                    "logical_worker_ranks": "0-99",
                    "maximum_concurrent_tasks": 100,
                },
                {
                    "task_indices": "0-19",
                    "rank_offset": 100,
                    "logical_worker_ranks": "100-119",
                    "maximum_concurrent_tasks": 20,
                },
            ],
            "collector_dependency": "afterany_both_arrays",
            "nodes_per_task": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 20,
            "exclusive_node_per_task": True,
            "worker_packing": False,
        }
        if execution_revision == 2
        else {
            "kind": "slurm_array_sequential_logical_workers",
            "logical_worker_ranks": "0-119",
            "cluster_max_submit_jobs": 100,
            "array_indices": "0-59",
            "physical_task_count": 60,
            "maximum_concurrent_tasks": 60,
            "logical_workers_per_task": 2,
            "logical_rank_mapping": ["task_id", "task_id+60"],
            "collector_dependency": "afterany_array",
            "transactional_hold_release": True,
            "submitted_job_count": 61,
            "worker_walltime": "04:00:00",
            "nodes_per_task": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 20,
            "exclusive_node_per_task": True,
            "worker_packing": "sequential_two_logical_workers",
        }
        if execution_revision == 3
        else {
            "kind": "slurm_array_delayed_start_sequential_logical_workers",
            "logical_worker_ranks": "0-119",
            "cluster_max_submit_jobs": 100,
            "array_indices": "0-59",
            "physical_task_count": 60,
            "maximum_concurrent_tasks": 60,
            "logical_workers_per_task": 2,
            "logical_rank_mapping": ["task_id", "task_id+60"],
            "collector_dependency": "afterany_array",
            "transactional_delayed_start": True,
            "array_begin_delay": "now+10minutes",
            "rollback_before_begin": True,
            "submitted_job_count": 61,
            "worker_walltime": "04:00:00",
            "nodes_per_task": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 20,
            "exclusive_node_per_task": True,
            "worker_packing": "sequential_two_logical_workers",
        }
    )
    plan_body = {
        "schema_version": PLAN_SCHEMA,
        "state": "declared_unexecuted_submission_unauthorized",
        "classification": "primary_population_performance_candidate",
        "declaration_identity": declaration["declaration_identity"],
        "study_identity": study_document["study_identity"],
        "cohort_manifest": "data/cohorts/paper120_v3/manifest.json",
        "cohort_manifest_sha256": file_hash(manifest_path),
        "cohort_declaration_hash": manifest["declaration_hash"],
        "target_count": TARGET_COUNT,
        "target_positions": list(range(TARGET_COUNT)),
        **(
            {"execution_revision": execution_revision}
            if execution_revision in {2, 3, 4}
            else {}
        ),
        "budget": budget,
        "methods": list(SCALAR_PRIMARY_METHODS),
        "settings_identity": settings_document["settings_identity"],
        "worker_count": WORKER_COUNT,
        "targets_per_worker": 1,
        "scheduler": scheduler,
        "maximum_calls_per_worker": 3 * (budget + 1),
        "maximum_search_calls": call_preview["maximum_search_calls"],
        "maximum_assessment_calls": call_preview["maximum_assessment_calls"],
        "maximum_total_calls": call_preview["maximum_total_calls"],
        "terminal_calls": 0,
        "primary_benchmark_evidence_eligible": True,
        "fallback_provenance_required": True,
        "fallback_provenance_implementation_sha256": (
            "7902560a253003a2bfc3f8dd03d4eb8d92db8c42cf8f13f77850277aa73dc9f1"
        ),
        "scientific_calls_executed": 0,
        "scientific_submission_authorized": False,
        "automatic_retry": False,
    }
    return declaration, {
        **plan_body,
        "plan_identity": stable_hash(PLAN_IDENTITY_DOMAIN, plan_body),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--budget", type=int, choices=(40, 100), required=True)
    parser.add_argument(
        "--execution-revision", type=int, choices=(1, 2, 3, 4), default=1
    )
    parser.add_argument("--declaration-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    declaration, plan = materialize_scalar_population(
        cohort_root=args.cohort_root,
        study=args.study,
        configuration=args.configuration,
        settings=args.settings,
        output_root=args.output_root,
        budget=args.budget,
        execution_revision=args.execution_revision,
    )
    print(json.dumps({"declaration": declaration, "plan": plan}, indent=2))
    if not args.write:
        return
    if not args.confirm_approved:
        raise SystemExit("writing scalar population artifacts requires --confirm-approved")
    for path, document in (
        (args.declaration_output, declaration),
        (args.plan_output, plan),
    ):
        if path.exists() or path.with_suffix(".sha256").exists():
            raise SystemExit("scalar population declaration outputs are append-only")
        write_pretty_json(path, document)
        path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


if __name__ == "__main__":
    main()
