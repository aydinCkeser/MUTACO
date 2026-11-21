"""Preview or execute the capped graph primary integration smoke worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from mutaco import (
    ScientificIdentity,
    SeedDeriver,
    SimulationContext,
    checked_evaluate,
    stable_hash,
)
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.methods.scalar_qlognei import (
    scalar_qlognei_settings_from_document,
)
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)
from experiments.graph_inversion.primary_integration_smoke_driver import (
    GraphPrimaryIntegrationSmokeDriver,
)
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark import (
    execute_primary_worker,
    generate_primary_sobol_design,
    load_primary_population_declaration,
    load_scalar_primary_population_declaration,
    primary_worker_preview,
)


def _validate_plan(plan: Mapping[str, Any], declaration_identity: str) -> None:
    schema = plan.get("schema_version")
    identity_domain = {
        "mutaco-primary-prequalification-smoke-plan-v1": (
            "primary-prequalification-smoke-plan-v1"
        ),
        "mutaco-scalar-primary-prequalification-smoke-plan-v1": (
            "scalar-primary-prequalification-smoke-plan-v1"
        ),
        "mutaco-scalar-primary-population-plan-v1": (
            "scalar-primary-population-plan-v1"
        ),
    }.get(schema)
    if (
        identity_domain is None
        or plan.get("state") != "declared_unexecuted_submission_unauthorized"
        or plan.get("declaration_identity") != declaration_identity
        or plan.get("scientific_submission_authorized") is not False
    ):
        raise PaperContractError("primary smoke plan is invalid")
    if schema == "mutaco-scalar-primary-prequalification-smoke-plan-v1":
        budget = plan.get("budget")
        if (
            budget not in {40, 100}
            or plan.get("methods") != ["mutaco", "scalar_qlognei", "sobol"]
            or plan.get("maximum_search_calls") != 3 * budget
            or plan.get("maximum_assessment_calls") != 3
            or plan.get("maximum_total_calls") != 3 * (budget + 1)
            or plan.get("primary_benchmark_evidence_eligible") is not False
            or not str(plan.get("settings_identity", "")).startswith(
                "scalar-qlognei-primary-settings-v1-"
            )
            or plan.get("automatic_retry") is not False
        ):
            raise PaperContractError("scalar primary smoke plan is invalid")
    if schema == "mutaco-primary-prequalification-smoke-plan-v1" and (
        plan.get("maximum_total_calls") != 123
        or plan.get("primary_benchmark_evidence_eligible") is not False
    ):
        raise PaperContractError("primary smoke plan is invalid")
    if schema == "mutaco-scalar-primary-population-plan-v1":
        budget = plan.get("budget")
        expected_total = 120 * 3 * (int(budget) + 1) if budget in {40, 100} else None
        scheduler = plan.get("scheduler")
        execution_revision = plan.get("execution_revision", 1)
        scheduler_valid = False
        if isinstance(scheduler, Mapping) and execution_revision == 1:
            scheduler_valid = (
                scheduler.get("kind") == "slurm_array"
                and scheduler.get("array_indices") == "0-119"
                and scheduler.get("maximum_concurrent_tasks") == 120
            )
        if isinstance(scheduler, Mapping) and execution_revision == 2:
            scheduler_valid = (
                scheduler.get("kind") == "slurm_array_segments"
                and scheduler.get("logical_worker_ranks") == "0-119"
                and scheduler.get("cluster_max_array_size") == 100
                and scheduler.get("physical_arrays")
                == [
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
                ]
                and scheduler.get("collector_dependency") == "afterany_both_arrays"
            )
        if isinstance(scheduler, Mapping) and execution_revision == 3:
            scheduler_valid = (
                scheduler.get("kind") == "slurm_array_sequential_logical_workers"
                and scheduler.get("logical_worker_ranks") == "0-119"
                and scheduler.get("cluster_max_submit_jobs") == 100
                and scheduler.get("array_indices") == "0-59"
                and scheduler.get("physical_task_count") == 60
                and scheduler.get("maximum_concurrent_tasks") == 60
                and scheduler.get("logical_workers_per_task") == 2
                and scheduler.get("logical_rank_mapping")
                == ["task_id", "task_id+60"]
                and scheduler.get("collector_dependency") == "afterany_array"
                and scheduler.get("transactional_hold_release") is True
                and scheduler.get("submitted_job_count") == 61
                and scheduler.get("worker_walltime") == "04:00:00"
            )
        if isinstance(scheduler, Mapping) and execution_revision == 4:
            scheduler_valid = (
                scheduler.get("kind")
                == "slurm_array_delayed_start_sequential_logical_workers"
                and scheduler.get("logical_worker_ranks") == "0-119"
                and scheduler.get("cluster_max_submit_jobs") == 100
                and scheduler.get("array_indices") == "0-59"
                and scheduler.get("physical_task_count") == 60
                and scheduler.get("maximum_concurrent_tasks") == 60
                and scheduler.get("logical_workers_per_task") == 2
                and scheduler.get("logical_rank_mapping")
                == ["task_id", "task_id+60"]
                and scheduler.get("collector_dependency") == "afterany_array"
                and scheduler.get("transactional_delayed_start") is True
                and scheduler.get("array_begin_delay") == "now+10minutes"
                and scheduler.get("rollback_before_begin") is True
                and scheduler.get("submitted_job_count") == 61
                and scheduler.get("worker_walltime") == "04:00:00"
            )
        if (
            budget not in {40, 100}
            or execution_revision not in {1, 2, 3, 4}
            or plan.get("classification") != "primary_population_performance_candidate"
            or plan.get("target_count") != 120
            or plan.get("target_positions") != list(range(120))
            or plan.get("methods") != ["mutaco", "scalar_qlognei", "sobol"]
            or plan.get("worker_count") != 120
            or plan.get("targets_per_worker") != 1
            or plan.get("maximum_calls_per_worker") != 3 * (int(budget) + 1)
            or plan.get("maximum_search_calls") != 120 * 3 * int(budget)
            or plan.get("maximum_assessment_calls") != 360
            or plan.get("maximum_total_calls") != expected_total
            or plan.get("primary_benchmark_evidence_eligible") is not True
            or plan.get("fallback_provenance_required") is not True
            or not scheduler_valid
            or scheduler.get("nodes_per_task") != 1
            or scheduler.get("tasks_per_node") != 1
            or scheduler.get("cpus_per_task") != 20
            or scheduler.get("exclusive_node_per_task") is not True
            or (
                scheduler.get("worker_packing") is not False
                if execution_revision in {1, 2}
                else scheduler.get("worker_packing")
                != "sequential_two_logical_workers"
            )
            or plan.get("automatic_retry") is not False
        ):
            raise PaperContractError("scalar primary population plan is invalid")
    body = {key: value for key, value in plan.items() if key != "plan_identity"}
    if plan.get("plan_identity") != stable_hash(
        identity_domain, body
    ):
        raise PaperContractError("primary smoke plan identity mismatch")


def _load_smoke_target(
    targets_root: Path,
    *,
    target_id: str,
    target_ledger_hash: str,
) -> dict[str, Mapping[str, Any]]:
    selected = []
    for path in sorted(targets_root.glob("target_*.json")):
        sidecar = path.with_suffix(".sha256")
        if not sidecar.is_file() or file_hash(path) != sidecar.read_text(
            encoding="ascii"
        ).strip():
            raise PaperContractError(f"primary smoke target hash mismatch: {path}")
        record = read_json(path)
        if record.get("target_record_id") == target_id:
            selected.append(record)
    if len(selected) != 1:
        raise PaperContractError("primary smoke target coverage is not exactly one")
    record = selected[0]
    if record.get("declaration_hash") != target_ledger_hash:
        raise PaperContractError("primary smoke target belongs to the wrong ledger")
    return {target_id: record}


def _load_population_targets(
    targets_root: Path,
    *,
    target_ids: Sequence[str],
    target_ledger_hash: str,
) -> dict[str, Mapping[str, Any]]:
    manifest_path = targets_root / "manifest.json"
    sidecar = manifest_path.with_suffix(".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip() != file_hash(manifest_path)
        or file_hash(manifest_path) != target_ledger_hash
    ):
        raise PaperContractError("scalar primary population manifest hash mismatch")
    manifest = read_json(manifest_path)
    records = manifest.get("records")
    if (
        len(target_ids) != 120
        or manifest.get("completed_count") != 120
        or manifest.get("missing_count") != 0
        or manifest.get("selected_indices") != list(range(120))
        or not isinstance(records, Mapping)
    ):
        raise PaperContractError("scalar primary population manifest is incomplete")
    loaded: dict[str, Mapping[str, Any]] = {}
    for index, expected_id in enumerate(target_ids):
        path = targets_root / f"target_{index:03d}.json"
        target_sidecar = path.with_suffix(".sha256")
        if (
            not target_sidecar.is_file()
            or target_sidecar.read_text(encoding="ascii").strip() != file_hash(path)
        ):
            raise PaperContractError(f"scalar primary population target hash mismatch: {path}")
        record = read_json(path)
        manifest_record = records.get(str(index))
        if (
            not isinstance(manifest_record, Mapping)
            or manifest_record.get("target_record_id") != expected_id
            or record.get("target_index") != index
            or record.get("target_record_id") != expected_id
            or record.get("declaration_hash") != manifest.get("declaration_hash")
        ):
            raise PaperContractError("scalar primary population target coverage mismatch")
        loaded[expected_id] = record
    return loaded


def _simulation_context(context: Mapping[str, Any]) -> SimulationContext:
    """Restore the complete primary search or assessment simulator address."""

    return SimulationContext(
        str(context["family_identity"]),
        str(context["purpose"]),
        int(context["logical_index"]),
        0,
        int(context["evaluation_seed"]),
    )


def _evaluate_graph_simulator(
    simulator: Any,
    theta: Sequence[float],
    context: Mapping[str, Any],
) -> list[float]:
    result = checked_evaluate(
        simulator,
        torch.tensor(theta, dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
        context=_simulation_context(context),
    )
    return result.values.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qlognei-settings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-calls", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--torch-intraop-threads", type=int, required=True)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validate-all-shards", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    declaration_document = read_json(args.declaration)
    if declaration_document.get("schema_version") == (
        "mutaco-scalar-primary-population-declaration-v1"
    ):
        declaration = load_scalar_primary_population_declaration(declaration_document)
        if args.qlognei_settings is None:
            raise PaperContractError("scalar primary smoke requires qLogNEI settings")
        settings_sidecar = args.qlognei_settings.with_suffix(".sha256")
        if (
            not settings_sidecar.is_file()
            or settings_sidecar.read_text(encoding="ascii").strip()
            != file_hash(args.qlognei_settings)
        ):
            raise PaperContractError("scalar qLogNEI settings hash mismatch")
        qlognei_settings = scalar_qlognei_settings_from_document(
            read_json(args.qlognei_settings)
        )
    else:
        declaration = load_primary_population_declaration(declaration_document)
        if args.qlognei_settings is not None:
            raise PaperContractError("legacy primary smoke forbids qLogNEI settings")
        qlognei_settings = None
    plan = read_json(args.plan)
    _validate_plan(plan, declaration.document()["declaration_identity"])
    if plan.get("budget", declaration.budget) != declaration.budget:
        raise PaperContractError("primary smoke plan/declaration budget mismatch")
    if qlognei_settings is not None and plan.get("settings_identity") != read_json(
        args.qlognei_settings
    ).get("settings_identity"):
        raise PaperContractError("scalar smoke plan/settings identity mismatch")
    is_population = plan.get("schema_version") == "mutaco-scalar-primary-population-plan-v1"
    if is_population and (
        plan.get("cohort_manifest_sha256") != declaration.target_ledger_hash
        or plan.get("maximum_search_calls")
        != declaration.document()["call_preview"]["maximum_search_calls"]
        or plan.get("maximum_assessment_calls")
        != declaration.document()["call_preview"]["maximum_assessment_calls"]
        or plan.get("maximum_total_calls")
        != declaration.document()["call_preview"]["maximum_total_calls"]
    ):
        raise PaperContractError("scalar population plan/declaration accounting mismatch")
    expected_calls = (
        int(plan["maximum_calls_per_worker"])
        if is_population
        else declaration.document()["call_preview"]["maximum_total_calls"]
    )
    if (
        args.maximum_calls != expected_calls
        or (
            not is_population and plan.get("maximum_total_calls") != expected_calls
        )
    ):
        raise PaperContractError("primary smoke worker call cap differs from declaration")
    if is_population:
        if args.workers != 120 or args.rank not in range(120):
            raise PaperContractError("scalar primary population requires 120 one-target workers")
        targets = _load_population_targets(
            args.targets_root,
            target_ids=declaration.target_record_identities,
            target_ledger_hash=declaration.target_ledger_hash,
        )
    else:
        if args.workers != 1 or args.rank != 0:
            raise PaperContractError("primary smoke requires one worker")
        targets = _load_smoke_target(
            args.targets_root,
            target_id=declaration.target_record_identities[0],
            target_ledger_hash=declaration.target_ledger_hash,
        )
    preview = primary_worker_preview(
        declaration, rank=args.rank, workers=args.workers, maximum_calls=args.maximum_calls
    )
    all_shards_validated = False
    if args.validate_all_shards:
        if not is_population or args.execute:
            raise PaperContractError("all-shard validation is population preview only")
        previews = tuple(
            primary_worker_preview(
                declaration,
                rank=rank,
                workers=args.workers,
                maximum_calls=args.maximum_calls,
            )
            for rank in range(args.workers)
        )
        if sorted(
            position for item in previews for position in item["target_positions"]
        ) != list(range(120)) or any(
            len(item["target_positions"]) != 1 for item in previews
        ):
            raise PaperContractError("scalar primary population shards are not one-to-one")
        all_shards_validated = True
    driver = GraphPrimaryIntegrationSmokeDriver(
        declaration=declaration,
        configuration_path=args.config,
        qlognei_settings=qlognei_settings,
    )
    print(json.dumps({
        **preview,
        "call_preview": declaration.document()["call_preview"],
        "plan_identity": plan["plan_identity"],
        "target_ledger_validated": True,
        "target_count_validated": len(targets),
        "all_shards_validated": all_shards_validated,
        "configuration_sha256": file_hash(args.config),
        "qlognei_settings_sha256": (
            None if args.qlognei_settings is None else file_hash(args.qlognei_settings)
        ),
        "diagnostics_contract": {
            "scalar_qlognei_official_botorch": qlognei_settings is not None,
            "scalar_qlognei_float32_only": qlognei_settings is not None,
            "nei_baseline_factorization": qlognei_settings is None,
            "mutaco_terminal_restart_histories": True,
            "nei_terminal_restart_histories": True,
            "sobol_surrogate_free": True,
        },
        "job_submitted": False,
        "scientific_calls_executed": 0,
    }, indent=2, sort_keys=True))
    if not args.execute:
        return
    if not args.confirm_approved:
        raise SystemExit("primary smoke execution requires --confirm-approved")
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).strip()
    if commit != args.expected_commit or dirty:
        raise SystemExit("primary smoke execution requires pinned clean source")
    torch.set_num_threads(args.torch_intraop_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    simulator = GraphScientificD4Adapter().build_target_simulator(
        read_json(args.config), device=torch.device("cpu"), dtype=torch.float32
    )

    def build_design(target_id: str):
        family = stable_hash("primary-design-family-v1", {
            "protocol_id": declaration.comparison.protocol_id,
            "campaign_id": declaration.campaign_id,
            "study_identity": declaration.study_identity,
            "target_id": target_id,
            "budget": declaration.budget,
            "comparison_identity": declaration.comparison.identity,
        })
        return generate_primary_sobol_design(
            campaign_id=declaration.campaign_id,
            target_id=target_id,
            budget=declaration.budget,
            parameter_spec=driver.parameter_spec,
            seed_deriver=SeedDeriver(
                ScientificIdentity(family), declaration.comparison.design_campaign_seed
            ),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    execute_primary_worker(
        declaration=declaration,
        output=args.output,
        rank=args.rank,
        workers=args.workers,
        maximum_calls=args.maximum_calls,
        source_commit=commit,
        load_target=lambda target_id: targets[target_id],
        build_design=build_design,
        run_method=lambda method, context, design, callback: driver.run_method(
            targets[design.target_id], method, context, design, callback
        ),
        evaluate_search=lambda method, theta, context: _evaluate_graph_simulator(
            simulator, theta, context
        ),
        evaluate_assessment=lambda theta, context: _evaluate_graph_simulator(
            simulator, theta, context
        ),
    )


if __name__ == "__main__":
    main()
