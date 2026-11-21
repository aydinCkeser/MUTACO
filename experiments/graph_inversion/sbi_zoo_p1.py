"""Canonical Paper-120 P1 shared-dataset execution and provenance surface.

Declaration and preview commands are zero-call.  Physical execution requires
both explicit CLI confirmation and the separately gated TRUBA profile.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from mutaco import SeedCoordinate, SimulationContext, canonical_json_bytes, checked_evaluate, stable_hash
from mutaco.atomic_io import atomic_write_bytes
from experiments.benchmarking.artifacts import file_hash, read_json
# Initialize the paper protocol before the typed SBI-zoo failure hierarchy is
# imported through accounting (the same ordering used by qualification).
from experiments.benchmarking.methods import direct_posterior_decision as _decision_surface
from experiments.benchmarking.methods.sbi_zoo_accounting import WriteAheadScientificCallLedger
from experiments.benchmarking.methods.sbi_zoo_contract import (
    BUDGETS,
    PRIOR_LAW,
    dataset_seed_deriver,
    randomness_contract,
    require_pinned_runtime,
)
from experiments.benchmarking.methods.sbi_zoo_dataset import (
    canonical_vector_sha256,
    load_shared_training_dataset,
    response_ordering_identity,
)
from experiments.benchmarking.methods.sbi_zoo_dataset_generation import (
    _assemble_slot_responses,
    generate_shared_training_dataset,
)
from experiments.graph_inversion.sbi_zoo_qualification import (
    LIFECYCLE_RECEIPT_SCHEMA,
    SCIENTIFIC_CONTRACT_SOURCE,
    SCIENTIFIC_CONTRACT_TREE,
    SIMULATOR_CONFIGURATION_ID,
    SIMULATOR_CONFIGURATION_SHA256,
    SIMULATOR_ID,
)
from mutaco_graph.profiles import SCIENTIFIC_M20


ROOT = Path(__file__).resolve().parents[2]
P1_SCHEMA = "mutaco-paper120-sbi-zoo-p1-declaration-v1"
P1_ID = "paper120_sbi_zoo_shared_dataset_p1_v1"
P1_PREFLIGHT_SCHEMA = "mutaco-paper120-sbi-zoo-p1-lifecycle-preflight-v1"
P1_COLLECTION_SCHEMA = "mutaco-paper120-sbi-zoo-p1-collection-v1"
P1_DATASET_FREEZE_SCHEMA = "mutaco-paper120-sbi-zoo-p1-dataset-freeze-v1"
P1_SCHEDULER_SCHEMA = "mutaco-paper120-sbi-zoo-p1-scheduler-record-v1"
P1_TARGET_COMPLETION_SCHEMA = "mutaco-paper120-sbi-zoo-p1-target-completion-v1"
PAPER120_MANIFEST_SHA256 = "359fc0bb59fc9a8a52f5da684e7a7eb352466bdc0d0a54fb8efa94131ce423bb"
TARGET_COUNT = 120
PHYSICAL_ARRAY_TASKS = 60
LOGICAL_TARGETS_PER_TASK = 2
TOTAL_SLOTS = 16_800
EXPECTED_DATASETS = 240
REPAIRED_HELDOUT_ROOTS = (92021540, 92021600)
SLURM_GEOMETRY = {
    "partition": "barbun",
    "nodes": 1,
    "ntasks": 1,
    "ntasks_per_node": 1,
    "runtime_threads_per_worker": 20,
    "slurm_cpus_per_task": 20,
    "requested_memory_per_node": "64G",
    "logical_workers_per_allocation": 1,
    "node_colocation": "scheduler_controlled",
    "expected_slurm_alloc_cpus_per_worker": 20,
    "observed_slurm_alloc_cpus_per_worker": None,
    "walltime": "00:30:00",
    "array": "0-59%20",
    "array_elements": PHYSICAL_ARRAY_TASKS,
    "logical_targets": TARGET_COUNT,
    "logical_targets_per_task": LOGICAL_TARGETS_PER_TASK,
    "logical_target_mapping": ["task_id", "task_id+60"],
    "worker_execution": "sequential_separate_python_processes",
    "maximum_concurrency": 20,
    "job_exclusive": False,
    "requeue": False,
}
COLLECTOR_SLURM_GEOMETRY = {
    "partition": "barbun",
    "nodes": 1,
    "ntasks": 1,
    "slurm_cpus_per_task": 20,
    "requested_memory_per_node": "64G",
    "walltime": "00:15:00",
    "job_exclusive": False,
    "requeue": False,
    "dependency": "afterany_worker_array",
}
RESPONSE_NAMES = SCIENTIFIC_M20.names
RESPONSE_ORDERING_IDENTITY = response_ordering_identity(RESPONSE_NAMES)


class P1ExecutionError(RuntimeError):
    """Fail-closed P1 declaration, execution, or population-integrity error."""


def logical_target_positions(physical_task_index: int) -> tuple[int, int]:
    """Map one physical Slurm task to its two ordered Paper-120 targets."""

    if physical_task_index < 0 or physical_task_index >= PHYSICAL_ARRAY_TASKS:
        raise P1ExecutionError("physical array task index is outside [0, 59]")
    targets = (physical_task_index, physical_task_index + PHYSICAL_ARRAY_TASKS)
    if any(target < 0 or target >= TARGET_COUNT for target in targets):
        raise P1ExecutionError("packed logical target position is outside Paper-120")
    return targets


def physical_task_mapping() -> list[dict[str, Any]]:
    """Return the complete deterministic 60-by-2 scheduler mapping."""

    return [
        {
            "physical_array_task_index": task_index,
            "logical_target_positions": list(logical_target_positions(task_index)),
        }
        for task_index in range(PHYSICAL_ARRAY_TASKS)
    ]


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _require_external(path: Path) -> Path:
    destination = path.resolve(strict=False)
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError:
        return destination
    raise P1ExecutionError("P1 output must remain outside the source checkout")


def _require_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise P1ExecutionError(f"missing immutable artifact or sidecar: {path}")
    actual = file_hash(path)
    if sidecar.read_text(encoding="ascii").strip() != actual:
        raise P1ExecutionError(f"immutable artifact hash mismatch: {path}")
    return actual


def resolve_paper120_cohort(cohort_root: Path) -> dict[str, Any]:
    """Resolve the exact ordered Paper-120 v3 cohort and all target hashes."""

    manifest_path = cohort_root / "manifest.json"
    manifest_sha = _require_sidecar(manifest_path)
    manifest = read_json(manifest_path)
    if manifest_sha != PAPER120_MANIFEST_SHA256 or manifest.get("completed_count") != TARGET_COUNT:
        raise P1ExecutionError("Paper-120 cohort manifest is not authoritative")
    records = manifest.get("records")
    if not isinstance(records, Mapping) or set(records) != {str(i) for i in range(TARGET_COUNT)}:
        raise P1ExecutionError("Paper-120 cohort positions are incomplete or noncanonical")
    targets = []
    seen_ids: set[str] = set()
    for position in range(TARGET_COUNT):
        path = cohort_root / f"target_{position:03d}.json"
        target_sha = _require_sidecar(path)
        target = read_json(path)
        expected = records[str(position)]
        target_id = target.get("target_record_id")
        if (
            target.get("schema_version") != "mutaco-frozen-target-record-v2"
            or target_id != expected.get("target_record_id")
            or target.get("content_hash") != expected.get("content_hash")
            or target.get("declaration_hash") != manifest.get("declaration_hash")
            or target.get("simulator_configuration_sha256") != SIMULATOR_CONFIGURATION_SHA256
            or target_id in seen_ids
        ):
            raise P1ExecutionError(f"Paper-120 target {position} disagrees with the cohort")
        seen_ids.add(str(target_id))
        targets.append(
            {
                "logical_worker_index": position,
                "physical_array_task_index": position % PHYSICAL_ARRAY_TASKS,
                "physical_task_sequence_index": position // PHYSICAL_ARRAY_TASKS,
                "target_position": position,
                "target_record_id": target_id,
                "target_content_identity": target["content_hash"],
                "target_artifact_path": str(path),
                "target_artifact_sha256": target_sha,
            }
        )
    return {
        "identity": "paper120_v3_ordered_cohort",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "declaration_hash": manifest["declaration_hash"],
        "target_count": TARGET_COUNT,
        "scheduler_indexing": "physical_task_i_runs_logical_targets_i_then_i_plus_60",
        "targets": targets,
    }


def _root_seed(cohort_identity: str) -> int:
    payload = canonical_json_bytes(
        {
            "domain": "paper120-sbi-zoo-p1-root-seed-v1",
            "scientific_contract_source": SCIENTIFIC_CONTRACT_SOURCE,
            "p1_id": P1_ID,
            "cohort_identity": cohort_identity,
        }
    )
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _load_lifecycle_receipt(path: Path, *, expected_source: str) -> dict[str, Any]:
    if not path.is_file():
        raise P1ExecutionError("lifecycle validation receipt is required")
    payload = path.read_bytes()
    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P1ExecutionError("lifecycle validation receipt is not JSON") from exc
    body = {key: value for key, value in receipt.items() if key != "receipt_identity"}
    materialization = receipt.get("source_materialization", {})
    preview = receipt.get("worker_preview", {})
    if (
        receipt.get("schema_version") != LIFECYCLE_RECEIPT_SCHEMA
        or receipt.get("status") != "validated_zero_call"
        or receipt.get("receipt_identity") != stable_hash(LIFECYCLE_RECEIPT_SCHEMA, body)
        or receipt.get("source_commit") != expected_source
        or materialization.get("status") != "materialized_clean"
        or materialization.get("source_commit") != expected_source
        or preview.get("schema_version") != P1_PREFLIGHT_SCHEMA
        or preview.get("p1_execution_source") != expected_source
        or preview.get("scientific_calls_executed") != 0
        or preview.get("job_submitted") is not False
        or preview.get("declaration_published") is not False
        or receipt.get("scientific_calls") != 0
        or receipt.get("job_submitted") is not False
        or receipt.get("submission_authorized") is not False
    ):
        raise P1ExecutionError("lifecycle validation receipt is invalid or mismatched")
    return {
        "receipt_identity": receipt["receipt_identity"],
        "receipt_sha256": _sha(payload),
        "receipt": receipt,
    }


def build_zero_call_preflight(
    *, cohort_root: Path, simulator_configuration_path: Path, output_root: Path,
    execution_source: str, execution_tree: str,
) -> dict[str, Any]:
    cohort = resolve_paper120_cohort(cohort_root)
    if file_hash(simulator_configuration_path) != SIMULATOR_CONFIGURATION_SHA256:
        raise P1ExecutionError("simulator configuration hash is not authoritative")
    _require_external(output_root)
    require_pinned_runtime()
    return {
        "schema_version": P1_PREFLIGHT_SCHEMA,
        "status": "zero_call_preflight_receipt_required_before_declaration",
        "p1_execution_source": execution_source,
        "p1_execution_tree": execution_tree,
        "paper120_targets": cohort["target_count"],
        "declared_slots": TOTAL_SLOTS,
        "expected_datasets": EXPECTED_DATASETS,
        "slurm": SLURM_GEOMETRY,
        "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
        "scientific_calls_executed": 0,
        "call_preview": {"scientific_calls_executed": 0, "maximum_calls": TOTAL_SLOTS},
        "heldout_calls": 0,
        "heldout_accesses": 0,
        "job_submitted": False,
        "declaration_published": False,
        "lifecycle_receipt_binding_required": True,
    }


def _dataset_declaration_id(
    p1_identity: str, *, target_position: int, target_record_id: str, budget: int, root_seed: int
) -> str:
    return stable_hash(
        "paper120-sbi-zoo-p1-dataset-declaration-v1",
        {
            "p1_declaration_identity": p1_identity,
            "target_position": target_position,
            "target_record_id": target_record_id,
            "budget": budget,
            "root_seed": root_seed,
        },
    )


def _enumerate_dataset(
    *, p1_identity: str, root_seed: int, target: Mapping[str, Any], budget: int
) -> dict[str, Any]:
    target_id = str(target["target_record_id"])
    declaration_id = _dataset_declaration_id(
        p1_identity,
        target_position=int(target["target_position"]),
        target_record_id=target_id,
        budget=budget,
        root_seed=root_seed,
    )
    ledger_identity = stable_hash(
        "sbi-zoo-call-ledger-v1",
        {"declaration_identity": declaration_id, "maximum_calls": budget},
    )
    deriver = dataset_seed_deriver(p1_identity, root_seed)
    theta, theta_seed = PRIOR_LAW.sample(
        budget, deriver=deriver, target_record_id=target_id, budget=budget
    )
    slots = []
    for row_index in range(budget):
        evaluation_seed = deriver.seed_for(
            SeedCoordinate(
                "sbi_zoo_shared_dataset_simulation_row",
                (target_id, budget, row_index),
            )
        )
        context_body = {
            "schema_version": "mutaco-sbi-zoo-training-context-v1",
            "target_record_id": target_id,
            "budget": budget,
            "row_index": row_index,
            "evaluation_seed": evaluation_seed,
            "purpose": "sbi_zoo_shared_prior_predictive_training",
        }
        context_identity = stable_hash("sbi-zoo-training-context-v1", context_body)
        theta_sha = canonical_vector_sha256(
            theta[row_index].contiguous(), length=4, field="dataset theta row"
        )
        slot_name = f"row-{row_index:03d}"
        slot_body = {
            "dataset_generation_declaration_identity": declaration_id,
            "call_ledger_identity": ledger_identity,
            "target_record_id": target_id,
            "budget": budget,
            "slot": slot_name,
            "row_index": row_index,
            "theta_canonical_sha256": theta_sha,
            "stochastic_context_identity": context_identity,
        }
        slots.append(
            {
                "slot": slot_name,
                "row_index": row_index,
                "scientific_slot_identity": stable_hash("sbi-zoo-scientific-slot-v1", slot_body),
                "theta_canonical_sha256": theta_sha,
                "stochastic_context_identity": context_identity,
                "evaluation_seed": evaluation_seed,
            }
        )
    return {
        "target_position": target["target_position"],
        "target_record_id": target_id,
        "budget": budget,
        "dataset_generation_declaration_identity": declaration_id,
        "call_ledger_identity": ledger_identity,
        "theta_seed": theta_seed,
        "expected_rows": budget,
        "expected_artifact_path": f"targets/target-{int(target['target_position']):03d}/b{budget}/artifact",
        "slots": slots,
    }


def build_declaration(
    *, cohort_root: Path, simulator_configuration_path: Path, output_root: Path,
    execution_source: str, execution_tree: str, lifecycle_receipt_path: Path,
) -> dict[str, Any]:
    cohort = resolve_paper120_cohort(cohort_root)
    config_sha = file_hash(simulator_configuration_path)
    if config_sha != SIMULATOR_CONFIGURATION_SHA256:
        raise P1ExecutionError("simulator configuration hash is not authoritative")
    output = _require_external(output_root)
    receipt = _load_lifecycle_receipt(lifecycle_receipt_path, expected_source=execution_source)
    root_seed = _root_seed(cohort["declaration_hash"])
    core = {
        "schema_version": P1_SCHEMA,
        "status": "declared_unexecuted_separate_p1_authorization_required",
        "p1_id": P1_ID,
        "scientific_contract_source": SCIENTIFIC_CONTRACT_SOURCE,
        "scientific_contract_tree": SCIENTIFIC_CONTRACT_TREE,
        "qualification_execution_source": "bffa31ec729dfb90fa294c8c02f47634f0a0c7be",
        "p1_execution_source": execution_source,
        "p1_execution_tree": execution_tree,
        "lifecycle_validation_receipt": receipt,
        "cohort": cohort,
        "simulator": {
            "identity": SIMULATOR_ID,
            "configuration_identity": SIMULATOR_CONFIGURATION_ID,
            "configuration_path": str(simulator_configuration_path),
            "configuration_sha256": config_sha,
            "dtype": "float32",
            "device": "cpu",
            "response_names": list(RESPONSE_NAMES),
            "response_ordering_identity": RESPONSE_ORDERING_IDENTITY,
        },
        "prior": PRIOR_LAW.document(),
        "randomness": {
            "root_seed": root_seed,
            "root_seed_derivation": "sha256_domain_scientific_source_p1_cohort_first_u64_mask63",
            "contract": randomness_contract(),
            "b40_b100_relation": "independent_not_nested_budget_is_seed_coordinate",
        },
        "scientific_call_contract": {
            "b40_slots": 4_800,
            "b100_slots": 12_000,
            "total_slots": TOTAL_SLOTS,
            "automatic_retry": False,
            "replacement_slots": 0,
            "durable_attempted_is_irreversible": True,
            "attempted_without_terminal": "attempted_outcome_unknown",
            "scheduler_restart": "only_durably_proven_unattempted_slots",
        },
        "datasets": {
            "expected_b40": 120,
            "expected_b100": 120,
            "expected_total": EXPECTED_DATASETS,
            "progression": "freeze_b40_then_generate_and_freeze_independent_b100",
            "p1_pass": "all_240_datasets_and_all_16800_slots_success",
        },
        "execution": {
            "physical_execution_authorized_by_declaration": False,
            "requires_cli_execute_and_confirm_approved": True,
            "source_checkout_must_remain_clean": True,
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "dependencies": require_pinned_runtime(),
                "torch_intraop_threads": 20,
                "torch_interop_threads": 1,
                "thread_environment": {
                    "OMP_NUM_THREADS": "20",
                    "MKL_NUM_THREADS": "20",
                    "OPENBLAS_NUM_THREADS": "20",
                },
            },
            "slurm": SLURM_GEOMETRY,
            "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
            "physical_task_mapping": physical_task_mapping(),
        },
        "output_root": str(output),
        "append_only": True,
        "heldout_firewall": {
            "heldout_calls": 0,
            "heldout_accesses": 0,
            "repaired_family_roots_forbidden": list(REPAIRED_HELDOUT_ROOTS),
        },
        "p2_gate": "blocked_until_read_only_population_collection_pass_240_of_240",
        "expected_schemas": {
            "slot": "mutaco-sbi-zoo-slot-outcome-v1",
            "dataset": "mutaco-sbi-zoo-shared-dataset-v1",
            "freeze": P1_DATASET_FREEZE_SCHEMA,
            "collection": P1_COLLECTION_SCHEMA,
            "canonical_hashing": "sha256_over_schema_specific_canonical_payloads",
        },
    }
    identity = stable_hash(P1_SCHEMA, core)
    enumerated = [
        _enumerate_dataset(
            p1_identity=identity, root_seed=root_seed, target=target, budget=budget
        )
        for target in cohort["targets"]
        for budget in BUDGETS
    ]
    document = {
        **core,
        "p1_declaration_identity": identity,
        "population_call_ledger_identity": stable_hash(
            "mutaco-paper120-sbi-zoo-p1-population-ledger-v1",
            {
                "p1_declaration_identity": identity,
                "dataset_call_ledger_identities": [
                    item["call_ledger_identity"] for item in enumerated
                ],
                "total_slots": TOTAL_SLOTS,
            },
        ),
        "ordered_dataset_declarations": enumerated,
    }
    document["p1_complete_content_identity"] = stable_hash(
        "mutaco-paper120-sbi-zoo-p1-complete-content-v1", document
    )
    return document


def validate_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    content_identity = document.get("p1_complete_content_identity")
    without_content = {k: v for k, v in document.items() if k != "p1_complete_content_identity"}
    core = {
        k: v
        for k, v in without_content.items()
        if k
        not in {
            "p1_declaration_identity",
            "population_call_ledger_identity",
            "ordered_dataset_declarations",
        }
    }
    identity = stable_hash(P1_SCHEMA, core)
    datasets = document.get("ordered_dataset_declarations", [])
    slots = [slot for dataset in datasets for slot in dataset.get("slots", [])]
    lifecycle_binding = document.get("lifecycle_validation_receipt", {})
    lifecycle_receipt = lifecycle_binding.get("receipt", {})
    lifecycle_body = {
        key: value for key, value in lifecycle_receipt.items() if key != "receipt_identity"
    }
    lifecycle_bytes = (
        json.dumps(
            lifecycle_receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    expected_datasets = [
        _enumerate_dataset(
            p1_identity=identity,
            root_seed=int(document.get("randomness", {}).get("root_seed", -1)),
            target=target,
            budget=budget,
        )
        for target in document.get("cohort", {}).get("targets", [])
        for budget in BUDGETS
    ]
    independent_budget_pairs = all(
        expected_datasets[index]["dataset_generation_declaration_identity"]
        != expected_datasets[index + 1]["dataset_generation_declaration_identity"]
        and [slot["theta_canonical_sha256"] for slot in expected_datasets[index]["slots"]]
        != [
            slot["theta_canonical_sha256"]
            for slot in expected_datasets[index + 1]["slots"][:40]
        ]
        for index in range(0, len(expected_datasets), 2)
    )
    expected_population_ledger_identity = stable_hash(
        "mutaco-paper120-sbi-zoo-p1-population-ledger-v1",
        {
            "p1_declaration_identity": identity,
            "dataset_call_ledger_identities": [
                item["call_ledger_identity"] for item in expected_datasets
            ],
            "total_slots": TOTAL_SLOTS,
        },
    )
    if (
        document.get("schema_version") != P1_SCHEMA
        or document.get("p1_declaration_identity") != identity
        or content_identity
        != stable_hash("mutaco-paper120-sbi-zoo-p1-complete-content-v1", without_content)
        or document.get("scientific_contract_source") != SCIENTIFIC_CONTRACT_SOURCE
        or document.get("cohort", {}).get("manifest_sha256") != PAPER120_MANIFEST_SHA256
        or len(document.get("cohort", {}).get("targets", [])) != TARGET_COUNT
        or len(datasets) != EXPECTED_DATASETS
        or len(slots) != TOTAL_SLOTS
        or len({slot.get("scientific_slot_identity") for slot in slots}) != TOTAL_SLOTS
        or datasets != expected_datasets
        or document.get("population_call_ledger_identity")
        != expected_population_ledger_identity
        or not independent_budget_pairs
        or document.get("execution", {}).get("slurm") != SLURM_GEOMETRY
        or document.get("execution", {}).get("collector_slurm")
        != COLLECTOR_SLURM_GEOMETRY
        or document.get("execution", {}).get("physical_task_mapping")
        != physical_task_mapping()
        or document.get("heldout_firewall", {}).get("heldout_accesses") != 0
        or lifecycle_receipt.get("receipt_identity")
        != stable_hash(LIFECYCLE_RECEIPT_SCHEMA, lifecycle_body)
        or lifecycle_binding.get("receipt_identity") != lifecycle_receipt.get("receipt_identity")
        or lifecycle_binding.get("receipt_sha256") != _sha(lifecycle_bytes)
        or lifecycle_receipt.get("source_commit") != document.get("p1_execution_source")
    ):
        raise P1ExecutionError("P1 declaration is invalid")
    return dict(document)


def write_declaration(path: Path, document: Mapping[str, Any]) -> None:
    validate_declaration(document)
    if path.exists() or path.with_suffix(".sha256").exists():
        raise P1ExecutionError("P1 declaration is append-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    atomic_write_bytes(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))


def load_declaration(path: Path) -> dict[str, Any]:
    _require_sidecar(path)
    return validate_declaration(read_json(path))


def preview(document: Mapping[str, Any]) -> dict[str, Any]:
    validate_declaration(document)
    return {
        "schema_version": "mutaco-paper120-sbi-zoo-p1-preview-v1",
        "p1_declaration_identity": document["p1_declaration_identity"],
        "paper120_targets": TARGET_COUNT,
        "b40_slots": 4_800,
        "b100_slots": 12_000,
        "total_slots": TOTAL_SLOTS,
        "expected_datasets": EXPECTED_DATASETS,
        "slurm": SLURM_GEOMETRY,
        "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
        "scientific_calls_executed": 0,
        "call_preview": {"scientific_calls_executed": 0, "maximum_calls": TOTAL_SLOTS},
        "heldout_calls": 0,
        "heldout_accesses": 0,
        "job_submitted": False,
    }


def _require_exact_clean_source(document: Mapping[str, Any]) -> None:
    if (
        _git("rev-parse", "HEAD") != document["p1_execution_source"]
        or _git("rev-parse", "HEAD^{tree}") != document["p1_execution_tree"]
        or _git("status", "--porcelain", "--untracked-files=all")
    ):
        raise P1ExecutionError("P1 requires the exact clean execution source")


def physical_task_worker_command(
    declaration_path: Path, *, target_index: int
) -> tuple[str, ...]:
    """Build one independently constructed logical-target worker command."""

    if target_index < 0 or target_index >= TARGET_COUNT:
        raise P1ExecutionError("logical target index is outside Paper-120")
    return (
        sys.executable,
        "-u",
        "-m",
        "experiments.graph_inversion.sbi_zoo_p1",
        "worker",
        "--declaration",
        str(declaration_path),
        "--target-index",
        str(target_index),
        "--execute",
        "--confirm-approved",
    )


def execute_physical_task(
    declaration_path: Path,
    *,
    physical_task_index: int,
    runner: Any = subprocess.run,
) -> tuple[int, dict[str, Any]]:
    """Run both packed logical targets, without suppressing the second on failure."""

    targets = logical_target_positions(physical_task_index)
    results: list[dict[str, int]] = []
    for target_index in targets:
        completed = runner(
            physical_task_worker_command(declaration_path, target_index=target_index),
            check=False,
        )
        results.append(
            {"target_position": target_index, "returncode": int(completed.returncode)}
        )
    all_succeeded = all(item["returncode"] == 0 for item in results)
    document = {
        "schema_version": "mutaco-paper120-sbi-zoo-p1-physical-task-terminal-v1",
        "physical_array_task_index": physical_task_index,
        "logical_target_positions": list(targets),
        "logical_worker_results": results,
        "all_logical_workers_succeeded": all_succeeded,
        "automatic_retry": False,
        "slurm_requeue": False,
    }
    return (0 if all_succeeded else 1), document


def _dataset_spec(document: Mapping[str, Any], target_index: int, budget: int) -> dict[str, Any]:
    offset = target_index * len(BUDGETS) + BUDGETS.index(budget)
    value = document["ordered_dataset_declarations"][offset]
    if value["target_position"] != target_index or value["budget"] != budget:
        raise P1ExecutionError("dataset declaration order is corrupt")
    return value


def _freeze_path(root: Path) -> Path:
    return root / "freeze.json"


def _load_freeze(root: Path, *, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = _freeze_path(root)
    _require_sidecar(path)
    value = read_json(path)
    body = {k: v for k, v in value.items() if k != "freeze_identity"}
    if value.get("freeze_identity") != stable_hash(P1_DATASET_FREEZE_SCHEMA, body):
        raise P1ExecutionError("dataset freeze identity mismatch")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise P1ExecutionError(f"dataset freeze {key} mismatch")
    return value


def _publish_freeze(
    root: Path, *, document: Mapping[str, Any], spec: Mapping[str, Any], dataset: Any,
    ledger: WriteAheadScientificCallLedger,
) -> dict[str, Any]:
    path = _freeze_path(root)
    if path.exists() or path.with_suffix(".sha256").exists():
        return _load_freeze(
            root,
            expected={
                "p1_declaration_identity": document["p1_declaration_identity"],
                "target_position": spec["target_position"],
                "target_record_id": spec["target_record_id"],
                "budget": spec["budget"],
                "dataset_identity": dataset.identity,
                "canonical_content_sha256": dataset.content_sha256,
            },
        )
    accounting = ledger.accounting()
    expected_accounting = {
        "maximum_calls": spec["budget"],
        "attempted_calls": spec["budget"],
        "successful_calls": spec["budget"],
        "failed_calls": 0,
        "attempted_outcome_unknown": 0,
        "retry_calls": 0,
    }
    if accounting != expected_accounting:
        raise P1ExecutionError("only a completely successful dataset may be frozen")
    body = {
        "schema_version": P1_DATASET_FREEZE_SCHEMA,
        "p1_declaration_identity": document["p1_declaration_identity"],
        "target_position": spec["target_position"],
        "target_record_id": spec["target_record_id"],
        "budget": spec["budget"],
        "dataset_generation_declaration_identity": spec[
            "dataset_generation_declaration_identity"
        ],
        "call_ledger_identity": ledger.identity,
        "dataset_identity": dataset.identity,
        "canonical_content_sha256": dataset.content_sha256,
        "accounting": accounting,
        "immutable": True,
    }
    value = {**body, "freeze_identity": stable_hash(P1_DATASET_FREEZE_SCHEMA, body)}
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))
    return _load_freeze(root, expected={k: value[k] for k in body if k != "accounting"})


def _publish_target_completion(
    target_root: Path, *, document: Mapping[str, Any], target_index: int,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": P1_TARGET_COMPLETION_SCHEMA,
        "p1_declaration_identity": document["p1_declaration_identity"],
        "target_position": target_index,
        "target_record_id": document["cohort"]["targets"][target_index]["target_record_id"],
        "dataset_freeze_identities": {
            str(item["budget"]): item["freeze_identity"] for item in results
        },
        "progression": "b40_frozen_before_b100_then_target_complete",
        "heldout_calls": 0,
        "heldout_accesses": 0,
    }
    value = {
        **body,
        "target_completion_identity": stable_hash(P1_TARGET_COMPLETION_SCHEMA, body),
    }
    path = target_root / "complete.json"
    if path.exists() or path.with_suffix(".sha256").exists():
        _require_sidecar(path)
        if read_json(path) != value:
            raise P1ExecutionError("target completion identity mismatch")
        return value
    payload = canonical_json_bytes(value)
    atomic_write_bytes(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))
    return value


def _load_target_completion(
    target_root: Path, *, document: Mapping[str, Any], target_index: int,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = target_root / "complete.json"
    _require_sidecar(path)
    value = read_json(path)
    body = {key: item for key, item in value.items() if key != "target_completion_identity"}
    expected_freezes = {str(item["budget"]): item["freeze_identity"] for item in receipts}
    if (
        value.get("target_completion_identity")
        != stable_hash(P1_TARGET_COMPLETION_SCHEMA, body)
        or value.get("p1_declaration_identity") != document["p1_declaration_identity"]
        or value.get("target_position") != target_index
        or value.get("target_record_id")
        != document["cohort"]["targets"][target_index]["target_record_id"]
        or value.get("dataset_freeze_identities") != expected_freezes
        or value.get("heldout_calls") != 0
        or value.get("heldout_accesses") != 0
    ):
        raise P1ExecutionError("target completion artifact is invalid")
    return value


def _validate_target_artifact(document: Mapping[str, Any], target_index: int) -> Path:
    target = document["cohort"]["targets"][target_index]
    path = ROOT / target["target_artifact_path"]
    if _require_sidecar(path) != target["target_artifact_sha256"]:
        raise P1ExecutionError("target artifact changed after declaration")
    return path


def execute_target(
    document: Mapping[str, Any], *, target_index: int, simulator_factory: Any | None = None
) -> dict[str, Any]:
    if target_index not in range(TARGET_COUNT):
        raise P1ExecutionError("array index is outside Paper-120")
    _require_exact_clean_source(document)
    _validate_target_artifact(document, target_index)
    config_path = ROOT / document["simulator"]["configuration_path"]
    if file_hash(config_path) != document["simulator"]["configuration_sha256"]:
        raise P1ExecutionError("simulator configuration changed after declaration")
    if simulator_factory is None:
        from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter

        simulator = GraphScientificD4Adapter().build_target_simulator(
            read_json(config_path), device=torch.device("cpu"), dtype=torch.float32
        )
    else:
        simulator = simulator_factory(read_json(config_path))
    if simulator.capabilities.simulator_identity != SIMULATOR_ID:
        raise P1ExecutionError("simulator identity differs from declaration")
    response_names = tuple(axis.name for axis in simulator.capabilities.response_spec.axes)
    if response_names != RESPONSE_NAMES:
        raise P1ExecutionError("simulator response ordering differs from declaration")
    deriver = dataset_seed_deriver(
        document["p1_declaration_identity"], int(document["randomness"]["root_seed"])
    )
    target_root = Path(document["output_root"]) / "targets" / f"target-{target_index:03d}"
    results = []
    started = perf_counter()
    for budget in BUDGETS:
        spec = _dataset_spec(document, target_index, budget)
        root = target_root / f"b{budget}"
        ledger = WriteAheadScientificCallLedger(
            root / "ledger", spec["dataset_generation_declaration_identity"], budget
        )

        def evaluate(theta: torch.Tensor, context: Mapping[str, Any]) -> torch.Tensor:
            result = checked_evaluate(
                simulator,
                theta,
                device=torch.device("cpu"),
                dtype=torch.float32,
                context=SimulationContext(
                    str(context["context_identity"]),
                    str(context["purpose"]),
                    int(context["row_index"]),
                    0,
                    int(context["evaluation_seed"]),
                ),
            )
            return result.values.detach().cpu().contiguous()

        artifact_path = root / "artifact"
        if artifact_path.is_dir():
            dataset = load_shared_training_dataset(artifact_path)
        else:
            dataset = generate_shared_training_dataset(
                artifact_path,
                target_record_id=spec["target_record_id"],
                budget=budget,
                deriver=deriver,
                response_names=response_names,
                simulator_identity=SIMULATOR_ID,
                simulator_configuration_identity=SIMULATOR_CONFIGURATION_ID,
                dataset_generation_declaration_identity=spec[
                    "dataset_generation_declaration_identity"
                ],
                ledger=ledger,
                evaluate=evaluate,
                resume_unattempted=True,
            )
        freeze = _publish_freeze(root, document=document, spec=spec, dataset=dataset, ledger=ledger)
        results.append(
            {
                "budget": budget,
                "dataset_identity": dataset.identity,
                "freeze_identity": freeze["freeze_identity"],
                "accounting": ledger.accounting(),
            }
        )
    completion = _publish_target_completion(
        target_root, document=document, target_index=target_index, results=results
    )
    return {
        "target_index": target_index,
        "target_record_id": document["cohort"]["targets"][target_index]["target_record_id"],
        "datasets": results,
        "target_completion_identity": completion["target_completion_identity"],
        "elapsed_seconds": perf_counter() - started,
        "heldout_calls": 0,
        "heldout_accesses": 0,
    }


def _validate_dataset(document: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    root = (
        Path(document["output_root"])
        / "targets"
        / f"target-{int(spec['target_position']):03d}"
        / f"b{int(spec['budget'])}"
    )
    dataset = load_shared_training_dataset(root / "artifact")
    if not (root / "ledger" / "ledger.json").is_file():
        raise P1ExecutionError("dataset call ledger is missing")
    ledger = WriteAheadScientificCallLedger(
        root / "ledger", spec["dataset_generation_declaration_identity"], int(spec["budget"])
    )
    freeze = _load_freeze(
        root,
        expected={
            "p1_declaration_identity": document["p1_declaration_identity"],
            "target_position": spec["target_position"],
            "target_record_id": spec["target_record_id"],
            "budget": spec["budget"],
            "dataset_generation_declaration_identity": spec[
                "dataset_generation_declaration_identity"
            ],
            "call_ledger_identity": spec["call_ledger_identity"],
            "dataset_identity": dataset.identity,
            "canonical_content_sha256": dataset.content_sha256,
        },
    )
    if dataset.metadata.get("target_record_id") != spec["target_record_id"]:
        raise P1ExecutionError("dataset target identity mismatch")
    if dataset.metadata.get("budget") != spec["budget"]:
        raise P1ExecutionError("dataset budget mismatch")
    expected_contexts = [slot["stochastic_context_identity"] for slot in spec["slots"]]
    if (
        dataset.metadata.get("parameter_names") != list(PRIOR_LAW.parameter_names)
        or dataset.metadata.get("response_names") != list(RESPONSE_NAMES)
        or dataset.metadata.get("row_context_identities") != expected_contexts
        or dataset.metadata.get("dataset_generation_declaration_identity")
        != spec["dataset_generation_declaration_identity"]
        or dataset.metadata.get("scientific_call_ledger_identity")
        != spec["call_ledger_identity"]
        or dataset.metadata.get("simulator_identity") != SIMULATOR_ID
        or dataset.metadata.get("simulator_configuration_identity")
        != SIMULATOR_CONFIGURATION_ID
        or dataset.metadata.get("theta_seed") != spec["theta_seed"]
    ):
        raise P1ExecutionError("dataset scientific metadata differs from declaration")
    actual_theta_hashes = [
        canonical_vector_sha256(
            dataset.theta[row].contiguous(), length=4, field="dataset theta row"
        )
        for row in range(int(spec["budget"]))
    ]
    if actual_theta_hashes != [slot["theta_canonical_sha256"] for slot in spec["slots"]]:
        raise P1ExecutionError("dataset theta rows differ from declaration")
    if ledger.accounting() != freeze["accounting"]:
        raise P1ExecutionError("dataset freeze and ledger accounting mismatch")
    rebuilt_response = _assemble_slot_responses(
        ledger=ledger,
        target_record_id=str(spec["target_record_id"]),
        budget=int(spec["budget"]),
        theta=dataset.theta,
        context_identities=expected_contexts,
        response_ordering_identity_value=RESPONSE_ORDERING_IDENTITY,
        dataset_generation_declaration_identity=str(
            spec["dataset_generation_declaration_identity"]
        ),
        simulator_identity=SIMULATOR_ID,
        simulator_configuration_identity=SIMULATOR_CONFIGURATION_ID,
    )
    if not torch.equal(rebuilt_response, dataset.response):
        raise P1ExecutionError("dataset response differs from validated slot outcomes")
    return {
        "target_position": spec["target_position"],
        "target_record_id": spec["target_record_id"],
        "budget": spec["budget"],
        "dataset_identity": dataset.identity,
        "canonical_content_sha256": dataset.content_sha256,
        "freeze_identity": freeze["freeze_identity"],
        "accounting": ledger.accounting(),
    }


def evaluate_population_receipts(
    document: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {
        (int(spec["target_position"]), int(spec["budget"])): spec
        for spec in document["ordered_dataset_declarations"]
    }
    keyed: dict[tuple[int, int], Mapping[str, Any]] = {}
    errors = []
    for receipt in receipts:
        key = (int(receipt.get("target_position", -1)), int(receipt.get("budget", -1)))
        if key not in expected or key in keyed:
            errors.append(f"unexpected_or_duplicate_dataset:{key}")
        else:
            keyed[key] = receipt
    missing = sorted(set(expected) - set(keyed))
    if missing:
        errors.append(f"missing_datasets:{len(missing)}")
    attempted = successful = failed = unknown = 0
    for key, receipt in keyed.items():
        spec = expected[key]
        accounting = receipt.get("accounting", {})
        attempted += int(accounting.get("attempted_calls", 0))
        successful += int(accounting.get("successful_calls", 0))
        failed += int(accounting.get("failed_calls", 0))
        unknown += int(accounting.get("attempted_outcome_unknown", 0))
        if (
            receipt.get("target_record_id") != spec["target_record_id"]
            or accounting.get("maximum_calls") != spec["budget"]
            or accounting.get("attempted_calls") != spec["budget"]
            or accounting.get("successful_calls") != spec["budget"]
            or accounting.get("failed_calls") != 0
            or accounting.get("attempted_outcome_unknown") != 0
            or accounting.get("retry_calls") != 0
        ):
            errors.append(f"invalid_dataset:{key}")
    for target_position in range(TARGET_COUNT):
        b40 = keyed.get((target_position, 40), {})
        b100 = keyed.get((target_position, 100), {})
        if b40 and b100 and (
            b40.get("dataset_identity") == b100.get("dataset_identity")
            or b40.get("canonical_content_sha256") == b100.get("canonical_content_sha256")
        ):
            errors.append(f"b40_b100_alias:{target_position}")
    passed = (
        not errors
        and len(receipts) == EXPECTED_DATASETS
        and attempted == TOTAL_SLOTS
        and successful == TOTAL_SLOTS
        and failed == 0
        and unknown == 0
    )
    return {
        "status": "PASS" if passed else "BLOCKED",
        "datasets": len(receipts),
        "attempted_calls": attempted,
        "successful_calls": successful,
        "failed_calls": failed,
        "attempted_outcome_unknown": unknown,
        "errors": errors,
        "p2_gate": "OPEN" if passed else "BLOCKED",
        "heldout_calls": 0,
        "heldout_accesses": 0,
    }


def collect_population(document: Mapping[str, Any], *, write: bool = False) -> dict[str, Any]:
    validate_declaration(document)
    output_root = Path(document["output_root"])
    target_root = output_root / "targets"
    expected_target_dirs = {f"target-{i:03d}" for i in range(TARGET_COUNT)}
    actual_target_dirs = (
        {path.name for path in target_root.iterdir() if path.is_dir()}
        if target_root.is_dir()
        else set()
    )
    receipts = []
    errors = []
    if actual_target_dirs != expected_target_dirs:
        errors.append("target_directory_set_mismatch")
    for spec in document["ordered_dataset_declarations"]:
        try:
            receipts.append(_validate_dataset(document, spec))
        except Exception as exc:
            errors.append(
                f"target_{spec['target_position']}_b{spec['budget']}:{type(exc).__name__}:{exc}"
            )
    for target_index in range(TARGET_COUNT):
        target_receipts = [
            item for item in receipts if item["target_position"] == target_index
        ]
        try:
            _load_target_completion(
                target_root / f"target-{target_index:03d}",
                document=document,
                target_index=target_index,
                receipts=target_receipts,
            )
        except Exception as exc:
            errors.append(
                f"target_{target_index}_completion:{type(exc).__name__}:{exc}"
            )
    for target_index in range(TARGET_COUNT):
        directory = target_root / f"target-{target_index:03d}"
        if directory.is_dir() and {path.name for path in directory.iterdir() if path.is_dir()} != {
            "b40",
            "b100",
        }:
            errors.append(f"target_{target_index}_budget_directory_set_mismatch")
    result = evaluate_population_receipts(document, receipts)
    result["errors"] = errors + result["errors"]
    if errors:
        result["status"] = "BLOCKED"
        result["p2_gate"] = "BLOCKED"
    body = {
        "schema_version": P1_COLLECTION_SCHEMA,
        "p1_declaration_identity": document["p1_declaration_identity"],
        "p1_complete_content_identity": document["p1_complete_content_identity"],
        "p1_execution_source": document["p1_execution_source"],
        **result,
        "dataset_receipts": receipts,
    }
    collection = {**body, "collection_identity": stable_hash(P1_COLLECTION_SCHEMA, body)}
    if write:
        collection_root = output_root / "collection"
        ordinal = len(tuple(collection_root.glob("attempt-*"))) if collection_root.exists() else 0
        destination = collection_root / f"attempt-{ordinal:03d}" / "manifest.json"
        if destination.exists() or destination.with_suffix(".sha256").exists():
            raise P1ExecutionError("P1 collection attempt is append-only")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(collection)
        atomic_write_bytes(destination, payload)
        atomic_write_bytes(
            destination.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii")
        )
    return collection


def inspect_restart(document: Mapping[str, Any], *, target_index: int) -> dict[str, Any]:
    statuses = []
    for budget in BUDGETS:
        spec = _dataset_spec(document, target_index, budget)
        root = (
            Path(document["output_root"])
            / "targets"
            / f"target-{target_index:03d}"
            / f"b{budget}"
        )
        ledger_root = root / "ledger"
        if (ledger_root / "ledger.json").is_file():
            ledger = WriteAheadScientificCallLedger(
                ledger_root, spec["dataset_generation_declaration_identity"], budget
            )
            slot_statuses = [ledger.slot_status(f"row-{i:03d}") for i in range(budget)]
        else:
            slot_statuses = [None] * budget
        blocked = any(status in {"failed", "attempted_outcome_unknown"} for status in slot_statuses)
        statuses.append(
            {
                "budget": budget,
                "frozen": _freeze_path(root).is_file(),
                "proven_unattempted": sum(status in {None, "intent"} for status in slot_statuses),
                "successful": slot_statuses.count("success"),
                "nonretryable_blocked": blocked,
                "restart_permitted": not blocked and not _freeze_path(root).is_file(),
            }
        )
    return {"target_index": target_index, "datasets": statuses}


def load_passing_collection(document: Mapping[str, Any]) -> dict[str, Any]:
    """Require one immutable published PASS collection before opening P2."""

    root = Path(document["output_root"]) / "collection"
    passing = []
    for path in sorted(root.glob("attempt-*/manifest.json")):
        _require_sidecar(path)
        value = read_json(path)
        body = {key: item for key, item in value.items() if key != "collection_identity"}
        if value.get("collection_identity") != stable_hash(P1_COLLECTION_SCHEMA, body):
            raise P1ExecutionError("P1 collection identity mismatch")
        if value.get("p1_declaration_identity") != document["p1_declaration_identity"]:
            raise P1ExecutionError("P1 collection declaration mismatch")
        if value.get("status") == "PASS" and value.get("p2_gate") == "OPEN":
            passing.append(value)
    if len(passing) != 1:
        raise P1ExecutionError("P2 requires exactly one immutable P1 PASS collection")
    return passing[0]


def write_scheduler_record(
    document: Mapping[str, Any], *, array_job_id: str, collector_job_id: str
) -> dict[str, Any]:
    _require_exact_clean_source(document)
    if not re.fullmatch(r"[0-9]+", array_job_id) or not re.fullmatch(r"[0-9]+", collector_job_id):
        raise P1ExecutionError("scheduler job identities are invalid")
    body = {
        "schema_version": P1_SCHEDULER_SCHEMA,
        "p1_declaration_identity": document["p1_declaration_identity"],
        "p1_execution_source": document["p1_execution_source"],
        "lifecycle_validation_receipt_identity": document[
            "lifecycle_validation_receipt"
        ]["receipt_identity"],
        "array_job_id": array_job_id,
        "collector_job_id": collector_job_id,
        "resources": SLURM_GEOMETRY,
        "collector_resources": COLLECTOR_SLURM_GEOMETRY,
        "scientific_calls_at_record": 0,
    }
    record = {**body, "scheduler_record_identity": stable_hash(P1_SCHEDULER_SCHEMA, body)}
    destination = Path(document["output_root"]) / "scheduler" / "submission.json"
    if destination.exists() or destination.with_suffix(".sha256").exists():
        raise P1ExecutionError("scheduler provenance is append-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(record)
    atomic_write_bytes(destination, payload)
    atomic_write_bytes(
        destination.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii")
    )
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    declare = sub.add_parser("declare")
    for command in (preflight, declare):
        command.add_argument("--cohort-root", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--execution-commit", required=True)
        command.add_argument("--execution-tree", required=True)
    declare.add_argument("--lifecycle-receipt", type=Path, required=True)
    declare.add_argument("--write", type=Path)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--declaration", type=Path, required=True)
    sub.add_parser("preview", parents=[common])
    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--target-index", type=int, required=True)
    worker.add_argument("--execute", action="store_true")
    worker.add_argument("--confirm-approved", action="store_true")
    physical_task = sub.add_parser("physical-task", parents=[common])
    physical_task.add_argument("--physical-task-index", type=int, required=True)
    physical_task.add_argument("--execute", action="store_true")
    physical_task.add_argument("--confirm-approved", action="store_true")
    collect = sub.add_parser("collect", parents=[common])
    collect.add_argument("--write", action="store_true")
    inspect = sub.add_parser("inspect-restart", parents=[common])
    inspect.add_argument("--target-index", type=int, required=True)
    scheduler = sub.add_parser("scheduler-record", parents=[common])
    scheduler.add_argument("--array-job-id", required=True)
    scheduler.add_argument("--collector-job-id", required=True)
    sub.add_parser("p2-gate", parents=[common])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = build_zero_call_preflight(
            cohort_root=args.cohort_root,
            simulator_configuration_path=args.config,
            output_root=args.output_root,
            execution_source=args.execution_commit,
            execution_tree=args.execution_tree,
        )
    elif args.command == "declare":
        document = build_declaration(
            cohort_root=args.cohort_root,
            simulator_configuration_path=args.config,
            output_root=args.output_root,
            execution_source=args.execution_commit,
            execution_tree=args.execution_tree,
            lifecycle_receipt_path=args.lifecycle_receipt,
        )
        if args.write:
            write_declaration(args.write, document)
        result = preview(document)
    elif args.command == "physical-task":
        if not args.execute or not args.confirm_approved:
            raise P1ExecutionError("P1 physical task execution requires approval flags")
        returncode, result = execute_physical_task(
            args.declaration, physical_task_index=args.physical_task_index
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(returncode)
    else:
        document = load_declaration(args.declaration)
        if args.command == "preview":
            result = preview(document)
        elif args.command == "worker":
            if not args.execute:
                result = {**preview(document), "selected_target_index": args.target_index}
            else:
                if not args.confirm_approved:
                    raise P1ExecutionError("P1 worker execution requires approval flag")
                torch.set_num_threads(20)
                torch.set_num_interop_threads(1)
                result = execute_target(document, target_index=args.target_index)
        elif args.command == "collect":
            result = collect_population(document, write=args.write)
        elif args.command == "inspect-restart":
            result = inspect_restart(document, target_index=args.target_index)
        elif args.command == "scheduler-record":
            result = write_scheduler_record(
                document, array_job_id=args.array_job_id, collector_job_id=args.collector_job_id
            )
        elif args.command == "p2-gate":
            collection = load_passing_collection(document)
            result = {
                "p1_collection_identity": collection["collection_identity"],
                "p1_status": collection["status"],
                "p2_gate": collection["p2_gate"],
            }
        else:  # pragma: no cover
            raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "collect" and args.write and result.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
