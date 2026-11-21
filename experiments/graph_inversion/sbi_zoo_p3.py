"""Paper-120 SBI-zoo P3 held-out evaluation on the repaired SBI family."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
from time import perf_counter, process_time
from typing import Any, Mapping, Sequence

import torch

from mutaco import SimulationContext, canonical_json_bytes, checked_evaluate, stable_hash
from mutaco.atomic_io import atomic_write_bytes
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.methods import direct_posterior_decision as _decision_surface
from experiments.benchmarking.methods.sbi_zoo_accounting import WriteAheadScientificCallLedger
from experiments.benchmarking.methods.sbi_zoo_contract import BUDGETS, METHODS, PRIOR_LAW
from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter
from experiments.graph_inversion import sbi_zoo_p2 as p2
from experiments.graph_inversion.paper_v3_callbacks import _assessment_randomness_evidence
from experiments.paper.historical_workflows.repaired_sbi_assessment_preparation import (
    ASSESSMENT_SEEDS,
    SIMULATOR_CONFIGURATION_IDENTITY,
    SIMULATOR_DEFINITION_IDENTITY,
    TARGET_CONFIG,
    TARGET_CONFIG_SHA256,
    assessment_context,
)


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_CONTRACT_SOURCE = "d57313a2e9affdb1e6db30a045a8b1a6ef27da33"
P2_EXECUTION_SOURCE = "ea582fe46b0b1a3e819d0eb66931805215cfe5e5"
P2_COLLECTION_IDENTITY = (
    "mutaco-paper120-sbi-zoo-p2-collection-v1-"
    "7bc671817b515d988b13b8e842e3f28c34688744c8949db4d4a0324f0981590a"
)
P2_COLLECTION_SHA256 = "645418a40beb04604af75bd83a07d72d688ba9059dbad86c0f297f4b5ac883c9"
P3_SCHEMA = "mutaco-paper120-sbi-zoo-p3-declaration-v1"
P3_PREFLIGHT_SCHEMA = "mutaco-paper120-sbi-zoo-p3-preflight-v1"
P3_RESULT_SCHEMA = "mutaco-paper120-sbi-zoo-p3-result-v1"
P3_COLLECTION_SCHEMA = "mutaco-paper120-sbi-zoo-p3-collection-v1"
P3_SCHEDULER_SCHEMA = "mutaco-paper120-sbi-zoo-p3-scheduler-v1"
LIFECYCLE_RECEIPT_SCHEMA = "mutaco-truba-validation-receipt-v1"
TARGET_COUNT = 120
TOTAL_CALLS = 480
COHORT_ROOT = Path("experiments/graph_inversion/frozen_targets/paper120_v3")

SLURM_GEOMETRY = {
    "partition": "barbun",
    "nodes": 1,
    "ntasks": 1,
    "ntasks_per_node": 1,
    "slurm_cpus_per_task": 20,
    "runtime_threads_per_worker": 20,
    "requested_memory_per_node": "64G",
    "walltime": "00:30:00",
    "array": "0-59%20",
    "array_elements": 60,
    "maximum_concurrency": 20,
    "logical_target_mapping": ["task_id", "task_id+60"],
    "calls_per_logical_target": 4,
    "calls_per_array_task": 8,
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
    "dependency": "afterany_worker_array",
    "job_exclusive": False,
    "requeue": False,
}


class P3ExecutionError(RuntimeError):
    """Fail-closed P3 declaration, execution, or evidence error."""


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _require_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise P3ExecutionError(f"missing immutable artifact or sidecar: {path}")
    actual = file_hash(path)
    if sidecar.read_text(encoding="ascii").strip() != actual:
        raise P3ExecutionError(f"immutable artifact hash mismatch: {path}")
    return actual


def _require_external(path: Path) -> Path:
    destination = path.resolve(strict=False)
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError:
        return destination
    raise P3ExecutionError("P3 output must remain outside the source checkout")


def logical_target_positions(physical_task_index: int) -> tuple[int, int]:
    if physical_task_index not in range(60):
        raise P3ExecutionError("physical task index is outside 0..59")
    return physical_task_index, physical_task_index + 60


def _load_p2_inputs(
    p2_declaration_path: Path, p2_collection_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration_sha = _require_sidecar(p2_declaration_path)
    collection_sha = _require_sidecar(p2_collection_path)
    if collection_sha != P2_COLLECTION_SHA256:
        raise P3ExecutionError("P2 collection SHA-256 is not authoritative")
    declaration = p2.load_declaration(p2_declaration_path)
    collection = read_json(p2_collection_path)
    body = {key: value for key, value in collection.items() if key != "collection_identity"}
    if (
        declaration.get("p2_execution_source") != P2_EXECUTION_SOURCE
        or collection.get("collection_identity") != stable_hash(p2.P2_COLLECTION_SCHEMA, body)
        or collection.get("collection_identity") != P2_COLLECTION_IDENTITY
        or collection.get("p2_declaration_identity") != declaration["p2_declaration_identity"]
        or collection.get("status") != "PASS"
        or collection.get("p3_gate") != "OPEN"
        or collection.get("recommendation_count") != TOTAL_CALLS
        or collection.get("failed_components") != 0
        or collection.get("attempted_outcome_unknown") != 0
        or collection.get("physical_simulator_calls") != 0
        or collection.get("heldout_accesses") != 0
        or len(collection.get("recommendations", ())) != TOTAL_CALLS
    ):
        raise P3ExecutionError("P2 collection does not open the frozen P3 gate")
    return declaration, {
        **collection,
        "declaration_path": str(p2_declaration_path.resolve()),
        "declaration_sha256": declaration_sha,
        "collection_path": str(p2_collection_path.resolve()),
        "collection_sha256": collection_sha,
    }


def _recommendation_metadata_path(
    declaration: Mapping[str, Any], recommendation: Mapping[str, Any]
) -> Path:
    return (
        Path(declaration["output_root"])
        / "targets"
        / f"target-{int(recommendation['target_position']):03d}"
        / f"{recommendation['method']}-b{int(recommendation['budget'])}"
        / "recommendation"
        / "metadata.json"
    )


def _tasks(
    declaration: Mapping[str, Any], collection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    specs = {
        item["component_identity"]: item for item in declaration["components"]
    }
    tasks = []
    for recommendation in collection["recommendations"]:
        spec = specs.get(recommendation["component_identity"])
        if spec is None:
            raise P3ExecutionError("P2 recommendation component is absent from declaration")
        p2._load_recommendation(declaration, spec)
        metadata = read_json(_recommendation_metadata_path(declaration, recommendation))
        theta = metadata.get("sample_map", {}).get("decision", {}).get("recommended_theta")
        if (
            metadata.get("component_identity") != spec["component_identity"]
            or metadata.get("method") != spec["method"]
            or metadata.get("budget") != spec["budget"]
            or not isinstance(theta, list)
            or len(theta) != 4
            or not all(math.isfinite(float(value)) for value in theta)
        ):
            raise P3ExecutionError("P2 recommendation metadata is invalid")
        lower, upper = PRIOR_LAW.lower, PRIOR_LAW.upper
        if any(not lower[i] <= float(theta[i]) <= upper[i] for i in range(4)):
            raise P3ExecutionError("P2 recommendation violates physical support")
        context = assessment_context(
            budget=int(spec["budget"]),
            target_position=int(spec["target_position"]),
            target_id=str(spec["target_record_id"]),
        )
        body = {
            "target_position": spec["target_position"],
            "target_record_id": spec["target_record_id"],
            "budget": spec["budget"],
            "method": spec["method"],
            "component_identity": spec["component_identity"],
            "recommendation_identity": recommendation["recommendation_identity"],
            "recommended_theta": [float(value) for value in theta],
            "assessment_context": context,
            "nominal_assessment_calls": 1,
            "automatic_retry": False,
        }
        tasks.append({**body, "task_identity": stable_hash("mutaco-paper120-sbi-zoo-p3-task-v1", body)})
    expected = {
        (position, budget, method)
        for position in range(TARGET_COUNT)
        for budget in BUDGETS
        for method in METHODS
    }
    actual = {
        (int(item["target_position"]), int(item["budget"]), str(item["method"]))
        for item in tasks
    }
    if len(tasks) != TOTAL_CALLS or actual != expected or len({item["task_identity"] for item in tasks}) != TOTAL_CALLS:
        raise P3ExecutionError("P3 task enumeration is not exactly 480 unique calls")
    contexts = {
        (item["target_position"], item["budget"]): item["assessment_context"]
        for item in tasks
    }
    if len(contexts) != 240 or any(
        context["family_definition"]["assessment_campaign_seed"]
        != ASSESSMENT_SEEDS[budget]
        for (_, budget), context in contexts.items()
    ):
        raise P3ExecutionError("P3 repaired held-out context matrix is invalid")
    return tasks


def _load_lifecycle_receipt(path: Path, *, execution_source: str) -> dict[str, Any]:
    receipt_sha = _require_sidecar(path)
    receipt = read_json(path)
    body = {key: value for key, value in receipt.items() if key != "receipt_identity"}
    preview = receipt.get("worker_preview", {})
    if (
        receipt.get("receipt_identity") != stable_hash(LIFECYCLE_RECEIPT_SCHEMA, body)
        or receipt.get("schema_version") != LIFECYCLE_RECEIPT_SCHEMA
        or receipt.get("status") != "validated_zero_call"
        or receipt.get("source_commit") != execution_source
        or receipt.get("source_materialization", {}).get("source_commit") != execution_source
        or preview.get("schema_version") != P3_PREFLIGHT_SCHEMA
        or preview.get("p3_execution_source") != execution_source
        or preview.get("scientific_calls_executed") != 0
        or receipt.get("scientific_calls") != 0
        or receipt.get("job_submitted") is not False
    ):
        raise P3ExecutionError("lifecycle receipt is not a matching zero-call P3 receipt")
    return {
        "receipt_path": str(path.resolve()),
        "receipt_sha256": receipt_sha,
        "receipt_identity": receipt["receipt_identity"],
        "status": receipt["status"],
    }


def build_zero_call_preflight(
    *, p2_declaration_path: Path, p2_collection_path: Path, output_root: Path,
    execution_source: str, execution_tree: str,
) -> dict[str, Any]:
    output = _require_external(output_root)
    if output.exists():
        raise P3ExecutionError("P3 output root must be fresh")
    declaration, collection = _load_p2_inputs(p2_declaration_path, p2_collection_path)
    tasks = _tasks(declaration, collection)
    return {
        "schema_version": P3_PREFLIGHT_SCHEMA,
        "p3_execution_source": execution_source,
        "p3_execution_tree": execution_tree,
        "p2_collection_identity": collection["collection_identity"],
        "p2_collection_sha256": collection["collection_sha256"],
        "unique_recommendations": len({item["recommendation_identity"] for item in tasks}),
        "unique_heldout_slots": len({item["task_identity"] for item in tasks}),
        "maximum_physical_calls": TOTAL_CALLS,
        "heldout_family_roots": [ASSESSMENT_SEEDS[40], ASSESSMENT_SEEDS[100]],
        "slurm": SLURM_GEOMETRY,
        "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
        "scientific_calls_executed": 0,
        "job_submitted": False,
        "declaration_published": False,
    }


def build_declaration(
    *, p2_declaration_path: Path, p2_collection_path: Path, output_root: Path,
    execution_source: str, execution_tree: str, lifecycle_receipt_path: Path,
) -> dict[str, Any]:
    output = _require_external(output_root)
    if output.exists():
        raise P3ExecutionError("P3 output root must be fresh before declaration")
    p2_declaration, collection = _load_p2_inputs(p2_declaration_path, p2_collection_path)
    lifecycle = _load_lifecycle_receipt(lifecycle_receipt_path, execution_source=execution_source)
    tasks = _tasks(p2_declaration, collection)
    body = {
        "schema_version": P3_SCHEMA,
        "status": "declared_repaired_family_heldout_assessment",
        "scientific_contract_source": SCIENTIFIC_CONTRACT_SOURCE,
        "p3_execution_source": execution_source,
        "p3_execution_tree": execution_tree,
        "lifecycle_validation_receipt": lifecycle,
        "p2": {
            "execution_source": p2_declaration["p2_execution_source"],
            "declaration_identity": p2_declaration["p2_declaration_identity"],
            "declaration_path": collection["declaration_path"],
            "declaration_sha256": collection["declaration_sha256"],
            "collection_identity": collection["collection_identity"],
            "collection_path": collection["collection_path"],
            "collection_sha256": collection["collection_sha256"],
            "recommendations": TOTAL_CALLS,
            "read_only": True,
        },
        "tasks": tasks,
        "assessment": {
            "family_roots": {"40": ASSESSMENT_SEEDS[40], "100": ASSESSMENT_SEEDS[100]},
            "context_source": "existing_repaired_sbi_assessment_context_function",
            "simulator_definition_identity": SIMULATOR_DEFINITION_IDENTITY,
            "simulator_configuration_identity": SIMULATOR_CONFIGURATION_IDENTITY,
            "simulator_configuration_path": TARGET_CONFIG,
            "simulator_configuration_sha256": TARGET_CONFIG_SHA256,
            "calls_per_recommendation": 1,
            "maximum_calls": TOTAL_CALLS,
            "automatic_retry": False,
            "replacement_calls": False,
        },
        "execution": {
            "slurm": SLURM_GEOMETRY,
            "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_intraop_threads": 20,
                "torch_interop_threads": 1,
            },
        },
        "output_root": str(output),
        "p1_read_only": True,
        "p2_read_only": True,
        "analysis_authorized": False,
        "append_only": True,
    }
    return {**body, "p3_declaration_identity": stable_hash(P3_SCHEMA, body)}


def validate_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != "p3_declaration_identity"}
    if (
        document.get("schema_version") != P3_SCHEMA
        or document.get("p3_declaration_identity") != stable_hash(P3_SCHEMA, body)
        or document.get("scientific_contract_source") != SCIENTIFIC_CONTRACT_SOURCE
        or document.get("p2", {}).get("execution_source") != P2_EXECUTION_SOURCE
        or document.get("p2", {}).get("collection_identity") != P2_COLLECTION_IDENTITY
        or document.get("p2", {}).get("collection_sha256") != P2_COLLECTION_SHA256
        or len(document.get("tasks", ())) != TOTAL_CALLS
        or len({item.get("task_identity") for item in document.get("tasks", ())}) != TOTAL_CALLS
        or document.get("assessment", {}).get("maximum_calls") != TOTAL_CALLS
        or document.get("execution", {}).get("slurm") != SLURM_GEOMETRY
        or document.get("execution", {}).get("collector_slurm") != COLLECTOR_SLURM_GEOMETRY
    ):
        raise P3ExecutionError("P3 declaration is invalid")
    _load_lifecycle_receipt(
        Path(document["lifecycle_validation_receipt"]["receipt_path"]),
        execution_source=str(document["p3_execution_source"]),
    )
    p2_declaration, collection = _load_p2_inputs(
        Path(document["p2"]["declaration_path"]), Path(document["p2"]["collection_path"])
    )
    if p2_declaration["p2_declaration_identity"] != document["p2"]["declaration_identity"] or collection["collection_identity"] != document["p2"]["collection_identity"]:
        raise P3ExecutionError("P2 evidence changed after P3 declaration")
    return dict(document)


def write_declaration(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise P3ExecutionError("P3 declaration is append-only")
    payload = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))


def load_declaration(path: Path) -> dict[str, Any]:
    _require_sidecar(path)
    return validate_declaration(read_json(path))


def _require_exact_clean_source(document: Mapping[str, Any]) -> None:
    if _git("rev-parse", "HEAD") != document["p3_execution_source"] or _git("rev-parse", "HEAD^{tree}") != document["p3_execution_tree"] or _git("status", "--porcelain", "--untracked-files=all"):
        raise P3ExecutionError("P3 requires its exact clean execution source")


def initialize_output(document: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_clean_source(document)
    root = Path(document["output_root"])
    if root.exists():
        raise P3ExecutionError("P3 output root is not fresh")
    ledger = WriteAheadScientificCallLedger(root / "ledger", document["p3_declaration_identity"], TOTAL_CALLS)
    return {"ledger_identity": ledger.identity, "maximum_calls": TOTAL_CALLS, "scientific_calls_executed": 0}


def _task(document: Mapping[str, Any], target_position: int, budget: int, method: str) -> dict[str, Any]:
    matches = [item for item in document["tasks"] if item["target_position"] == target_position and item["budget"] == budget and item["method"] == method]
    if len(matches) != 1:
        raise P3ExecutionError("P3 task is missing or duplicated")
    return matches[0]


def _target(target_position: int, expected_id: str) -> dict[str, Any]:
    path = ROOT / COHORT_ROOT / f"target_{target_position:03d}.json"
    _require_sidecar(path)
    target = read_json(path)
    if target.get("target_index") != target_position or target.get("target_record_id") != expected_id:
        raise P3ExecutionError("Paper-120 target identity mismatch")
    return target


def _result_path(document: Mapping[str, Any], task: Mapping[str, Any]) -> Path:
    return Path(document["output_root"]) / "results" / f"target-{int(task['target_position']):03d}" / f"{task['method']}-b{int(task['budget'])}.json"


def _publish_result(
    path: Path, *, document: Mapping[str, Any], task: Mapping[str, Any], target: Mapping[str, Any], evaluation: Any, wall_seconds: float, cpu_seconds: float,
) -> dict[str, Any]:
    response = [float(value) for value in evaluation.values.detach().cpu().tolist()]
    goal = [float(value) for value in target["target_response"]]
    theta = [float(value) for value in task["recommended_theta"]]
    if len(response) != 20 or len(goal) != 20 or not all(math.isfinite(value) for value in response + goal + theta):
        raise P3ExecutionError("P3 response, target, or recommendation is invalid")
    randomness = _assessment_randomness_evidence(task["assessment_context"], evaluation.metadata)
    differences = [value - expected for value, expected in zip(response, goal)]
    truth = target.get("theta_star")
    parameter_error = None if truth is None else math.sqrt(sum((value - float(expected)) ** 2 for value, expected in zip(theta, truth)))
    body = {
        "schema_version": P3_RESULT_SCHEMA,
        "p3_declaration_identity": document["p3_declaration_identity"],
        "task_identity": task["task_identity"],
        "target_position": task["target_position"],
        "target_record_id": task["target_record_id"],
        "budget": task["budget"],
        "method": task["method"],
        "recommendation_identity": task["recommendation_identity"],
        "recommended_theta": theta,
        "assessment_context": task["assessment_context"],
        "assessment_response": response,
        "assessment_randomness_evidence": randomness["compact"],
        "assessment_randomness_raw": randomness["raw"],
        "response_error_l1_sum": sum(abs(value) for value in differences),
        "response_error_mae": sum(abs(value) for value in differences) / 20,
        "response_error_l2": math.sqrt(sum(value * value for value in differences)),
        "parameter_error_l2": parameter_error,
        "timing": {"wall_seconds": wall_seconds, "process_cpu_seconds": cpu_seconds},
        "physical_calls": 1,
        "retry_calls": 0,
    }
    result = {**body, "result_identity": stable_hash(P3_RESULT_SCHEMA, body)}
    if path.exists() or path.with_suffix(".sha256").exists():
        raise P3ExecutionError("P3 result is append-only")
    payload = canonical_json_bytes(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))
    if _require_sidecar(path) != _sha(payload) or read_json(path) != result:
        raise P3ExecutionError("P3 result reload verification failed")
    return {"result_identity": result["result_identity"], "result_sha256": _sha(payload)}


def execute_physical_task(document: Mapping[str, Any], *, physical_task_index: int) -> tuple[int, dict[str, Any]]:
    _require_exact_clean_source(document)
    config = ROOT / document["assessment"]["simulator_configuration_path"]
    if file_hash(config) != document["assessment"]["simulator_configuration_sha256"]:
        raise P3ExecutionError("P3 simulator configuration mismatch")
    simulator = GraphScientificD4Adapter().build_target_simulator(read_json(config), device=torch.device("cpu"), dtype=torch.float32)
    if simulator.capabilities.simulator_identity != SIMULATOR_DEFINITION_IDENTITY:
        raise P3ExecutionError("P3 simulator identity mismatch")
    ledger = WriteAheadScientificCallLedger(Path(document["output_root"]) / "ledger", document["p3_declaration_identity"], TOTAL_CALLS)
    outcomes = []
    all_succeeded = True
    for target_position in logical_target_positions(physical_task_index):
        for budget in BUDGETS:
            for method in METHODS:
                task = _task(document, target_position, budget, method)
                target = _target(target_position, task["target_record_id"])
                slot = f"target-{target_position:03d}-{method}-b{budget}"
                wall_start, cpu_start = perf_counter(), process_time()
                try:
                    def invoke() -> Any:
                        context = task["assessment_context"]
                        return checked_evaluate(
                            simulator, torch.tensor(task["recommended_theta"], dtype=torch.float32),
                            device=torch.device("cpu"), dtype=torch.float32,
                            context=SimulationContext(
                                str(context["family_identity"]), str(context["purpose"]),
                                int(context["logical_index"]), int(context["replicate"]),
                                int(context["evaluation_seed"]),
                            ),
                        )

                    result_path = _result_path(document, task)
                    persisted: dict[str, Any] = {}

                    def persist(evaluation: Any) -> Mapping[str, Any]:
                        persisted.update(_publish_result(
                            result_path, document=document, task=task, target=target,
                            evaluation=evaluation, wall_seconds=perf_counter() - wall_start,
                            cpu_seconds=process_time() - cpu_start,
                        ))
                        return {"task_identity": task["task_identity"], **persisted}

                    ledger.invoke_unattempted_once(
                        slot,
                        intent={
                            "p3_declaration_identity": document["p3_declaration_identity"],
                            "task_identity": task["task_identity"],
                            "recommendation_identity": task["recommendation_identity"],
                            "assessment_context_identity": task["assessment_context"]["family_identity"],
                        },
                        invoke=invoke,
                        persist_outcome=persist,
                    )
                    outcomes.append({"task_identity": task["task_identity"], "status": "success", **persisted})
                except Exception as exc:
                    all_succeeded = False
                    outcomes.append({"task_identity": task["task_identity"], "status": ledger.slot_status(slot), "failure_type": type(exc).__name__, "failure_explanation": str(exc)})
    return (0 if all_succeeded else 1), {
        "physical_task_index": physical_task_index,
        "logical_target_positions": list(logical_target_positions(physical_task_index)),
        "outcomes": outcomes,
        "all_succeeded": all_succeeded,
        "automatic_retry": False,
        "slurm_requeue": False,
    }


def _load_result(document: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    path = _result_path(document, task)
    result_sha = _require_sidecar(path)
    result = read_json(path)
    body = {key: value for key, value in result.items() if key != "result_identity"}
    if (
        result.get("result_identity") != stable_hash(P3_RESULT_SCHEMA, body)
        or result.get("p3_declaration_identity") != document["p3_declaration_identity"]
        or result.get("task_identity") != task["task_identity"]
        or result.get("recommendation_identity") != task["recommendation_identity"]
        or result.get("assessment_context") != task["assessment_context"]
        or result.get("physical_calls") != 1
        or result.get("retry_calls") != 0
    ):
        raise P3ExecutionError("P3 result identity validation failed")
    return {**result, "result_sha256": result_sha}


def collect_population(document: Mapping[str, Any], *, write: bool = False) -> dict[str, Any]:
    validate_declaration(document)
    ledger = WriteAheadScientificCallLedger(Path(document["output_root"]) / "ledger", document["p3_declaration_identity"], TOTAL_CALLS)
    counts = {"NLE_B40": 0, "NLE_B100": 0, "NRE_B40": 0, "NRE_B100": 0}
    records, errors = [], []
    for task in document["tasks"]:
        slot = f"target-{int(task['target_position']):03d}-{task['method']}-b{int(task['budget'])}"
        try:
            if ledger.slot_status(slot) != "success":
                raise P3ExecutionError(f"held-out slot status is {ledger.slot_status(slot)}")
            result = _load_result(document, task)
            terminal = ledger.terminal_payload(slot)
            if terminal is None or terminal.get("result_identity") != result["result_identity"] or terminal.get("result_sha256") != result["result_sha256"]:
                raise P3ExecutionError("ledger success does not bind the P3 result")
            label = ("NLE" if task["method"] == METHODS[0] else "NRE") + f"_B{task['budget']}"
            counts[label] += 1
            records.append({key: result[key] for key in (
                "result_identity", "result_sha256", "task_identity", "target_position",
                "target_record_id", "budget", "method", "recommendation_identity",
                "recommended_theta", "assessment_context", "assessment_response",
                "assessment_randomness_evidence", "response_error_l1_sum",
                "response_error_mae", "response_error_l2", "parameter_error_l2", "timing",
            )})
        except Exception as exc:
            errors.append({"task_identity": task["task_identity"], "failure_type": type(exc).__name__, "failure_explanation": str(exc)})
    accounting = ledger.accounting()
    expected_counts = {"NLE_B40": 120, "NLE_B100": 120, "NRE_B40": 120, "NRE_B100": 120}
    passed = not errors and counts == expected_counts and len(records) == TOTAL_CALLS and accounting == {
        "maximum_calls": 480, "attempted_calls": 480, "successful_calls": 480,
        "failed_calls": 0, "attempted_outcome_unknown": 0, "retry_calls": 0,
    }
    body = {
        "schema_version": P3_COLLECTION_SCHEMA,
        "p3_declaration_identity": document["p3_declaration_identity"],
        "p3_execution_source": document["p3_execution_source"],
        "p2_collection_identity": document["p2"]["collection_identity"],
        "status": "PASS" if passed else "FAILED",
        "counts": counts,
        "record_count": len(records),
        "physical_calls": accounting,
        "records": records,
        "errors": errors,
        "analysis_authorized": False,
    }
    collection = {**body, "collection_identity": stable_hash(P3_COLLECTION_SCHEMA, body)}
    if write:
        destination = Path(document["output_root"]) / "collection" / "manifest.json"
        if destination.exists() or destination.with_suffix(".sha256").exists():
            raise P3ExecutionError("P3 collection is append-only")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(collection)
        atomic_write_bytes(destination, payload)
        atomic_write_bytes(destination.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))
    return collection


def write_scheduler_record(document: Mapping[str, Any], *, array_job_id: str, collector_job_id: str) -> dict[str, Any]:
    _require_exact_clean_source(document)
    if not re.fullmatch(r"[0-9]+", array_job_id) or not re.fullmatch(r"[0-9]+", collector_job_id):
        raise P3ExecutionError("P3 scheduler job IDs are invalid")
    body = {
        "schema_version": P3_SCHEDULER_SCHEMA,
        "p3_declaration_identity": document["p3_declaration_identity"],
        "p3_execution_source": document["p3_execution_source"],
        "array_job_id": array_job_id,
        "collector_job_id": collector_job_id,
        "resources": SLURM_GEOMETRY,
        "collector_resources": COLLECTOR_SLURM_GEOMETRY,
    }
    record = {**body, "scheduler_record_identity": stable_hash(P3_SCHEDULER_SCHEMA, body)}
    destination = Path(document["output_root"]) / "scheduler" / "submission.json"
    if destination.exists() or destination.with_suffix(".sha256").exists():
        raise P3ExecutionError("P3 scheduler record is append-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(record)
    atomic_write_bytes(destination, payload)
    atomic_write_bytes(destination.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))
    return record


def preview(document: Mapping[str, Any]) -> dict[str, Any]:
    validate_declaration(document)
    return {
        "schema_version": P3_PREFLIGHT_SCHEMA,
        "p3_declaration_identity": document["p3_declaration_identity"],
        "p3_execution_source": document["p3_execution_source"],
        "p2_collection_identity": document["p2"]["collection_identity"],
        "unique_recommendations": len({item["recommendation_identity"] for item in document["tasks"]}),
        "unique_heldout_slots": len({item["task_identity"] for item in document["tasks"]}),
        "maximum_physical_calls": TOTAL_CALLS,
        "heldout_family_roots": [ASSESSMENT_SEEDS[40], ASSESSMENT_SEEDS[100]],
        "slurm": SLURM_GEOMETRY,
        "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
        "scientific_calls_executed": 0,
        "job_submitted": False,
        "declaration_published": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "declare"):
        command = sub.add_parser(name)
        command.add_argument("--p2-declaration", type=Path, required=True)
        command.add_argument("--p2-collection", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--execution-commit", required=True)
        command.add_argument("--execution-tree", required=True)
        if name == "declare":
            command.add_argument("--lifecycle-receipt", type=Path, required=True)
            command.add_argument("--write", type=Path)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--declaration", type=Path, required=True)
    sub.add_parser("preview", parents=[common])
    sub.add_parser("initialize", parents=[common])
    task = sub.add_parser("physical-task", parents=[common])
    task.add_argument("--physical-task-index", type=int, required=True)
    task.add_argument("--execute", action="store_true")
    collect = sub.add_parser("collect", parents=[common])
    collect.add_argument("--write", action="store_true")
    scheduler = sub.add_parser("scheduler-record", parents=[common])
    scheduler.add_argument("--array-job-id", required=True)
    scheduler.add_argument("--collector-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = build_zero_call_preflight(
            p2_declaration_path=args.p2_declaration, p2_collection_path=args.p2_collection,
            output_root=args.output_root, execution_source=args.execution_commit,
            execution_tree=args.execution_tree,
        )
    elif args.command == "declare":
        document = build_declaration(
            p2_declaration_path=args.p2_declaration, p2_collection_path=args.p2_collection,
            output_root=args.output_root, execution_source=args.execution_commit,
            execution_tree=args.execution_tree, lifecycle_receipt_path=args.lifecycle_receipt,
        )
        if args.write:
            write_declaration(args.write, document)
        result = preview(document)
    elif args.command == "physical-task":
        if not args.execute:
            raise P3ExecutionError("P3 physical task requires explicit execution flag")
        document = load_declaration(args.declaration)
        torch.set_num_threads(20)
        torch.set_num_interop_threads(1)
        returncode, result = execute_physical_task(document, physical_task_index=args.physical_task_index)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(returncode)
    else:
        document = load_declaration(args.declaration)
        if args.command == "preview":
            result = preview(document)
        elif args.command == "initialize":
            result = initialize_output(document)
        elif args.command == "collect":
            result = collect_population(document, write=args.write)
        elif args.command == "scheduler-record":
            result = write_scheduler_record(document, array_job_id=args.array_job_id, collector_job_id=args.collector_job_id)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "collect" and args.write and result.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
