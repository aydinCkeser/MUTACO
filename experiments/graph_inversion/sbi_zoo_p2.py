"""Paper-120 SBI-zoo P2: zero-call training and frozen recommendations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch

from mutaco import canonical_json_bytes, stable_hash
from mutaco.atomic_io import atomic_write_bytes
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.methods.sbi_zoo_contract import (
    BUDGETS,
    METHODS,
    POSTERIOR_SAMPLE_COUNT,
    PRIOR_LAW,
    dataset_seed_deriver,
    estimator_contract,
    estimator_seed,
    mcmc_contract,
    randomness_contract,
    recommendation_seed,
    require_pinned_runtime,
)
from experiments.benchmarking.methods.sbi_zoo_dataset import (
    load_shared_training_dataset,
)
from experiments.benchmarking.methods.sbi_zoo_inference import (
    build_mcmc_posterior,
    recommend_sample_map,
    train_single_round,
)
from experiments.graph_inversion import sbi_zoo_p1 as p1


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_CONTRACT_SOURCE = "d57313a2e9affdb1e6db30a045a8b1a6ef27da33"
P1_EXECUTION_SOURCE = "55ce098f53c6e14749dcbc1aa8fa54b2de902026"
P1_COLLECTION_IDENTITY = (
    "mutaco-paper120-sbi-zoo-p1-collection-v1-"
    "07baf458f14d4332db73c136736822113934e208cdc419c1b4842267397dbd7e"
)
P1_COLLECTION_SHA256 = "2819ef58f65d0f2f4f46e8a179afe2e8d1c50e4596b3c6c7c5aaec28bef89907"
P2_SCHEMA = "mutaco-paper120-sbi-zoo-p2-declaration-v1"
P2_PREFLIGHT_SCHEMA = "mutaco-paper120-sbi-zoo-p2-preflight-v1"
P2_EVENT_SCHEMA = "mutaco-paper120-sbi-zoo-p2-method-event-v1"
P2_RECOMMENDATION_SCHEMA = "mutaco-paper120-sbi-zoo-p2-recommendation-v1"
P2_COLLECTION_SCHEMA = "mutaco-paper120-sbi-zoo-p2-collection-v1"
P2_SCHEDULER_SCHEMA = "mutaco-paper120-sbi-zoo-p2-scheduler-v1"
LIFECYCLE_RECEIPT_SCHEMA = "mutaco-truba-validation-receipt-v1"
TARGET_COUNT = 120
EXPECTED_RECOMMENDATIONS = 480
REPAIRED_HELDOUT_ROOTS = (92021540, 92021600)

SLURM_GEOMETRY = {
    "partition": "barbun",
    "nodes": 1,
    "ntasks": 1,
    "ntasks_per_node": 1,
    "slurm_cpus_per_task": 20,
    "runtime_threads_per_worker": 20,
    "requested_memory_per_node": "64G",
    "walltime": "01:00:00",
    "array": "0-59%10",
    "array_elements": 60,
    "maximum_concurrency": 10,
    "logical_targets": 120,
    "logical_targets_per_task": 2,
    "logical_target_mapping": ["task_id", "task_id+60"],
    "worker_execution": "two_targets_sequential_four_components_per_target",
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


class P2ExecutionError(RuntimeError):
    """Fail-closed P2 declaration, execution, or evidence error."""


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _require_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise P2ExecutionError(f"missing immutable artifact or sidecar: {path}")
    actual = file_hash(path)
    if sidecar.read_text(encoding="ascii").strip() != actual:
        raise P2ExecutionError(f"immutable artifact hash mismatch: {path}")
    return actual


def _require_external(path: Path) -> Path:
    destination = path.resolve(strict=False)
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError:
        return destination
    raise P2ExecutionError("P2 output must remain outside the source checkout")


def logical_target_positions(physical_task_index: int) -> tuple[int, int]:
    if physical_task_index not in range(60):
        raise P2ExecutionError("physical task index is outside 0..59")
    return physical_task_index, physical_task_index + 60


def _load_p1_inputs(
    p1_declaration_path: Path, p1_collection_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration_sha = _require_sidecar(p1_declaration_path)
    collection_sha = _require_sidecar(p1_collection_path)
    if collection_sha != P1_COLLECTION_SHA256:
        raise P2ExecutionError("P1 collection SHA-256 is not authoritative")
    declaration = p1.load_declaration(p1_declaration_path)
    collection = p1.load_passing_collection(declaration)
    supplied = read_json(p1_collection_path)
    if supplied != collection:
        raise P2ExecutionError("supplied P1 collection is not the unique PASS collection")
    if (
        declaration.get("p1_execution_source") != P1_EXECUTION_SOURCE
        or collection.get("collection_identity") != P1_COLLECTION_IDENTITY
        or collection.get("status") != "PASS"
        or collection.get("p2_gate") != "OPEN"
        or collection.get("datasets") != 240
        or collection.get("successful_calls") != 16_800
        or collection.get("failed_calls") != 0
        or collection.get("attempted_outcome_unknown") != 0
        or collection.get("heldout_calls") != 0
        or collection.get("heldout_accesses") != 0
        or len(collection.get("dataset_receipts", ())) != 240
    ):
        raise P2ExecutionError("P1 input collection does not open the frozen P2 gate")
    return declaration, {
        **collection,
        "declaration_path": str(p1_declaration_path.resolve()),
        "declaration_sha256": declaration_sha,
        "collection_path": str(p1_collection_path.resolve()),
        "collection_sha256": collection_sha,
    }


def _load_lifecycle_receipt(path: Path, *, execution_source: str) -> dict[str, Any]:
    receipt_sha = _require_sidecar(path)
    receipt = read_json(path)
    body = {key: value for key, value in receipt.items() if key != "receipt_identity"}
    if receipt.get("receipt_identity") != stable_hash(LIFECYCLE_RECEIPT_SCHEMA, body):
        raise P2ExecutionError("lifecycle receipt identity mismatch")
    preview = receipt.get("worker_preview", {})
    materialization = receipt.get("source_materialization", {})
    if (
        receipt.get("schema_version") != LIFECYCLE_RECEIPT_SCHEMA
        or receipt.get("status") != "validated_zero_call"
        or receipt.get("source_commit") != execution_source
        or materialization.get("source_commit") != execution_source
        or preview.get("schema_version") != P2_PREFLIGHT_SCHEMA
        or preview.get("p2_execution_source") != execution_source
        or preview.get("scientific_calls_executed") != 0
        or preview.get("heldout_accesses") != 0
        or receipt.get("scientific_calls") != 0
        or receipt.get("job_submitted") is not False
        or receipt.get("submission_authorized") is not False
    ):
        raise P2ExecutionError("lifecycle receipt is not a matching zero-call P2 receipt")
    return {
        "receipt_path": str(path.resolve()),
        "receipt_sha256": receipt_sha,
        "receipt_identity": receipt["receipt_identity"],
        "status": receipt["status"],
    }


def _component_specs(
    p1_declaration: Mapping[str, Any], collection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    receipts = {
        (int(item["target_position"]), int(item["budget"])): item
        for item in collection["dataset_receipts"]
    }
    deriver = dataset_seed_deriver(
        str(p1_declaration["p1_declaration_identity"]),
        int(p1_declaration["randomness"]["root_seed"]),
    )
    specs = []
    for target in p1_declaration["cohort"]["targets"]:
        position = int(target["target_position"])
        for budget in BUDGETS:
            receipt = receipts[(position, budget)]
            for method in METHODS:
                config = estimator_contract(method, budget)
                config_identity = stable_hash("sbi-zoo-method-configuration-v1", config)
                body = {
                    "target_position": position,
                    "target_record_id": target["target_record_id"],
                    "budget": budget,
                    "method": method,
                    "dataset_identity": receipt["dataset_identity"],
                    "dataset_content_sha256": receipt["canonical_content_sha256"],
                    "dataset_freeze_identity": receipt["freeze_identity"],
                    "training_seed": estimator_seed(
                        deriver,
                        method=method,
                        target_record_id=target["target_record_id"],
                        budget=budget,
                    ),
                    "recommendation_seed": recommendation_seed(
                        deriver,
                        method=method,
                        target_record_id=target["target_record_id"],
                        budget=budget,
                    ),
                    "method_configuration_identity": config_identity,
                }
                specs.append(
                    {
                        **body,
                        "component_identity": stable_hash(
                            "mutaco-paper120-sbi-zoo-p2-component-v1", body
                        ),
                    }
                )
    if len(specs) != EXPECTED_RECOMMENDATIONS or len(
        {item["component_identity"] for item in specs}
    ) != EXPECTED_RECOMMENDATIONS:
        raise P2ExecutionError("P2 component enumeration is not exactly 480 unique identities")
    return specs


def build_zero_call_preflight(
    *,
    p1_declaration_path: Path,
    p1_collection_path: Path,
    output_root: Path,
    execution_source: str,
    execution_tree: str,
) -> dict[str, Any]:
    output = _require_external(output_root)
    if output.exists():
        raise P2ExecutionError("P2 output root must be fresh")
    declaration, collection = _load_p1_inputs(p1_declaration_path, p1_collection_path)
    specs = _component_specs(declaration, collection)
    return {
        "schema_version": P2_PREFLIGHT_SCHEMA,
        "p2_execution_source": execution_source,
        "p2_execution_tree": execution_tree,
        "p1_collection_identity": collection["collection_identity"],
        "p1_collection_sha256": collection["collection_sha256"],
        "p1_datasets_addressable": len(collection["dataset_receipts"]),
        "unique_p2_components": len({item["component_identity"] for item in specs}),
        "expected_recommendations": EXPECTED_RECOMMENDATIONS,
        "slurm": SLURM_GEOMETRY,
        "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
        "simulator_access": "forbidden_no_import_or_construction_path",
        "heldout_roots_accessible": False,
        "heldout_accesses": 0,
        "scientific_calls_executed": 0,
        "job_submitted": False,
        "declaration_published": False,
    }


def build_declaration(
    *,
    p1_declaration_path: Path,
    p1_collection_path: Path,
    output_root: Path,
    execution_source: str,
    execution_tree: str,
    lifecycle_receipt_path: Path,
) -> dict[str, Any]:
    output = _require_external(output_root)
    if output.exists():
        raise P2ExecutionError("P2 output root must be fresh before declaration")
    p1_declaration, collection = _load_p1_inputs(p1_declaration_path, p1_collection_path)
    lifecycle = _load_lifecycle_receipt(
        lifecycle_receipt_path, execution_source=execution_source
    )
    body = {
        "schema_version": P2_SCHEMA,
        "status": "declared_zero_simulator_call_training_and_recommendation",
        "scientific_contract_source": SCIENTIFIC_CONTRACT_SOURCE,
        "p2_execution_source": execution_source,
        "p2_execution_tree": execution_tree,
        "lifecycle_validation_receipt": lifecycle,
        "p1": {
            "execution_source": p1_declaration["p1_execution_source"],
            "declaration_identity": p1_declaration["p1_declaration_identity"],
            "declaration_path": collection["declaration_path"],
            "declaration_sha256": collection["declaration_sha256"],
            "collection_identity": collection["collection_identity"],
            "collection_path": collection["collection_path"],
            "collection_sha256": collection["collection_sha256"],
            "complete_content_identity": p1_declaration["p1_complete_content_identity"],
            "datasets": 240,
            "successful_calls": 16_800,
            "read_only": True,
        },
        "cohort": {
            "identity": p1_declaration["cohort"]["identity"],
            "manifest_sha256": p1_declaration["cohort"]["manifest_sha256"],
            "target_count": TARGET_COUNT,
            "target_record_ids": [
                item["target_record_id"] for item in p1_declaration["cohort"]["targets"]
            ],
        },
        "randomness": {
            "source_identity": p1_declaration["p1_declaration_identity"],
            "root_seed": p1_declaration["randomness"]["root_seed"],
            "contract": randomness_contract(),
            "execution_source_does_not_perturb_seeds": True,
        },
        "components": _component_specs(p1_declaration, collection),
        "training": {
            method: {str(budget): estimator_contract(method, budget) for budget in BUDGETS}
            for method in METHODS
        },
        "mcmc": mcmc_contract(),
        "recommendation": {
            "policy": "posterior_sample_map_physical_v1",
            "candidate_count": POSTERIOR_SAMPLE_COUNT,
            "candidate_order": "20_chains_times_500_chain_major_C_order",
            "physical_coordinate_scoring": True,
            "transform_jacobian_in_final_ranking": False,
            "tie_break": "earliest_exact_score_tie",
            "fallback": None,
        },
        "execution": {
            "slurm": SLURM_GEOMETRY,
            "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "dependencies": require_pinned_runtime(),
                "torch_intraop_threads": 20,
                "torch_interop_threads": 1,
                "thread_environment": {
                    name: os.environ.get(name, "20")
                    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
            },
        },
        "output_root": str(output),
        "expected_recommendations": EXPECTED_RECOMMENDATIONS,
        "physical_simulator_calls": 0,
        "heldout_firewall": {
            "heldout_calls": 0,
            "heldout_accesses": 0,
            "repaired_family_roots_forbidden": list(REPAIRED_HELDOUT_ROOTS),
        },
        "p3_gate": "blocked_until_480_recommendations_are_collected_and_frozen",
        "append_only": True,
    }
    return {**body, "p2_declaration_identity": stable_hash(P2_SCHEMA, body)}


def validate_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != "p2_declaration_identity"}
    identity = stable_hash(P2_SCHEMA, body)
    if (
        document.get("schema_version") != P2_SCHEMA
        or document.get("p2_declaration_identity") != identity
        or document.get("scientific_contract_source") != SCIENTIFIC_CONTRACT_SOURCE
        or document.get("p1", {}).get("execution_source") != P1_EXECUTION_SOURCE
        or document.get("p1", {}).get("collection_identity") != P1_COLLECTION_IDENTITY
        or document.get("p1", {}).get("collection_sha256") != P1_COLLECTION_SHA256
        or document.get("expected_recommendations") != EXPECTED_RECOMMENDATIONS
        or len(document.get("components", ())) != EXPECTED_RECOMMENDATIONS
        or len({item.get("component_identity") for item in document.get("components", ())})
        != EXPECTED_RECOMMENDATIONS
        or document.get("physical_simulator_calls") != 0
        or document.get("heldout_firewall", {}).get("heldout_accesses") != 0
        or document.get("execution", {}).get("slurm") != SLURM_GEOMETRY
        or document.get("execution", {}).get("collector_slurm")
        != COLLECTOR_SLURM_GEOMETRY
    ):
        raise P2ExecutionError("P2 declaration is invalid")
    _load_lifecycle_receipt(
        Path(document["lifecycle_validation_receipt"]["receipt_path"]),
        execution_source=str(document["p2_execution_source"]),
    )
    p1_declaration, p1_collection = _load_p1_inputs(
        Path(document["p1"]["declaration_path"]),
        Path(document["p1"]["collection_path"]),
    )
    if (
        p1_declaration["p1_declaration_identity"]
        != document["p1"]["declaration_identity"]
        or p1_collection["collection_identity"]
        != document["p1"]["collection_identity"]
    ):
        raise P2ExecutionError("P1 declaration or collection changed after P2 declaration")
    return dict(document)


def write_declaration(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise P2ExecutionError("P2 declaration is append-only")
    payload = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))


def load_declaration(path: Path) -> dict[str, Any]:
    _require_sidecar(path)
    return validate_declaration(read_json(path))


def _require_exact_clean_source(document: Mapping[str, Any]) -> None:
    if (
        _git("rev-parse", "HEAD") != document["p2_execution_source"]
        or _git("rev-parse", "HEAD^{tree}") != document["p2_execution_tree"]
        or _git("status", "--porcelain", "--untracked-files=all")
    ):
        raise P2ExecutionError("P2 requires its exact clean execution source")


def _component_spec(
    document: Mapping[str, Any], *, target_position: int, budget: int, method: str
) -> dict[str, Any]:
    matches = [
        item
        for item in document["components"]
        if item["target_position"] == target_position
        and item["budget"] == budget
        and item["method"] == method
    ]
    if len(matches) != 1:
        raise P2ExecutionError("P2 component identity is missing or duplicated")
    return matches[0]


def _p1_dataset_path(document: Mapping[str, Any], target_position: int, budget: int) -> Path:
    p1_declaration = p1.load_declaration(Path(document["p1"]["declaration_path"]))
    return (
        Path(p1_declaration["output_root"])
        / "targets"
        / f"target-{target_position:03d}"
        / f"b{budget}"
        / "artifact"
    )


def _target_response(document: Mapping[str, Any], target_position: int) -> torch.Tensor:
    p1_declaration = p1.load_declaration(Path(document["p1"]["declaration_path"]))
    target = p1_declaration["cohort"]["targets"][target_position]
    path = Path(target["target_artifact_path"])
    if _require_sidecar(path) != target["target_artifact_sha256"]:
        raise P2ExecutionError("Paper-120 target artifact changed after P1")
    response = torch.tensor(read_json(path)["target_response"], dtype=torch.float32)
    if response.shape != (20,) or not bool(torch.isfinite(response).all()):
        raise P2ExecutionError("Paper-120 target response is invalid")
    return response.contiguous()


def _event(root: Path, ordinal: int, status: str, body: Mapping[str, Any]) -> None:
    path = root / f"event-{ordinal:02d}-{status}.json"
    if path.exists():
        raise P2ExecutionError("P2 method event is append-only")
    value = {**dict(body), "status": status}
    value["event_identity"] = stable_hash(P2_EVENT_SCHEMA, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, canonical_json_bytes(value))


def _publish_recommendation(
    root: Path,
    *,
    document: Mapping[str, Any],
    spec: Mapping[str, Any],
    trained: Any,
    recommendation: Any,
) -> dict[str, Any]:
    destination = root / "recommendation"
    if destination.exists():
        raise P2ExecutionError("P2 recommendation already exists")
    samples = np.asarray(
        recommendation.samples.detach().cpu().contiguous().numpy(), dtype="<f4", order="C"
    ).tobytes(order="C")
    buffer = io.BytesIO()
    torch.save(trained.estimator.state_dict(), buffer)
    model = buffer.getvalue()
    metadata = {
        "schema_version": P2_RECOMMENDATION_SCHEMA,
        "p2_declaration_identity": document["p2_declaration_identity"],
        "component_identity": spec["component_identity"],
        "target_position": spec["target_position"],
        "target_record_id": spec["target_record_id"],
        "method": trained.method,
        "budget": trained.budget,
        "dataset_receipt": trained.dataset_receipt,
        "dataset_freeze_identity": spec["dataset_freeze_identity"],
        "training_seed": trained.training_seed,
        "recommendation_seed": recommendation.recommendation_seed,
        "method_configuration_identity": spec["method_configuration_identity"],
        "effective_configuration": trained.effective_configuration,
        "mcmc_configuration": recommendation.mcmc_configuration,
        "posterior_candidate_count": POSTERIOR_SAMPLE_COUNT,
        "posterior_candidate_shape": [POSTERIOR_SAMPLE_COUNT, 4],
        "candidate_order": "20_chains_times_500_chain_major_C_order",
        "sample_map": asdict(recommendation.sample_map),
        "physical_simulator_calls": 0,
        "heldout_calls": 0,
        "heldout_accesses": 0,
        "immutable": True,
    }
    metadata_bytes = canonical_json_bytes(metadata)
    manifest_body = {
        "schema_version": P2_RECOMMENDATION_SCHEMA,
        "p2_declaration_identity": document["p2_declaration_identity"],
        "component_identity": spec["component_identity"],
        "target_position": spec["target_position"],
        "target_record_id": spec["target_record_id"],
        "method": spec["method"],
        "budget": spec["budget"],
        "metadata_sha256": _sha(metadata_bytes),
        "samples_sha256": _sha(samples),
        "model_state_sha256": _sha(model),
    }
    manifest = {
        **manifest_body,
        "recommendation_identity": stable_hash(P2_RECOMMENDATION_SCHEMA, manifest_body),
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        atomic_write_bytes(temporary / "metadata.json", metadata_bytes)
        atomic_write_bytes(temporary / "samples.f32le", samples)
        atomic_write_bytes(temporary / "model-state.pt", model)
        atomic_write_bytes(temporary / "manifest.json", canonical_json_bytes(manifest))
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def execute_component(
    document: Mapping[str, Any], *, target_position: int, budget: int, method: str
) -> dict[str, Any]:
    _require_exact_clean_source(document)
    spec = _component_spec(
        document, target_position=target_position, budget=budget, method=method
    )
    dataset = load_shared_training_dataset(
        _p1_dataset_path(document, target_position, budget)
    )
    if (
        dataset.identity != spec["dataset_identity"]
        or dataset.content_sha256 != spec["dataset_content_sha256"]
        or dataset.metadata.get("target_record_id") != spec["target_record_id"]
        or dataset.metadata.get("budget") != budget
    ):
        raise P2ExecutionError("P1 dataset bytes do not match the frozen P2 component")
    target_response = _target_response(document, target_position)
    root = (
        Path(document["output_root"])
        / "targets"
        / f"target-{target_position:03d}"
        / f"{method}-b{budget}"
    )
    if root.exists():
        raise P2ExecutionError("P2 component is non-retryable and already exists")
    intent = {
        "p2_declaration_identity": document["p2_declaration_identity"],
        **dict(spec),
        "physical_simulator_calls": 0,
        "heldout_calls": 0,
        "heldout_accesses": 0,
    }
    _event(root, 0, "intent", intent)
    _event(root, 1, "attempted", intent)
    started = perf_counter()
    try:
        trained = train_single_round(
            dataset,
            method=method,
            training_seed=int(spec["training_seed"]),
            training_log_dir=root / "training-log",
        )
        posterior = build_mcmc_posterior(trained, target_response)
        recommendation = recommend_sample_map(
            trained,
            posterior,
            recommendation_seed=int(spec["recommendation_seed"]),
            method_configuration_identity=str(spec["method_configuration_identity"]),
        )
        if recommendation.sample_map.decision.status != "recommended":
            raise P2ExecutionError("sample-MAP did not produce a recommendation")
        manifest = _publish_recommendation(
            root,
            document=document,
            spec=spec,
            trained=trained,
            recommendation=recommendation,
        )
        success = {
            **intent,
            "recommendation_identity": manifest["recommendation_identity"],
            "elapsed_seconds": perf_counter() - started,
        }
        _event(root, 2, "success", success)
        return success
    except Exception as exc:
        _event(
            root,
            2,
            "failed",
            {**intent, "failure_type": type(exc).__name__, "failure_explanation": str(exc)},
        )
        raise


def execute_physical_task(
    declaration_path: Path, *, physical_task_index: int, runner: Any = subprocess.run
) -> tuple[int, dict[str, Any]]:
    targets = logical_target_positions(physical_task_index)
    results = []
    all_succeeded = True
    for target_position in targets:
        for budget in BUDGETS:
            for method in METHODS:
                command = [
                    os.environ.get("MUTACO_V3_PYTHON", "python"),
                    "-u",
                    "-m",
                    "experiments.graph_inversion.sbi_zoo_p2",
                    "component",
                    "--declaration",
                    str(declaration_path),
                    "--target-position",
                    str(target_position),
                    "--budget",
                    str(budget),
                    "--method",
                    method,
                    "--execute",
                ]
                completed = runner(command, check=False)
                succeeded = completed.returncode == 0
                all_succeeded = all_succeeded and succeeded
                results.append(
                    {
                        "target_position": target_position,
                        "budget": budget,
                        "method": method,
                        "returncode": completed.returncode,
                    }
                )
    return (0 if all_succeeded else 1), {
        "physical_task_index": physical_task_index,
        "logical_target_positions": list(targets),
        "component_results": results,
        "all_components_succeeded": all_succeeded,
        "physical_simulator_calls": 0,
        "heldout_accesses": 0,
        "automatic_retry": False,
        "slurm_requeue": False,
    }


def _component_root(document: Mapping[str, Any], spec: Mapping[str, Any]) -> Path:
    return (
        Path(document["output_root"])
        / "targets"
        / f"target-{int(spec['target_position']):03d}"
        / f"{spec['method']}-b{int(spec['budget'])}"
    )


def _terminal_status(root: Path) -> tuple[str, dict[str, Any]]:
    events = []
    for path in sorted(root.glob("event-*.json")):
        value = read_json(path)
        identity = value.pop("event_identity", None)
        if identity != stable_hash(P2_EVENT_SCHEMA, value):
            raise P2ExecutionError("P2 method event identity mismatch")
        events.append(value)
    statuses = tuple(value.get("status") for value in events)
    if statuses == ("intent", "attempted", "success"):
        return "success", events[-1]
    if statuses == ("intent", "attempted", "failed"):
        return "failed", events[-1]
    if statuses == ("intent", "attempted"):
        return "attempted_outcome_unknown", events[-1]
    if not statuses:
        return "unattempted", {}
    raise P2ExecutionError("invalid P2 method event sequence")


def _load_recommendation(
    document: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    directory = _component_root(document, spec) / "recommendation"
    expected = {"metadata.json", "samples.f32le", "model-state.pt", "manifest.json"}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected:
        raise P2ExecutionError("P2 recommendation artifact is incomplete")
    metadata_bytes = (directory / "metadata.json").read_bytes()
    samples = (directory / "samples.f32le").read_bytes()
    model = (directory / "model-state.pt").read_bytes()
    manifest_bytes = (directory / "manifest.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    manifest = json.loads(manifest_bytes)
    if canonical_json_bytes(metadata) != metadata_bytes or canonical_json_bytes(manifest) != manifest_bytes:
        raise P2ExecutionError("P2 recommendation JSON is noncanonical")
    manifest_body = {
        "schema_version": P2_RECOMMENDATION_SCHEMA,
        "p2_declaration_identity": document["p2_declaration_identity"],
        "component_identity": spec["component_identity"],
        "target_position": spec["target_position"],
        "target_record_id": spec["target_record_id"],
        "method": spec["method"],
        "budget": spec["budget"],
        "metadata_sha256": _sha(metadata_bytes),
        "samples_sha256": _sha(samples),
        "model_state_sha256": _sha(model),
    }
    decision = metadata.get("sample_map", {}).get("decision", {})
    decision_metadata = decision.get("decision_metadata", {})
    expected_score = (
        "sbi-0.26.1-likelihood_plus_prior-physical-theta-v1"
        if spec["method"] == METHODS[0]
        else "sbi-0.26.1-ratio_plus_prior-physical-theta-v1"
    )
    if (
        {key: manifest.get(key) for key in manifest_body} != manifest_body
        or manifest.get("recommendation_identity")
        != stable_hash(P2_RECOMMENDATION_SCHEMA, manifest_body)
        or metadata.get("component_identity") != spec["component_identity"]
        or metadata.get("dataset_receipt", {}).get("dataset_identity")
        != spec["dataset_identity"]
        or metadata.get("dataset_receipt", {}).get("canonical_content_sha256")
        != spec["dataset_content_sha256"]
        or metadata.get("posterior_candidate_count") != POSTERIOR_SAMPLE_COUNT
        or metadata.get("candidate_order")
        != "20_chains_times_500_chain_major_C_order"
        or len(samples) != POSTERIOR_SAMPLE_COUNT * 4 * 4
        or decision.get("status") != "recommended"
        or decision_metadata.get("candidate_count") != POSTERIOR_SAMPLE_COUNT
        or decision_metadata.get("score_api")
        != "sbi_mcmc_posterior_potential_physical_no_transform_jacobian_v1"
        or decision_metadata.get("score_identity") != expected_score
        or decision_metadata.get("transform_jacobian_policy") != "none"
        or decision_metadata.get("refinement") != "none"
        or metadata.get("physical_simulator_calls") != 0
        or metadata.get("heldout_accesses") != 0
    ):
        raise P2ExecutionError("P2 recommendation identity validation failed")
    candidates = np.frombuffer(samples, dtype="<f4").reshape(POSTERIOR_SAMPLE_COUNT, 4)
    lower = np.asarray(PRIOR_LAW.lower, dtype="float32")
    upper = np.asarray(PRIOR_LAW.upper, dtype="float32")
    if not np.isfinite(candidates).all() or not ((candidates >= lower) & (candidates <= upper)).all():
        raise P2ExecutionError("P2 recommendation candidates are invalid")
    return manifest


def collect_population(document: Mapping[str, Any], *, write: bool = False) -> dict[str, Any]:
    validate_declaration(document)
    counts = {"NLE_B40": 0, "NLE_B100": 0, "NRE_B40": 0, "NRE_B100": 0}
    recommendations = []
    errors = []
    failed = unknown = 0
    for spec in document["components"]:
        label = ("NLE" if spec["method"] == METHODS[0] else "NRE") + f"_B{spec['budget']}"
        try:
            status, terminal = _terminal_status(_component_root(document, spec))
            if status == "failed":
                failed += 1
                errors.append(f"failed:{spec['component_identity']}:{terminal.get('failure_type')}")
                continue
            if status == "attempted_outcome_unknown":
                unknown += 1
                errors.append(f"unknown:{spec['component_identity']}")
                continue
            if status != "success":
                errors.append(f"incomplete:{spec['component_identity']}:{status}")
                continue
            manifest = _load_recommendation(document, spec)
            if terminal.get("recommendation_identity") != manifest["recommendation_identity"]:
                raise P2ExecutionError("success event recommendation identity mismatch")
            counts[label] += 1
            recommendations.append(
                {
                    "target_position": spec["target_position"],
                    "target_record_id": spec["target_record_id"],
                    "budget": spec["budget"],
                    "method": spec["method"],
                    "component_identity": spec["component_identity"],
                    "dataset_identity": spec["dataset_identity"],
                    "recommendation_identity": manifest["recommendation_identity"],
                }
            )
        except Exception as exc:
            errors.append(f"invalid:{spec['component_identity']}:{type(exc).__name__}:{exc}")
    expected_counts = {"NLE_B40": 120, "NLE_B100": 120, "NRE_B40": 120, "NRE_B100": 120}
    passed = not errors and counts == expected_counts and len(recommendations) == 480
    body = {
        "schema_version": P2_COLLECTION_SCHEMA,
        "p2_declaration_identity": document["p2_declaration_identity"],
        "p2_execution_source": document["p2_execution_source"],
        "p1_collection_identity": document["p1"]["collection_identity"],
        "status": "PASS" if passed else "BLOCKED",
        "counts": counts,
        "recommendations": recommendations,
        "recommendation_count": len(recommendations),
        "failed_components": failed,
        "attempted_outcome_unknown": unknown,
        "errors": errors,
        "physical_simulator_calls": 0,
        "heldout_calls": 0,
        "heldout_accesses": 0,
        "p3_gate": "OPEN" if passed else "BLOCKED",
    }
    collection = {**body, "collection_identity": stable_hash(P2_COLLECTION_SCHEMA, body)}
    if write:
        destination = Path(document["output_root"]) / "collection" / "manifest.json"
        if destination.exists() or destination.with_suffix(".sha256").exists():
            raise P2ExecutionError("P2 collection is append-only")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(collection)
        atomic_write_bytes(destination, payload)
        atomic_write_bytes(
            destination.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii")
        )
    return collection


def write_scheduler_record(
    document: Mapping[str, Any], *, array_job_id: str, collector_job_id: str
) -> dict[str, Any]:
    _require_exact_clean_source(document)
    if not re.fullmatch(r"[0-9]+", array_job_id) or not re.fullmatch(r"[0-9]+", collector_job_id):
        raise P2ExecutionError("P2 scheduler job IDs are invalid")
    body = {
        "schema_version": P2_SCHEDULER_SCHEMA,
        "p2_declaration_identity": document["p2_declaration_identity"],
        "p2_execution_source": document["p2_execution_source"],
        "array_job_id": array_job_id,
        "collector_job_id": collector_job_id,
        "resources": SLURM_GEOMETRY,
        "collector_resources": COLLECTOR_SLURM_GEOMETRY,
        "physical_simulator_calls": 0,
    }
    value = {**body, "scheduler_record_identity": stable_hash(P2_SCHEDULER_SCHEMA, body)}
    destination = Path(document["output_root"]) / "scheduler" / "submission.json"
    if destination.exists() or destination.with_suffix(".sha256").exists():
        raise P2ExecutionError("P2 scheduler record is append-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    atomic_write_bytes(destination, payload)
    atomic_write_bytes(destination.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))
    return value


def preview(document: Mapping[str, Any]) -> dict[str, Any]:
    validate_declaration(document)
    return {
        "schema_version": P2_PREFLIGHT_SCHEMA,
        "p2_declaration_identity": document["p2_declaration_identity"],
        "p2_execution_source": document["p2_execution_source"],
        "p1_collection_identity": document["p1"]["collection_identity"],
        "p1_datasets_addressable": document["p1"]["datasets"],
        "unique_p2_components": len(document["components"]),
        "expected_recommendations": EXPECTED_RECOMMENDATIONS,
        "slurm": SLURM_GEOMETRY,
        "collector_slurm": COLLECTOR_SLURM_GEOMETRY,
        "physical_simulator_calls": 0,
        "heldout_accesses": 0,
        "scientific_calls_executed": 0,
        "job_submitted": False,
        "declaration_published": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "declare"):
        command = sub.add_parser(name)
        command.add_argument("--p1-declaration", type=Path, required=True)
        command.add_argument("--p1-collection", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--execution-commit", required=True)
        command.add_argument("--execution-tree", required=True)
        if name == "declare":
            command.add_argument("--lifecycle-receipt", type=Path, required=True)
            command.add_argument("--write", type=Path)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--declaration", type=Path, required=True)
    sub.add_parser("preview", parents=[common])
    component = sub.add_parser("component", parents=[common])
    component.add_argument("--target-position", type=int, required=True)
    component.add_argument("--budget", type=int, required=True)
    component.add_argument("--method", choices=METHODS, required=True)
    component.add_argument("--execute", action="store_true")
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
            p1_declaration_path=args.p1_declaration,
            p1_collection_path=args.p1_collection,
            output_root=args.output_root,
            execution_source=args.execution_commit,
            execution_tree=args.execution_tree,
        )
    elif args.command == "declare":
        document = build_declaration(
            p1_declaration_path=args.p1_declaration,
            p1_collection_path=args.p1_collection,
            output_root=args.output_root,
            execution_source=args.execution_commit,
            execution_tree=args.execution_tree,
            lifecycle_receipt_path=args.lifecycle_receipt,
        )
        if args.write:
            write_declaration(args.write, document)
        result = preview(document)
    elif args.command == "physical-task":
        if not args.execute:
            raise P2ExecutionError("P2 physical task requires explicit execution flag")
        returncode, result = execute_physical_task(
            args.declaration, physical_task_index=args.physical_task_index
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(returncode)
    else:
        document = load_declaration(args.declaration)
        if args.command == "preview":
            result = preview(document)
        elif args.command == "component":
            if not args.execute:
                raise P2ExecutionError("P2 component requires explicit execution flag")
            torch.set_num_threads(20)
            torch.set_num_interop_threads(1)
            result = execute_component(
                document,
                target_position=args.target_position,
                budget=args.budget,
                method=args.method,
            )
        elif args.command == "collect":
            result = collect_population(document, write=args.write)
        elif args.command == "scheduler-record":
            result = write_scheduler_record(
                document,
                array_job_id=args.array_job_id,
                collector_job_id=args.collector_job_id,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "collect" and args.write and result.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
