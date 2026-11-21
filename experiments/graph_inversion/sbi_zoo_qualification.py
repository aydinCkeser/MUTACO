"""Canonical execution and evidence surface for SBI-zoo physical qualification.

Preview and declaration operations are zero-call.  Physical execution remains
guarded by explicit CLI flags and a separately authorized TRUBA profile.
"""

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

from mutaco import (
    ScientificIdentity,
    SeedDeriver,
    SimulationContext,
    canonical_json_bytes,
    checked_evaluate,
    stable_hash,
)
from mutaco.atomic_io import atomic_write_bytes
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.methods.sbi_zoo_accounting import (
    WriteAheadScientificCallLedger,
)
from experiments.benchmarking.methods.sbi_zoo_contract import (
    BUDGETS,
    METHODS,
    POSTERIOR_SAMPLE_COUNT,
    PRIOR_LAW,
    complete_implementation_contract,
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
from experiments.benchmarking.methods.sbi_zoo_dataset_generation import (
    generate_shared_training_dataset,
)
from experiments.benchmarking.methods.sbi_zoo_inference import (
    build_mcmc_posterior,
    recommend_sample_map,
    train_single_round,
)


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_CONTRACT_SOURCE = "d57313a2e9affdb1e6db30a045a8b1a6ef27da33"
SCIENTIFIC_CONTRACT_TREE = "43f539a8948966be16b2dfd15814252254712108"
QUALIFICATION_SCHEMA = "mutaco-sbi-zoo-physical-qualification-declaration-v1"
RECOMMENDATION_SCHEMA = "mutaco-sbi-zoo-qualification-recommendation-v1"
COLLECTION_SCHEMA = "mutaco-sbi-zoo-qualification-collection-v1"
QUALIFICATION_ID = "sbi_zoo_single_round_nle_nre_offcohort_target0_v1"
TARGET_RECORD_ID = (
    "target-record-v1-0344e748d87e1d5e323cb53eb67bc7c0995c229cbaa474ccf44a0c3c228817d8"
)
TARGET_LEDGER_ID = (
    "scientific-v1-69f3d19b0c102296a1236d7d0d3c4d6c188f4f5dc19530107e49b1fdf1b6d549"
)
TARGET_CONTENT_ID = (
    "target-record-content-v1-aa430de28044b58d39bd8e742302537f350ff8adc8970a29e83168c043d29cc0"
)
TARGET_SOURCE_SHA256 = "7c1f7ead8b1b8919105de6d41f0e9b460ce7a56d9552f49651a4e661231b2793"
SIMULATOR_ID = (
    "graph-oscillator.overdamped-scientific.v1.p4-paper-baseline."
    "m20-five-force-scientific.float32"
)
SIMULATOR_CONFIGURATION_ID = (
    "graph-paper-v3-simulator-configuration-v1-"
    "ff70b08584d86189e7464b20602142d9faeca01f588c0d691de6cdf2c7568a08"
)
SIMULATOR_CONFIGURATION_SHA256 = (
    "2614cdae3f2eb40ff1a0f0ee473056f05c8284d3ab59e2f159969ed1cc704a8a"
)
REPAIRED_HELDOUT_ROOTS = (92021540, 92021600)
LIFECYCLE_RECEIPT_SCHEMA = "mutaco-truba-validation-receipt-v1"


class QualificationExecutionError(RuntimeError):
    """Fail-closed qualification declaration or execution error."""


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True
    ).strip()


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _require_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise QualificationExecutionError(f"missing immutable artifact or sidecar: {path}")
    actual = file_hash(path)
    if sidecar.read_text(encoding="ascii").strip() != actual:
        raise QualificationExecutionError(f"immutable artifact hash mismatch: {path}")
    return actual


def _require_external_output(path: Path) -> Path:
    destination = path.resolve(strict=False)
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError:
        return destination
    raise QualificationExecutionError("qualification output must remain outside source checkout")


def resolve_qualification_target(
    target_record_path: Path, paper120_manifest_path: Path
) -> dict[str, Any]:
    """Recover the accepted historical Paper-100 target 0 qualification record."""

    target_sha = _require_sidecar(target_record_path)
    manifest_sha = _require_sidecar(paper120_manifest_path)
    target = read_json(target_record_path)
    manifest = read_json(paper120_manifest_path)
    paper120_ids = {
        str(value["target_record_id"])
        for value in manifest.get("records", {}).values()
        if isinstance(value, Mapping) and "target_record_id" in value
    }
    if (
        target.get("schema_version") != "mutaco-sbi-zoo-qualification-target-v1"
        or target.get("target_index") != 0
        or target.get("target_record_id") != TARGET_RECORD_ID
        or target.get("target_family_id") != TARGET_LEDGER_ID
        or target.get("content_hash") != TARGET_CONTENT_ID
        or target.get("source_sha256") != TARGET_SOURCE_SHA256
        or target.get("selection_rule") != "accepted_historical_paper100_target_position_0"
        or len(target.get("target_response", ())) != 20
        or TARGET_RECORD_ID in paper120_ids
        or manifest.get("completed_count") != 120
    ):
        raise QualificationExecutionError("qualification target provenance is not authoritative")
    return {
        "classification": "qualification_only_outside_paper120_not_scientific_evidence",
        "selection_rule": "accepted_historical_paper100_target_position_0",
        "target_record_id": TARGET_RECORD_ID,
        "target_content_identity": TARGET_CONTENT_ID,
        "target_ledger_identity": TARGET_LEDGER_ID,
        "target_index": 0,
        "target_record_path": str(target_record_path),
        "target_record_sha256": target_sha,
        "historical_source_path": str(target["source_path"]),
        "historical_source_sha256": TARGET_SOURCE_SHA256,
        "archive_branch": str(target["archive_branch"]),
        "archive_cohort_path": str(target["archive_cohort_path"]),
        "paper120_manifest_path": str(paper120_manifest_path),
        "paper120_manifest_sha256": manifest_sha,
        "paper120_membership": False,
    }


def _mechanical_root_seed(target_record_id: str) -> int:
    payload = canonical_json_bytes(
        {
            "domain": "sbi-zoo-offcohort-qualification-root-seed-v1",
            "scientific_contract_source": SCIENTIFIC_CONTRACT_SOURCE,
            "qualification_id": QUALIFICATION_ID,
            "target_record_id": target_record_id,
        }
    )
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _validate_lifecycle_receipt_binding(
    binding: Mapping[str, Any], *, expected_execution_source: str
) -> dict[str, Any]:
    receipt = binding.get("receipt")
    receipt_sha256 = binding.get("receipt_sha256")
    if not isinstance(receipt, Mapping) or not isinstance(receipt_sha256, str):
        raise QualificationExecutionError("lifecycle validation receipt binding is missing")
    receipt_document = dict(receipt)
    receipt_identity = receipt_document.get("receipt_identity")
    receipt_body = {
        key: value for key, value in receipt_document.items() if key != "receipt_identity"
    }
    expected_identity = stable_hash(LIFECYCLE_RECEIPT_SCHEMA, receipt_body)
    materialization = receipt_document.get("source_materialization")
    worker_preview = receipt_document.get("worker_preview")
    if (
        receipt_document.get("schema_version") != LIFECYCLE_RECEIPT_SCHEMA
        or receipt_document.get("status") != "validated_zero_call"
        or receipt_identity != expected_identity
        or binding.get("receipt_identity") != receipt_identity
        or not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256)
        or receipt_document.get("source_commit") != expected_execution_source
        or not isinstance(materialization, Mapping)
        or materialization.get("status") != "materialized_clean"
        or materialization.get("source_commit") != expected_execution_source
        or receipt_document.get("scientific_calls") != 0
        or receipt_document.get("job_submitted") is not False
        or receipt_document.get("submission_authorized") is not False
        or not isinstance(worker_preview, Mapping)
        or worker_preview.get("schema_version")
        != "mutaco-sbi-zoo-qualification-lifecycle-preflight-v1"
        or worker_preview.get("qualification_execution_source")
        != expected_execution_source
        or worker_preview.get("scientific_calls_executed") != 0
        or worker_preview.get("job_submitted") is not False
        or worker_preview.get("declaration_published") is not False
        or worker_preview.get("lifecycle_receipt_binding_required") is not True
    ):
        raise QualificationExecutionError("lifecycle validation receipt is invalid or mismatched")
    return {
        "receipt_identity": receipt_identity,
        "receipt_sha256": receipt_sha256,
        "receipt": receipt_document,
    }


def load_lifecycle_validation_receipt(
    path: Path, *, expected_execution_source: str
) -> dict[str, Any]:
    if not path.is_file():
        raise QualificationExecutionError("lifecycle validation receipt is required")
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationExecutionError("lifecycle validation receipt is not JSON") from exc
    if not isinstance(document, Mapping):
        raise QualificationExecutionError("lifecycle validation receipt is not an object")
    return _validate_lifecycle_receipt_binding(
        {
            "receipt_identity": document.get("receipt_identity"),
            "receipt_sha256": _sha(payload),
            "receipt": document,
        },
        expected_execution_source=expected_execution_source,
    )


def build_zero_call_preflight(
    *,
    target_record_path: Path,
    paper120_manifest_path: Path,
    simulator_configuration_path: Path,
    output_root: Path,
    qualification_execution_source: str,
    qualification_execution_tree: str,
) -> dict[str, Any]:
    """Validate declaration inputs before a lifecycle receipt exists.

    This surface cannot publish a qualification declaration.  Its sole purpose
    is the zero-call lifecycle validation that issues the receipt subsequently
    required by :func:`build_declaration`.
    """

    target = resolve_qualification_target(target_record_path, paper120_manifest_path)
    if file_hash(simulator_configuration_path) != SIMULATOR_CONFIGURATION_SHA256:
        raise QualificationExecutionError("simulator configuration hash is not authoritative")
    _require_external_output(output_root)
    require_pinned_runtime()
    return {
        "schema_version": "mutaco-sbi-zoo-qualification-lifecycle-preflight-v1",
        "status": "zero_call_preflight_receipt_binding_required_before_declaration",
        "qualification_execution_source": qualification_execution_source,
        "qualification_execution_tree": qualification_execution_tree,
        "target_record_id": target["target_record_id"],
        "budgets": [40, 100],
        "declared_slots": 140,
        "required_recommendations": 4,
        "scientific_calls_executed": 0,
        "call_preview": {"scientific_calls_executed": 0, "maximum_calls": 140},
        "heldout_calls": 0,
        "job_submitted": False,
        "declaration_published": False,
        "lifecycle_receipt_binding_required": True,
    }


def build_declaration(
    *,
    target_record_path: Path,
    paper120_manifest_path: Path,
    simulator_configuration_path: Path,
    output_root: Path,
    qualification_execution_source: str,
    qualification_execution_tree: str,
    lifecycle_validation_receipt_path: Path,
) -> dict[str, Any]:
    target = resolve_qualification_target(target_record_path, paper120_manifest_path)
    config_sha = file_hash(simulator_configuration_path)
    if config_sha != SIMULATOR_CONFIGURATION_SHA256:
        raise QualificationExecutionError("simulator configuration hash is not authoritative")
    output = _require_external_output(output_root)
    lifecycle_receipt = load_lifecycle_validation_receipt(
        lifecycle_validation_receipt_path,
        expected_execution_source=qualification_execution_source,
    )
    root_seed = _mechanical_root_seed(TARGET_RECORD_ID)
    core = {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": "declared_unexecuted_separate_physical_authorization_required",
        "qualification_id": QUALIFICATION_ID,
        "classification": "engineering_qualification_not_population_evidence",
        "scientific_contract_source": SCIENTIFIC_CONTRACT_SOURCE,
        "scientific_contract_tree": SCIENTIFIC_CONTRACT_TREE,
        "qualification_execution_source": qualification_execution_source,
        "qualification_execution_tree": qualification_execution_tree,
        "lifecycle_validation_receipt": lifecycle_receipt,
        "target": target,
        "simulator": {
            "identity": SIMULATOR_ID,
            "configuration_identity": SIMULATOR_CONFIGURATION_ID,
            "configuration_path": str(simulator_configuration_path),
            "configuration_sha256": config_sha,
            "dtype": "float32",
            "device": "cpu",
        },
        "datasets": {
            "budgets": [40, 100],
            "b40_b100_relation": "independent_not_nested_no_shared_prefix",
            "declared_slots": {"40": 40, "100": 100},
            "hard_attempt_cap": 140,
            "pass_requires_successful_slots": 140,
            "replacement_slots": 0,
            "automatic_retry": False,
        },
        "methods": list(METHODS),
        "required_recommendations": [
            {"method": method, "budget": budget}
            for budget in BUDGETS
            for method in METHODS
        ],
        "randomness": {
            "root_seed": root_seed,
            "root_seed_derivation": "sha256_domain_source_qualification_target_first_u64_mask63",
            "contract": randomness_contract(),
            "ambient_rng_isolation": True,
        },
        "implementation_contract": complete_implementation_contract(),
        "output_root": str(output),
        "append_only": True,
        "heldout_firewall": {
            "heldout_calls": 0,
            "repaired_family_roots_forbidden": list(REPAIRED_HELDOUT_ROOTS),
            "paper120_training_calls": 0,
        },
        "execution": {
            "physical_execution_authorized_by_declaration": False,
            "requires_cli_execute_and_confirm_approved": True,
            "slurm_no_requeue": True,
            "scientific_attempt_cap": 140,
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
            "slurm": {
                "partition": "barbun",
                "nodes": 1,
                "ntasks": 1,
                "ntasks_per_node": 1,
                "cpus_per_task": 20,
                "exclusive": True,
                "no_requeue": True,
            },
        },
        "expected_outputs": {
            "datasets": ["datasets/b40/artifact", "datasets/b100/artifact"],
            "recommendations": [
                f"methods/{method}-b{budget}/recommendation"
                for budget in BUDGETS
                for method in METHODS
            ],
            "collection": "collection/manifest.json",
            "canonical_hashing": "sha256_over_schema_specific_canonical_payloads",
        },
    }
    identity = stable_hash("sbi-zoo-physical-qualification-declaration-v1", core)
    dataset_declarations = {
        str(budget): stable_hash(
            "sbi-zoo-qualification-dataset-declaration-v1",
            {
                "qualification_declaration_identity": identity,
                "target_record_id": TARGET_RECORD_ID,
                "budget": budget,
                "root_seed": root_seed,
            },
        )
        for budget in BUDGETS
    }
    return {
        **core,
        "qualification_declaration_identity": identity,
        "dataset_generation_declaration_identities": dataset_declarations,
    }


def validate_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in document.items()
        if key not in {"qualification_declaration_identity", "dataset_generation_declaration_identities"}
    }
    identity = stable_hash("sbi-zoo-physical-qualification-declaration-v1", body)
    lifecycle_receipt = _validate_lifecycle_receipt_binding(
        document.get("lifecycle_validation_receipt", {}),
        expected_execution_source=str(document.get("qualification_execution_source", "")),
    )
    if (
        document.get("schema_version") != QUALIFICATION_SCHEMA
        or document.get("qualification_declaration_identity") != identity
        or document.get("scientific_contract_source") != SCIENTIFIC_CONTRACT_SOURCE
        or document.get("scientific_contract_tree") != SCIENTIFIC_CONTRACT_TREE
        or document.get("target", {}).get("target_record_id") != TARGET_RECORD_ID
        or document.get("datasets", {}).get("hard_attempt_cap") != 140
        or document.get("heldout_firewall", {}).get("heldout_calls") != 0
    ):
        raise QualificationExecutionError("qualification declaration is invalid")
    if document.get("lifecycle_validation_receipt") != lifecycle_receipt:
        raise QualificationExecutionError("lifecycle validation receipt binding is noncanonical")
    expected = {
        str(budget): stable_hash(
            "sbi-zoo-qualification-dataset-declaration-v1",
            {
                "qualification_declaration_identity": identity,
                "target_record_id": TARGET_RECORD_ID,
                "budget": budget,
                "root_seed": document["randomness"]["root_seed"],
            },
        )
        for budget in BUDGETS
    }
    if document.get("dataset_generation_declaration_identities") != expected:
        raise QualificationExecutionError("dataset declaration identities are invalid")
    return dict(document)


def write_declaration(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise QualificationExecutionError("qualification declaration is append-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    atomic_write_bytes(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), (_sha(payload) + "\n").encode("ascii"))


def load_declaration(path: Path) -> dict[str, Any]:
    _require_sidecar(path)
    return validate_declaration(read_json(path))


def _require_exact_clean_source(document: Mapping[str, Any]) -> None:
    if (
        _git("rev-parse", "HEAD") != document["qualification_execution_source"]
        or _git("rev-parse", "HEAD^{tree}") != document["qualification_execution_tree"]
        or _git("status", "--porcelain", "--untracked-files=all")
    ):
        raise QualificationExecutionError("qualification requires exact clean execution source")


def _target_response(path: Path, expected_sha: str) -> torch.Tensor:
    if _require_sidecar(path) != expected_sha:
        raise QualificationExecutionError("target record changed after declaration")
    response = torch.tensor(read_json(path)["target_response"], dtype=torch.float32)
    if response.shape != (20,) or not bool(torch.isfinite(response).all()):
        raise QualificationExecutionError("target response is invalid")
    return response.contiguous()


def execute_dataset(
    declaration: Mapping[str, Any], *, budget: int, target_record_path: Path
) -> dict[str, Any]:
    if budget not in BUDGETS:
        raise QualificationExecutionError("qualification budget must be B40 or B100")
    _require_exact_clean_source(declaration)
    target = declaration["target"]
    _target_response(target_record_path, target["target_record_sha256"])
    config_path = ROOT / declaration["simulator"]["configuration_path"]
    if file_hash(config_path) != declaration["simulator"]["configuration_sha256"]:
        raise QualificationExecutionError("simulator configuration changed after declaration")
    from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter

    simulator = GraphScientificD4Adapter().build_target_simulator(
        read_json(config_path), device=torch.device("cpu"), dtype=torch.float32
    )
    if simulator.capabilities.simulator_identity != SIMULATOR_ID:
        raise QualificationExecutionError("simulator identity differs from declaration")
    response_names = tuple(axis.name for axis in simulator.capabilities.response_spec.axes)
    output = Path(declaration["output_root"]) / "datasets" / f"b{budget}"
    ledger = WriteAheadScientificCallLedger(
        output / "ledger",
        declaration["dataset_generation_declaration_identities"][str(budget)],
        budget,
    )
    deriver = dataset_seed_deriver(
        declaration["qualification_declaration_identity"],
        int(declaration["randomness"]["root_seed"]),
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

    started = perf_counter()
    dataset = generate_shared_training_dataset(
        output / "artifact",
        target_record_id=TARGET_RECORD_ID,
        budget=budget,
        deriver=deriver,
        response_names=response_names,
        simulator_identity=SIMULATOR_ID,
        simulator_configuration_identity=SIMULATOR_CONFIGURATION_ID,
        dataset_generation_declaration_identity=(
            declaration["dataset_generation_declaration_identities"][str(budget)]
        ),
        ledger=ledger,
        evaluate=evaluate,
    )
    return {
        "budget": budget,
        "dataset_identity": dataset.identity,
        "canonical_content_sha256": dataset.content_sha256,
        "accounting": ledger.accounting(),
        "elapsed_seconds": perf_counter() - started,
    }


def _method_event(root: Path, ordinal: int, status: str, body: Mapping[str, Any]) -> None:
    path = root / f"event-{ordinal:02d}-{status}.json"
    if path.exists():
        raise QualificationExecutionError("method event is append-only")
    document = {**dict(body), "status": status}
    document["event_identity"] = stable_hash("sbi-zoo-qualification-method-event-v1", document)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, canonical_json_bytes(document) + b"\n")


def _publish_recommendation(
    root: Path, *, trained: Any, recommendation: Any, method_configuration_identity: str
) -> dict[str, Any]:
    destination = root / "recommendation"
    if destination.exists():
        raise QualificationExecutionError("recommendation artifact already exists")
    samples = np.asarray(
        recommendation.samples.detach().cpu().contiguous().numpy(), dtype="<f4", order="C"
    ).tobytes(order="C")
    buffer = io.BytesIO()
    torch.save(trained.estimator.state_dict(), buffer)
    model = buffer.getvalue()
    metadata = {
        "schema_version": RECOMMENDATION_SCHEMA,
        "method": trained.method,
        "budget": trained.budget,
        "dataset_receipt": trained.dataset_receipt,
        "training_seed": trained.training_seed,
        "recommendation_seed": recommendation.recommendation_seed,
        "method_configuration_identity": method_configuration_identity,
        "effective_configuration": trained.effective_configuration,
        "mcmc_configuration": recommendation.mcmc_configuration,
        "posterior_candidate_count": POSTERIOR_SAMPLE_COUNT,
        "posterior_candidate_shape": [POSTERIOR_SAMPLE_COUNT, 4],
        "candidate_order": "20_chains_times_500_chain_major_C_order",
        "sample_map": asdict(recommendation.sample_map),
        "qualification_only": True,
        "heldout_calls": 0,
    }
    metadata_bytes = canonical_json_bytes(metadata)
    manifest_body = {
        "schema_version": RECOMMENDATION_SCHEMA,
        "method": trained.method,
        "budget": trained.budget,
        "metadata_sha256": _sha(metadata_bytes),
        "samples_sha256": _sha(samples),
        "model_state_sha256": _sha(model),
    }
    manifest = {
        **manifest_body,
        "recommendation_identity": stable_hash(
            "sbi-zoo-qualification-recommendation-v1", manifest_body
        ),
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


def execute_method(
    declaration: Mapping[str, Any], *, method: str, budget: int, target_record_path: Path
) -> dict[str, Any]:
    if method not in METHODS or budget not in BUDGETS:
        raise QualificationExecutionError("unknown method-budget qualification component")
    _require_exact_clean_source(declaration)
    dataset = load_shared_training_dataset(
        Path(declaration["output_root"]) / "datasets" / f"b{budget}" / "artifact"
    )
    target_response = _target_response(
        target_record_path, declaration["target"]["target_record_sha256"]
    )
    deriver = dataset_seed_deriver(
        declaration["qualification_declaration_identity"],
        int(declaration["randomness"]["root_seed"]),
    )
    training_seed = estimator_seed(
        deriver, method=method, target_record_id=TARGET_RECORD_ID, budget=budget
    )
    posterior_seed = recommendation_seed(
        deriver, method=method, target_record_id=TARGET_RECORD_ID, budget=budget
    )
    config = estimator_contract(method, budget)
    config_identity = stable_hash("sbi-zoo-method-configuration-v1", config)
    root = Path(declaration["output_root"]) / "methods" / f"{method}-b{budget}"
    if root.exists():
        raise QualificationExecutionError("method component is non-retryable and already exists")
    intent = {
        "qualification_declaration_identity": declaration["qualification_declaration_identity"],
        "method": method,
        "budget": budget,
        "dataset_identity": dataset.identity,
        "dataset_content_sha256": dataset.content_sha256,
        "training_seed": training_seed,
        "recommendation_seed": posterior_seed,
        "method_configuration_identity": config_identity,
    }
    _method_event(root, 0, "intent", intent)
    _method_event(root, 1, "attempted", intent)
    started = perf_counter()
    try:
        trained = train_single_round(
            dataset,
            method=method,
            training_seed=training_seed,
            training_log_dir=root / "training-log",
        )
        posterior = build_mcmc_posterior(trained, target_response)
        recommendation = recommend_sample_map(
            trained,
            posterior,
            recommendation_seed=posterior_seed,
            method_configuration_identity=config_identity,
        )
        if recommendation.sample_map.decision.status != "recommended":
            raise QualificationExecutionError("sample-MAP did not produce a recommendation")
        manifest = _publish_recommendation(
            root,
            trained=trained,
            recommendation=recommendation,
            method_configuration_identity=config_identity,
        )
        success = {
            **intent,
            "recommendation_identity": manifest["recommendation_identity"],
            "elapsed_seconds": perf_counter() - started,
        }
        _method_event(root, 2, "success", success)
        return success
    except Exception as exc:
        _method_event(
            root,
            2,
            "failed",
            {**intent, "failure_type": type(exc).__name__, "failure_explanation": str(exc)},
        )
        raise


def _load_recommendation(root: Path, *, method: str, budget: int) -> dict[str, Any]:
    directory = root / "methods" / f"{method}-b{budget}" / "recommendation"
    expected = {"metadata.json", "samples.f32le", "model-state.pt", "manifest.json"}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected:
        raise QualificationExecutionError("recommendation artifact is incomplete")
    metadata_bytes = (directory / "metadata.json").read_bytes()
    samples = (directory / "samples.f32le").read_bytes()
    model = (directory / "model-state.pt").read_bytes()
    manifest_bytes = (directory / "manifest.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    manifest = json.loads(manifest_bytes)
    if canonical_json_bytes(metadata) != metadata_bytes or canonical_json_bytes(manifest) != manifest_bytes:
        raise QualificationExecutionError("recommendation JSON is noncanonical")
    expected_manifest_body = {
        "schema_version": RECOMMENDATION_SCHEMA,
        "method": method,
        "budget": budget,
        "metadata_sha256": _sha(metadata_bytes),
        "samples_sha256": _sha(samples),
        "model_state_sha256": _sha(model),
    }
    decision = metadata.get("sample_map", {}).get("decision", {})
    decision_metadata = decision.get("decision_metadata", {})
    expected_score_identity = (
        "sbi-0.26.1-likelihood_plus_prior-physical-theta-v1"
        if method == METHODS[0]
        else "sbi-0.26.1-ratio_plus_prior-physical-theta-v1"
    )
    if (
        metadata.get("method") != method
        or metadata.get("budget") != budget
        or metadata.get("posterior_candidate_count") != POSTERIOR_SAMPLE_COUNT
        or metadata.get("candidate_order") != "20_chains_times_500_chain_major_C_order"
        or len(samples) != POSTERIOR_SAMPLE_COUNT * 4 * 4
        or {key: manifest.get(key) for key in expected_manifest_body}
        != expected_manifest_body
        or manifest.get("recommendation_identity")
        != stable_hash("sbi-zoo-qualification-recommendation-v1", expected_manifest_body)
        or decision.get("status") != "recommended"
        or decision_metadata.get("candidate_count") != POSTERIOR_SAMPLE_COUNT
        or decision_metadata.get("score_api")
        != "sbi_mcmc_posterior_potential_physical_no_transform_jacobian_v1"
        or decision_metadata.get("score_identity") != expected_score_identity
        or decision_metadata.get("transform_jacobian_policy") != "none"
        or decision_metadata.get("refinement") != "none"
    ):
        raise QualificationExecutionError("recommendation identity validation failed")
    candidate_array = np.frombuffer(samples, dtype="<f4").reshape(
        POSTERIOR_SAMPLE_COUNT, 4
    )
    if not np.isfinite(candidate_array).all():
        raise QualificationExecutionError("recommendation candidates are nonfinite")
    lower = np.asarray(PRIOR_LAW.lower, dtype="float32")
    upper = np.asarray(PRIOR_LAW.upper, dtype="float32")
    if not ((candidate_array >= lower) & (candidate_array <= upper)).all():
        raise QualificationExecutionError("recommendation candidates violate support")
    return {"metadata": metadata, "manifest": manifest}


def _method_terminal_status(root: Path, *, method: str, budget: int) -> tuple[str, dict[str, Any]]:
    directory = root / "methods" / f"{method}-b{budget}"
    events = []
    for path in sorted(directory.glob("event-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        identity = value.pop("event_identity", None)
        if identity != stable_hash("sbi-zoo-qualification-method-event-v1", value):
            raise QualificationExecutionError("method event identity mismatch")
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
    raise QualificationExecutionError("invalid method event transition sequence")


def collect_qualification(
    declaration: Mapping[str, Any], *, write: bool = False
) -> dict[str, Any]:
    declaration = validate_declaration(declaration)
    root = Path(declaration["output_root"])
    components: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    dataset_pass = True
    for budget in BUDGETS:
        key = f"B{budget}"
        try:
            dataset = load_shared_training_dataset(root / "datasets" / f"b{budget}" / "artifact")
            ledger = WriteAheadScientificCallLedger(
                root / "datasets" / f"b{budget}" / "ledger",
                declaration["dataset_generation_declaration_identities"][str(budget)],
                budget,
            )
            accounting = ledger.accounting()
            if accounting != {
                "maximum_calls": budget,
                "attempted_calls": budget,
                "successful_calls": budget,
                "failed_calls": 0,
                "attempted_outcome_unknown": 0,
                "retry_calls": 0,
            }:
                raise QualificationExecutionError("dataset accounting is incomplete")
            datasets[key] = {
                "status": "PASS",
                "dataset_identity": dataset.identity,
                "canonical_content_sha256": dataset.content_sha256,
                "accounting": accounting,
            }
        except Exception as exc:
            dataset_pass = False
            status = "BLOCKED"
            ledger_root = root / "datasets" / f"b{budget}" / "ledger"
            accounting = None
            if (ledger_root / "ledger.json").is_file():
                try:
                    ledger = WriteAheadScientificCallLedger(
                        ledger_root,
                        declaration["dataset_generation_declaration_identities"][str(budget)],
                        budget,
                    )
                    accounting = ledger.accounting()
                    if accounting["failed_calls"] or accounting["attempted_outcome_unknown"]:
                        status = "FAIL"
                except Exception:
                    accounting = None
            datasets[key] = {
                "status": status,
                "failure_type": type(exc).__name__,
                "failure_explanation": str(exc),
                "accounting": accounting,
            }
    components["SHARED_DATASET_GENERATION"] = (
        "PASS"
        if dataset_pass
        else (
            "FAIL"
            if any(value.get("status") == "FAIL" for value in datasets.values())
            else "BLOCKED"
        )
    )
    for budget in BUDGETS:
        for method in METHODS:
            name = ("NLE" if method == METHODS[0] else "NRE") + f"_B{budget}"
            try:
                terminal_status, terminal = _method_terminal_status(
                    root, method=method, budget=budget
                )
                if terminal_status != "success":
                    if terminal_status == "failed":
                        components[name] = "FAIL"
                        components[f"{name}_failure"] = {
                            "failure_type": terminal.get("failure_type", "method_failure"),
                            "failure_explanation": terminal.get(
                                "failure_explanation", "method component failed"
                            ),
                        }
                        continue
                    raise QualificationExecutionError(
                        f"method terminal status is {terminal_status}"
                    )
                result = _load_recommendation(root, method=method, budget=budget)
                receipt = result["metadata"]["dataset_receipt"]
                expected = datasets[f"B{budget}"]
                if (
                    expected.get("status") != "PASS"
                    or receipt.get("dataset_identity") != expected["dataset_identity"]
                    or receipt.get("canonical_content_sha256")
                    != expected["canonical_content_sha256"]
                ):
                    raise QualificationExecutionError("method did not consume shared dataset")
                if terminal.get("recommendation_identity") != result["manifest"][
                    "recommendation_identity"
                ]:
                    raise QualificationExecutionError(
                        "method success event does not bind recommendation"
                    )
                components[name] = "PASS"
            except Exception as exc:
                components[name] = "BLOCKED"
                components[f"{name}_failure"] = {
                    "failure_type": type(exc).__name__,
                    "failure_explanation": str(exc),
                }
    required = ["SHARED_DATASET_GENERATION", "NLE_B40", "NLE_B100", "NRE_B40", "NRE_B100"]
    overall = "PASS" if all(components.get(item) == "PASS" for item in required) else "BLOCKED"
    body = {
        "schema_version": COLLECTION_SCHEMA,
        "qualification_declaration_identity": declaration["qualification_declaration_identity"],
        "scientific_contract_source": SCIENTIFIC_CONTRACT_SOURCE,
        "qualification_execution_source": declaration["qualification_execution_source"],
        "lifecycle_validation_receipt_identity": declaration[
            "lifecycle_validation_receipt"
        ]["receipt_identity"],
        "lifecycle_validation_receipt_sha256": declaration[
            "lifecycle_validation_receipt"
        ]["receipt_sha256"],
        "datasets": datasets,
        "components": components,
        "overall": overall,
        "scientific_calls": sum(
            (value.get("accounting") or {}).get("attempted_calls", 0)
            for value in datasets.values()
        ),
        "heldout_calls": 0,
        "paper120_training_calls": 0,
        "truba_submission_from_collector": False,
    }
    document = {**body, "collection_identity": stable_hash("sbi-zoo-qualification-collection-v1", body)}
    if write:
        destination = root / "collection" / "manifest.json"
        if destination.exists():
            raise QualificationExecutionError("qualification collection is append-only")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(document)
        atomic_write_bytes(destination, payload)
        atomic_write_bytes(destination.with_suffix(".sha256"), (_sha(payload) + "\n").encode())
    return document


def preview(document: Mapping[str, Any]) -> dict[str, Any]:
    validate_declaration(document)
    return {
        "schema_version": "mutaco-sbi-zoo-qualification-preview-v1",
        "qualification_declaration_identity": document["qualification_declaration_identity"],
        "target_record_id": TARGET_RECORD_ID,
        "budgets": [40, 100],
        "declared_slots": 140,
        "required_recommendations": 4,
        "scientific_calls_executed": 0,
        "call_preview": {"scientific_calls_executed": 0, "maximum_calls": 140},
        "heldout_calls": 0,
        "job_submitted": False,
    }


def write_scheduler_record(
    declaration: Mapping[str, Any], *, job_id: str, node_list: str
) -> dict[str, Any]:
    declaration = validate_declaration(declaration)
    _require_exact_clean_source(declaration)
    if not job_id.strip() or not node_list.strip():
        raise QualificationExecutionError("scheduler job and node identities must be nonempty")
    body = {
        "schema_version": "mutaco-sbi-zoo-qualification-scheduler-record-v1",
        "qualification_declaration_identity": declaration["qualification_declaration_identity"],
        "qualification_execution_source": declaration["qualification_execution_source"],
        "lifecycle_validation_receipt_identity": declaration[
            "lifecycle_validation_receipt"
        ]["receipt_identity"],
        "lifecycle_validation_receipt_sha256": declaration[
            "lifecycle_validation_receipt"
        ]["receipt_sha256"],
        "slurm_job_id": job_id,
        "slurm_node_list": node_list,
        "resources": declaration["execution"]["slurm"],
        "runtime": declaration["execution"]["runtime"],
        "scientific_calls_at_record": 0,
    }
    document = {
        **body,
        "scheduler_record_identity": stable_hash(
            "sbi-zoo-qualification-scheduler-record-v1", body
        ),
    }
    path = Path(declaration["output_root"]) / "scheduler" / "record.json"
    if path.exists():
        raise QualificationExecutionError("scheduler record is append-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(document)
    atomic_write_bytes(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), (_sha(payload) + "\n").encode())
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--target-record", type=Path, required=True)
    preflight.add_argument("--paper120-manifest", type=Path, required=True)
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--execution-commit", required=True)
    preflight.add_argument("--execution-tree", required=True)
    declare = sub.add_parser("declare")
    declare.add_argument("--target-record", type=Path, required=True)
    declare.add_argument("--paper120-manifest", type=Path, required=True)
    declare.add_argument("--config", type=Path, required=True)
    declare.add_argument("--output-root", type=Path, required=True)
    declare.add_argument("--execution-commit", required=True)
    declare.add_argument("--execution-tree", required=True)
    declare.add_argument("--lifecycle-receipt", type=Path, required=True)
    declare.add_argument("--write", type=Path)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--declaration", type=Path, required=True)
    preview_parser = sub.add_parser("preview", parents=[common])
    del preview_parser
    dataset = sub.add_parser("dataset", parents=[common])
    dataset.add_argument("--budget", type=int, required=True)
    dataset.add_argument("--target-record", type=Path, required=True)
    dataset.add_argument("--execute", action="store_true")
    dataset.add_argument("--confirm-approved", action="store_true")
    method = sub.add_parser("method", parents=[common])
    method.add_argument("--method", choices=METHODS, required=True)
    method.add_argument("--budget", type=int, required=True)
    method.add_argument("--target-record", type=Path, required=True)
    method.add_argument("--execute", action="store_true")
    method.add_argument("--confirm-approved", action="store_true")
    collect = sub.add_parser("collect", parents=[common])
    collect.add_argument("--write", action="store_true")
    scheduler = sub.add_parser("scheduler-record", parents=[common])
    scheduler.add_argument("--job-id", required=True)
    scheduler.add_argument("--node-list", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = build_zero_call_preflight(
            target_record_path=args.target_record,
            paper120_manifest_path=args.paper120_manifest,
            simulator_configuration_path=args.config,
            output_root=args.output_root,
            qualification_execution_source=args.execution_commit,
            qualification_execution_tree=args.execution_tree,
        )
    elif args.command == "declare":
        document = build_declaration(
            target_record_path=args.target_record,
            paper120_manifest_path=args.paper120_manifest,
            simulator_configuration_path=args.config,
            output_root=args.output_root,
            qualification_execution_source=args.execution_commit,
            qualification_execution_tree=args.execution_tree,
            lifecycle_validation_receipt_path=args.lifecycle_receipt,
        )
        if args.write:
            write_declaration(args.write, document)
        result = preview(document)
        result["declaration_sha256"] = _sha(canonical_json_bytes(document))
        result["qualification_execution_source"] = args.execution_commit
    else:
        document = load_declaration(args.declaration)
        if args.command == "preview":
            result = preview(document)
        elif args.command == "dataset":
            if not args.execute:
                result = {**preview(document), "selected_budget": args.budget}
            else:
                if not args.confirm_approved:
                    raise QualificationExecutionError("dataset execution requires approval flag")
                torch.set_num_threads(20)
                torch.set_num_interop_threads(1)
                result = execute_dataset(document, budget=args.budget, target_record_path=args.target_record)
        elif args.command == "method":
            if not args.execute:
                result = {
                    **preview(document),
                    "selected_method": args.method,
                    "selected_budget": args.budget,
                }
            else:
                if not args.confirm_approved:
                    raise QualificationExecutionError("method execution requires approval flag")
                torch.set_num_threads(20)
                torch.set_num_interop_threads(1)
                result = execute_method(
                    document,
                    method=args.method,
                    budget=args.budget,
                    target_record_path=args.target_record,
                )
        elif args.command == "collect":
            result = collect_qualification(document, write=args.write)
        else:
            result = write_scheduler_record(
                document, job_id=args.job_id, node_list=args.node_list
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
