"""Canonical byte-identical shared datasets for single-round NLE and NRE."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch

from mutaco import canonical_json_bytes, stable_hash
from mutaco.atomic_io import atomic_write_bytes

from .sbi_zoo_contract import (
    BUDGETS,
    METHODS,
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    PRIOR_LAW,
    RESPONSE_DIMENSION,
    SLOT_OUTCOME_SCHEMA,
)
from .sbi_zoo_failures import DatasetIntegrityError, DatasetSerializationError


DATASET_SCHEMA = "mutaco-sbi-zoo-shared-dataset-v1"
DATASET_HASH_ALGORITHM = "sha256"
THETA_FILE = "theta.f32le"
RESPONSE_FILE = "response.f32le"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"
SLOT_RESPONSE_FILE = "response.f32le"
SLOT_METADATA_FILE = "metadata.json"
SLOT_MANIFEST_FILE = "manifest.json"
SLOT_IDENTITY_FIELDS = (
    "dataset_generation_declaration_identity",
    "call_ledger_identity",
    "target_record_id",
    "budget",
    "scientific_slot_identity",
    "row_index",
    "theta_canonical_sha256",
    "stochastic_context_identity",
    "response_ordering_identity",
    "simulator_identity",
    "simulator_configuration_identity",
)


def _sha256(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _canonical_array_bytes(tensor: torch.Tensor, *, columns: int, field: str) -> bytes:
    if tensor.device.type != "cpu" or tensor.dtype is not torch.float32:
        raise DatasetIntegrityError(f"{field} must be CPU float32")
    if tensor.ndim != 2 or tensor.shape[1] != columns:
        raise DatasetIntegrityError(f"{field} has the wrong two-dimensional shape")
    if not bool(torch.isfinite(tensor).all()):
        raise DatasetIntegrityError(f"{field} forbids NaN and infinity")
    array = np.asarray(tensor.detach().contiguous().numpy(), dtype="<f4", order="C")
    return array.tobytes(order="C")


def canonical_vector_bytes(tensor: torch.Tensor, *, length: int, field: str) -> bytes:
    """Canonical little-endian C-order float32 bytes for one scientific vector."""

    if tensor.device.type != "cpu" or tensor.dtype is not torch.float32:
        raise DatasetIntegrityError(f"{field} must be CPU float32")
    if tensor.shape != (length,) or not bool(torch.isfinite(tensor).all()):
        raise DatasetIntegrityError(f"{field} must be one finite {length}-vector")
    array = np.asarray(tensor.detach().contiguous().numpy(), dtype="<f4", order="C")
    return array.tobytes(order="C")


def canonical_vector_sha256(tensor: torch.Tensor, *, length: int, field: str) -> str:
    return _sha256(canonical_vector_bytes(tensor, length=length, field=field))


def response_ordering_identity(response_names: Sequence[str]) -> str:
    names = [str(item) for item in response_names]
    if len(names) != RESPONSE_DIMENSION or len(set(names)) != RESPONSE_DIMENSION:
        raise DatasetIntegrityError("response ordering requires 20 distinct names")
    return stable_hash("sbi-zoo-response-ordering-v1", names)


@dataclass(frozen=True)
class SlotOutcomeArtifact:
    root: Path
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    response: torch.Tensor

    @property
    def identity(self) -> str:
        return str(self.manifest["outcome_identity"])

    @property
    def canonical_payload_sha256(self) -> str:
        return str(self.manifest["canonical_payload_sha256"])

    def receipt(self) -> dict[str, Any]:
        return {
            "slot_outcome_schema": SLOT_OUTCOME_SCHEMA,
            "outcome_identity": self.identity,
            "canonical_payload_sha256": self.canonical_payload_sha256,
            "response_content_sha256": self.metadata["response_content_sha256"],
            "scientific_slot_identity": self.metadata["scientific_slot_identity"],
            "row_index": self.metadata["row_index"],
        }


def _slot_metadata(identity: Mapping[str, Any], response_bytes: bytes) -> dict[str, Any]:
    if set(identity) != set(SLOT_IDENTITY_FIELDS):
        raise DatasetIntegrityError("slot outcome call-identity fields are malformed")
    document = dict(identity)
    for field in SLOT_IDENTITY_FIELDS:
        if field in {"budget", "row_index"}:
            continue
        if not isinstance(document[field], str) or not document[field].strip():
            raise DatasetIntegrityError(f"slot outcome {field} must be a nonempty identity")
    budget = document["budget"]
    row_index = document["row_index"]
    if budget not in BUDGETS or isinstance(row_index, bool) or not isinstance(row_index, int):
        raise DatasetIntegrityError("slot outcome budget/row identity is invalid")
    if not 0 <= row_index < budget:
        raise DatasetIntegrityError("slot outcome row lies outside its budget")
    document.update(
        {
            "schema_version": SLOT_OUTCOME_SCHEMA,
            "response_shape": [RESPONSE_DIMENSION],
            "dtype": "float32",
            "byte_order": "little_endian",
            "array_order": "C_contiguous",
            "response_content_sha256": _sha256(response_bytes),
            "canonical_payload": "u64be_length_plus_metadata_then_response",
        }
    )
    return document


def write_slot_outcome_artifact(
    root: Path,
    *,
    response: torch.Tensor,
    call_identity: Mapping[str, Any],
) -> SlotOutcomeArtifact:
    """Publish one immutable response bound to one attempted scientific call."""

    destination = Path(root)
    if destination.exists():
        raise DatasetSerializationError(f"slot outcome already exists: {destination}")
    response_bytes = canonical_vector_bytes(
        response, length=RESPONSE_DIMENSION, field="slot response"
    )
    metadata = _slot_metadata(call_identity, response_bytes)
    metadata_bytes = canonical_json_bytes(metadata)
    payload_sha = _sha256(_slot_payload(metadata_bytes, response_bytes))
    identity_body = {
        "schema_version": SLOT_OUTCOME_SCHEMA,
        "canonical_payload_sha256": payload_sha,
        "scientific_slot_identity": metadata["scientific_slot_identity"],
        "row_index": metadata["row_index"],
    }
    manifest = {
        **identity_body,
        "outcome_identity": stable_hash("sbi-zoo-slot-outcome-v1", identity_body),
        "hash_algorithm": "sha256",
        "files": {
            SLOT_METADATA_FILE: {
                "sha256": _sha256(metadata_bytes),
                "bytes": len(metadata_bytes),
            },
            SLOT_RESPONSE_FILE: {
                "sha256": _sha256(response_bytes),
                "bytes": len(response_bytes),
            },
        },
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        atomic_write_bytes(temporary / SLOT_METADATA_FILE, metadata_bytes)
        atomic_write_bytes(temporary / SLOT_RESPONSE_FILE, response_bytes)
        atomic_write_bytes(temporary / SLOT_MANIFEST_FILE, canonical_json_bytes(manifest))
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        if isinstance(exc, (DatasetIntegrityError, DatasetSerializationError)):
            raise
        raise DatasetSerializationError(str(exc)) from exc
    return load_slot_outcome_artifact(destination, expected_call_identity=call_identity)


def load_slot_outcome_artifact(
    root: Path, *, expected_call_identity: Mapping[str, Any] | None = None
) -> SlotOutcomeArtifact:
    directory = Path(root)
    expected_names = {SLOT_METADATA_FILE, SLOT_RESPONSE_FILE, SLOT_MANIFEST_FILE}
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != expected_names:
        raise DatasetIntegrityError("slot outcome artifact is incomplete or contains extras")
    try:
        manifest_bytes = (directory / SLOT_MANIFEST_FILE).read_bytes()
        metadata_bytes = (directory / SLOT_METADATA_FILE).read_bytes()
        response_bytes = (directory / SLOT_RESPONSE_FILE).read_bytes()
        manifest = json.loads(manifest_bytes)
        metadata = json.loads(metadata_bytes)
    except Exception as exc:
        raise DatasetIntegrityError("slot outcome artifact cannot be decoded") from exc
    if canonical_json_bytes(manifest) != manifest_bytes or canonical_json_bytes(metadata) != metadata_bytes:
        raise DatasetIntegrityError("slot outcome JSON is not canonical")
    if set(manifest) != {
        "schema_version",
        "canonical_payload_sha256",
        "scientific_slot_identity",
        "row_index",
        "outcome_identity",
        "hash_algorithm",
        "files",
    } or set(manifest.get("files", {})) != {SLOT_METADATA_FILE, SLOT_RESPONSE_FILE}:
        raise DatasetIntegrityError("slot outcome manifest is malformed")
    if metadata.get("schema_version") != SLOT_OUTCOME_SCHEMA:
        raise DatasetIntegrityError("slot outcome schema mismatch")
    identity = {field: metadata.get(field) for field in SLOT_IDENTITY_FIELDS}
    rebuilt = _slot_metadata(identity, response_bytes)
    if rebuilt != metadata:
        raise DatasetIntegrityError("slot outcome metadata or response hash mismatch")
    if expected_call_identity is not None and dict(expected_call_identity) != identity:
        raise DatasetIntegrityError("slot outcome does not match its scientific call identity")
    for name, payload in (
        (SLOT_METADATA_FILE, metadata_bytes),
        (SLOT_RESPONSE_FILE, response_bytes),
    ):
        record = manifest.get("files", {}).get(name, {})
        if record.get("sha256") != _sha256(payload) or record.get("bytes") != len(payload):
            raise DatasetIntegrityError(f"slot outcome file identity mismatch: {name}")
    payload_sha = _sha256(_slot_payload(metadata_bytes, response_bytes))
    identity_body = {
        "schema_version": SLOT_OUTCOME_SCHEMA,
        "canonical_payload_sha256": payload_sha,
        "scientific_slot_identity": metadata["scientific_slot_identity"],
        "row_index": metadata["row_index"],
    }
    if manifest.get("canonical_payload_sha256") != payload_sha or manifest.get(
        "outcome_identity"
    ) != stable_hash("sbi-zoo-slot-outcome-v1", identity_body):
        raise DatasetIntegrityError("slot outcome canonical identity mismatch")
    if len(response_bytes) != RESPONSE_DIMENSION * 4:
        raise DatasetIntegrityError("slot outcome response byte length mismatch")
    response_np = np.frombuffer(response_bytes, dtype="<f4").copy()
    response = torch.from_numpy(response_np).to(dtype=torch.float32).contiguous()
    if response.shape != (RESPONSE_DIMENSION,) or not bool(torch.isfinite(response).all()):
        raise DatasetIntegrityError("slot outcome response is invalid")
    return SlotOutcomeArtifact(directory, manifest, metadata, response)


def _content_payload(metadata_bytes: bytes, theta_bytes: bytes, response_bytes: bytes) -> bytes:
    """Unambiguous exact payload covered by the canonical content hash."""

    fields = (metadata_bytes, theta_bytes, response_bytes)
    return b"".join(len(field).to_bytes(8, "big") + field for field in fields)


def _slot_payload(metadata_bytes: bytes, response_bytes: bytes) -> bytes:
    return b"".join(
        len(field).to_bytes(8, "big") + field
        for field in (metadata_bytes, response_bytes)
    )


@dataclass(frozen=True)
class SharedTrainingDataset:
    """Validated immutable training evidence loaded from one artifact."""

    root: Path
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    theta: torch.Tensor
    response: torch.Tensor

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["canonical_content_sha256"])

    @property
    def identity(self) -> str:
        return str(self.manifest["dataset_identity"])

    def consumer_receipt(self, method: str) -> dict[str, Any]:
        if method not in METHODS:
            raise ValueError("unknown SBI-zoo dataset consumer")
        return {
            "schema_version": "mutaco-sbi-zoo-dataset-consumer-receipt-v1",
            "method": method,
            "dataset_identity": self.identity,
            "canonical_content_sha256": self.content_sha256,
            "theta_file_sha256": self.manifest["files"][THETA_FILE]["sha256"],
            "response_file_sha256": self.manifest["files"][RESPONSE_FILE]["sha256"],
            "row_order": "stored_C_order_row_major_0_to_budget_minus_1",
        }


def _validate_metadata(metadata: Mapping[str, Any], *, rows: int) -> dict[str, Any]:
    required = {
        "target_record_id",
        "budget",
        "parameter_names",
        "response_names",
        "parameter_bounds",
        "simulator_identity",
        "simulator_configuration_identity",
        "row_context_identities",
        "dataset_generation_declaration_identity",
        "scientific_call_ledger_identity",
        "theta_law",
        "theta_seed",
    }
    missing = required - set(metadata)
    if missing:
        raise DatasetIntegrityError(f"shared dataset metadata missing {sorted(missing)}")
    budget = metadata["budget"]
    if budget not in BUDGETS or budget != rows:
        raise DatasetIntegrityError("dataset row count must equal its B40/B100 budget")
    if tuple(metadata["parameter_names"]) != PARAMETER_NAMES:
        raise DatasetIntegrityError("parameter coordinate ordering mismatch")
    if len(metadata["response_names"]) != RESPONSE_DIMENSION or len(
        set(map(str, metadata["response_names"]))
    ) != RESPONSE_DIMENSION:
        raise DatasetIntegrityError("response coordinate ordering must contain 20 names")
    contexts = metadata["row_context_identities"]
    if not isinstance(contexts, Sequence) or isinstance(contexts, (str, bytes)):
        raise DatasetIntegrityError("row context identities must be an ordered sequence")
    if len(contexts) != rows or len(set(map(str, contexts))) != rows:
        raise DatasetIntegrityError("every dataset row requires one distinct context identity")
    bounds = metadata["parameter_bounds"]
    if not isinstance(bounds, Mapping):
        raise DatasetIntegrityError("parameter bounds metadata must be a mapping")
    lower, upper = bounds.get("lower"), bounds.get("upper")
    if not isinstance(lower, Sequence) or not isinstance(upper, Sequence):
        raise DatasetIntegrityError("parameter bounds require lower and upper sequences")
    if len(lower) != len(PARAMETER_NAMES) or len(upper) != len(PARAMETER_NAMES):
        raise DatasetIntegrityError("parameter bounds dimension mismatch")
    if tuple(float(item) for item in lower) != PARAMETER_LOWER or tuple(
        float(item) for item in upper
    ) != PARAMETER_UPPER:
        raise DatasetIntegrityError("parameter bounds do not match the authoritative prior")
    if metadata["theta_law"] != PRIOR_LAW.document():
        raise DatasetIntegrityError("theta-law metadata does not match the authoritative prior")
    for field in (
        "target_record_id",
        "simulator_identity",
        "simulator_configuration_identity",
        "dataset_generation_declaration_identity",
        "scientific_call_ledger_identity",
    ):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise DatasetIntegrityError(f"{field} must be a nonempty identity")
    document = dict(metadata)
    document.update(
        {
            "schema_version": DATASET_SCHEMA,
            "theta_shape": [rows, len(PARAMETER_NAMES)],
            "response_shape": [rows, RESPONSE_DIMENSION],
            "dtype": "float32",
            "byte_order": "little_endian",
            "array_order": "C_contiguous_row_major",
            "row_order": "logical_row_index_0_to_budget_minus_1",
            "nonfinite_policy": "forbidden",
            "canonical_metadata": "mutaco_canonical_json_utf8_no_trailing_newline",
            "hash_algorithm": DATASET_HASH_ALGORITHM,
            "cross_budget_relation": "independent_not_nested",
        }
    )
    return document


def write_shared_training_dataset(
    root: Path,
    *,
    theta: torch.Tensor,
    response: torch.Tensor,
    metadata: Mapping[str, Any],
) -> SharedTrainingDataset:
    """Atomically publish a complete append-only canonical dataset directory."""

    destination = Path(root)
    if destination.exists():
        raise DatasetSerializationError(f"shared dataset artifact already exists: {destination}")
    if theta.shape[0] != response.shape[0]:
        raise DatasetIntegrityError("theta and response row counts differ")
    rows = int(theta.shape[0])
    canonical_metadata = _validate_metadata(metadata, rows=rows)
    theta_bytes = _canonical_array_bytes(theta, columns=len(PARAMETER_NAMES), field="theta")
    response_bytes = _canonical_array_bytes(
        response, columns=RESPONSE_DIMENSION, field="response"
    )
    metadata_bytes = canonical_json_bytes(canonical_metadata)
    content_sha = _sha256(_content_payload(metadata_bytes, theta_bytes, response_bytes))
    identity_body = {
        "schema_version": DATASET_SCHEMA,
        "canonical_content_sha256": content_sha,
        "target_record_id": canonical_metadata["target_record_id"],
        "budget": canonical_metadata["budget"],
    }
    manifest = {
        **identity_body,
        "dataset_identity": stable_hash("sbi-zoo-shared-dataset-v1", identity_body),
        "hash_algorithm": DATASET_HASH_ALGORITHM,
        "canonical_hash_payload": "u64be_length_plus_metadata_then_theta_then_response",
        "atomic_persistence": "complete_sibling_directory_renamed_into_place_manifest_written_last",
        "files": {
            METADATA_FILE: {"sha256": _sha256(metadata_bytes), "bytes": len(metadata_bytes)},
            THETA_FILE: {"sha256": _sha256(theta_bytes), "bytes": len(theta_bytes)},
            RESPONSE_FILE: {"sha256": _sha256(response_bytes), "bytes": len(response_bytes)},
        },
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        atomic_write_bytes(temporary / METADATA_FILE, metadata_bytes)
        atomic_write_bytes(temporary / THETA_FILE, theta_bytes)
        atomic_write_bytes(temporary / RESPONSE_FILE, response_bytes)
        atomic_write_bytes(temporary / MANIFEST_FILE, canonical_json_bytes(manifest))
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        if isinstance(exc, DatasetIntegrityError):
            raise
        raise DatasetSerializationError(str(exc)) from exc
    return load_shared_training_dataset(destination)


def load_shared_training_dataset(root: Path) -> SharedTrainingDataset:
    """Reload and rehash every byte before exposing tensors to an estimator."""

    directory = Path(root)
    try:
        manifest_bytes = (directory / MANIFEST_FILE).read_bytes()
        manifest = json.loads(manifest_bytes)
        metadata_bytes = (directory / METADATA_FILE).read_bytes()
        theta_bytes = (directory / THETA_FILE).read_bytes()
        response_bytes = (directory / RESPONSE_FILE).read_bytes()
    except Exception as exc:
        raise DatasetIntegrityError(f"shared dataset artifact is incomplete: {directory}") from exc
    if not isinstance(manifest, dict):
        raise DatasetIntegrityError("shared dataset manifest must be an object")
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise DatasetIntegrityError("shared dataset manifest is not canonical JSON")
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise DatasetIntegrityError("shared dataset schema identity mismatch")
    for name, payload in (
        (METADATA_FILE, metadata_bytes),
        (THETA_FILE, theta_bytes),
        (RESPONSE_FILE, response_bytes),
    ):
        expected = manifest.get("files", {}).get(name, {})
        if expected.get("sha256") != _sha256(payload) or expected.get("bytes") != len(payload):
            raise DatasetIntegrityError(f"shared dataset file identity mismatch: {name}")
    try:
        metadata = json.loads(metadata_bytes)
    except Exception as exc:
        raise DatasetIntegrityError("shared dataset metadata is not JSON") from exc
    if canonical_json_bytes(metadata) != metadata_bytes:
        raise DatasetIntegrityError("shared dataset metadata is not canonical JSON")
    rows = int(metadata.get("budget", -1))
    canonical_metadata = _validate_metadata(metadata, rows=rows)
    if canonical_metadata != metadata:
        raise DatasetIntegrityError("shared dataset metadata contract mismatch")
    expected_theta_bytes = rows * len(PARAMETER_NAMES) * 4
    expected_response_bytes = rows * RESPONSE_DIMENSION * 4
    if len(theta_bytes) != expected_theta_bytes or len(response_bytes) != expected_response_bytes:
        raise DatasetIntegrityError("shared dataset raw byte length disagrees with shape")
    content_sha = _sha256(_content_payload(metadata_bytes, theta_bytes, response_bytes))
    if content_sha != manifest.get("canonical_content_sha256"):
        raise DatasetIntegrityError("shared dataset canonical content hash mismatch")
    identity_body = {
        "schema_version": DATASET_SCHEMA,
        "canonical_content_sha256": content_sha,
        "target_record_id": metadata["target_record_id"],
        "budget": metadata["budget"],
    }
    if manifest.get("dataset_identity") != stable_hash(
        "sbi-zoo-shared-dataset-v1", identity_body
    ):
        raise DatasetIntegrityError("shared dataset scientific identity mismatch")
    theta_np = np.frombuffer(theta_bytes, dtype="<f4").reshape(rows, len(PARAMETER_NAMES)).copy()
    response_np = np.frombuffer(response_bytes, dtype="<f4").reshape(rows, RESPONSE_DIMENSION).copy()
    theta = torch.from_numpy(theta_np).to(dtype=torch.float32).contiguous()
    response = torch.from_numpy(response_np).to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(theta).all() and torch.isfinite(response).all()):
        raise DatasetIntegrityError("reloaded shared dataset contains nonfinite values")
    lower = torch.tensor(metadata["parameter_bounds"]["lower"], dtype=torch.float32)
    upper = torch.tensor(metadata["parameter_bounds"]["upper"], dtype=torch.float32)
    if not bool(((theta >= lower) & (theta <= upper)).all()):
        raise DatasetIntegrityError("reloaded theta violates physical support")
    return SharedTrainingDataset(directory, manifest, metadata, theta, response)
