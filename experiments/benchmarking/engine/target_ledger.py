"""Method-independent immutable target records for graph benchmark campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from mutaco import (
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    SimulationContext,
    checked_evaluate,
    stable_hash,
    strict_metadata_document,
)

from ..artifacts import file_hash, read_json, write_pretty_json
from ..contracts import SimulatorAdapter


TARGET_IDENTITY_POLICY = "target_method_execution_v1"
REQUIRE_ALL_SOURCE_TARGETS = "require_all_v1"
REUSE_AVAILABLE_THEN_CREATE = "reuse_available_then_create_v1"


def uses_independent_target_ledger(config: Mapping[str, Any]) -> bool:
    return config.get("identity_policy") == TARGET_IDENTITY_POLICY


def _record_document(
    *,
    target_family: ScientificIdentity,
    index: int,
    theta: torch.Tensor,
    target_seed: int,
    response: torch.Tensor,
    result_metadata: Mapping[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    context = {
        "scientific_run_id": target_family.value,
        "purpose": "target",
        "logical_index": index,
        "repeat_index": 0,
        "evaluation_seed": target_seed,
        "attempt_index": 0,
    }
    content = {
        "schema_version": "mutaco-target-record-v1",
        "target_family_id": target_family.value,
        "target_index": index,
        "theta_star": theta.detach().cpu().tolist(),
        "target_seed": target_seed,
        "target_context": context,
        "target_response": response.detach().cpu().tolist(),
        "simulator_result_metadata": strict_metadata_document(result_metadata),
        "dtype": str(dtype).removeprefix("torch."),
    }
    content_hash = stable_hash("target-record-content-v1", content)
    return {
        **content,
        "target_record_id": stable_hash(
            "target-record-v1", {"content_hash": content_hash, **content}
        ),
        "content_hash": content_hash,
    }


def _recover_record(
    path: Path,
    *,
    target_family: ScientificIdentity,
    index: int,
    theta: torch.Tensor,
    target_seed: int,
) -> dict[str, Any] | None:
    hash_path = path.with_suffix(".sha256")
    if not path.exists() and not hash_path.exists():
        return None
    if not path.is_file() or not hash_path.is_file():
        raise ValueError(f"incomplete target-ledger record: {path}")
    if file_hash(path) != hash_path.read_text(encoding="ascii").strip():
        raise ValueError(f"target-ledger hash mismatch: {path}")
    record = read_json(path)
    expected = {
        "target_family_id": target_family.value,
        "target_index": index,
        "theta_star": theta.detach().cpu().tolist(),
        "target_seed": target_seed,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"target-ledger identity mismatch: {path}")
    identity_content = {
        key: value
        for key, value in record.items()
        if key not in {"target_record_id", "content_hash"}
    }
    content_hash = stable_hash("target-record-content-v1", identity_content)
    record_id = stable_hash(
        "target-record-v1", {"content_hash": content_hash, **identity_content}
    )
    if record.get("content_hash") != content_hash or record.get(
        "target_record_id"
    ) != record_id:
        raise ValueError(f"target-ledger content identity mismatch: {path}")
    return record


def load_or_create_target_ledger(
    *,
    adapter: SimulatorAdapter,
    config: Mapping[str, Any],
    theta_bank: torch.Tensor,
    target_indices: tuple[int, ...],
    target_family: ScientificIdentity,
    output: Path,
    device: torch.device,
    dtype: torch.dtype,
    restart: bool,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Create or recover immutable targets before any method is run."""

    simulator = adapter.build_simulator(config, device=device, dtype=dtype)
    campaign_seed = int(config["campaign"]["campaign_seed"])
    randomness = config["randomness"]
    namespace = str(randomness["target_seed_namespace"])
    deriver = SeedDeriver(target_family, campaign_seed)
    records: dict[int, dict[str, Any]] = {}
    created = 0
    recovered = 0
    source_reused = 0
    ledger_root = output / "target_ledger"
    source_setting = config.get("runtime_limits", {}).get("target_ledger_source")
    source_root = Path(str(source_setting)).resolve() if source_setting else None
    source_policy = str(
        config.get("runtime_limits", {}).get(
            "target_ledger_source_policy", REQUIRE_ALL_SOURCE_TARGETS
        )
    )
    if source_policy not in {REQUIRE_ALL_SOURCE_TARGETS, REUSE_AVAILABLE_THEN_CREATE}:
        raise ValueError(f"unknown target-ledger source policy: {source_policy!r}")

    def notify(event: Mapping[str, Any]) -> None:
        if progress_callback is not None:
            try:
                progress_callback(event)
            except Exception:
                pass

    for index in target_indices:
        theta = theta_bank[index]
        target_seed = deriver.seed_for(SeedCoordinate(namespace, (index,)))
        path = ledger_root / f"target_{index:03d}.json"
        record = (
            _recover_record(
                path,
                target_family=target_family,
                index=index,
                theta=theta,
                target_seed=target_seed,
            )
            if restart
            else None
        )
        if record is None and source_root is not None:
            source_path = source_root / f"target_{index:03d}.json"
            source_hash_path = source_path.with_suffix(".sha256")
            source_absent = not source_path.exists() and not source_hash_path.exists()
            if not source_absent or source_policy == REQUIRE_ALL_SOURCE_TARGETS:
                record = _recover_record(
                    source_path,
                    target_family=target_family,
                    index=index,
                    theta=theta,
                    target_seed=target_seed,
                )
                if record is None:
                    raise ValueError(
                        f"required source target record is missing: {source_path}"
                    )
                write_pretty_json(path, record)
                path.with_suffix(".sha256").write_text(
                    file_hash(path), encoding="ascii"
                )
                if path.read_bytes() != source_path.read_bytes():
                    raise ValueError(
                        f"reused target record is not byte-identical: {source_path}"
                    )
                source_reused += 1
        if record is None:
            notify(
                {
                    "phase": "target_generation",
                    "target_index": index,
                    "target_calls_started": created + 1,
                    "target_calls_completed": created,
                    "target_records_reused": recovered + source_reused,
                }
            )
            result = checked_evaluate(
                simulator,
                theta,
                device=device,
                dtype=dtype,
                context=SimulationContext(
                    target_family.value,
                    "target",
                    index,
                    0,
                    target_seed,
                ),
            )
            record = _record_document(
                target_family=target_family,
                index=index,
                theta=theta,
                target_seed=target_seed,
                response=result.values,
                result_metadata=result.metadata,
                dtype=dtype,
            )
            write_pretty_json(path, record)
            path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")
            created += 1
            notify(
                {
                    "phase": "target_generation",
                    "target_index": index,
                    "target_calls_started": created,
                    "target_calls_completed": created,
                    "target_records_reused": recovered + source_reused,
                }
            )
        else:
            recovered += 1
            notify(
                {
                    "phase": "target_reuse",
                    "target_index": index,
                    "target_calls_started": created,
                    "target_calls_completed": created,
                    "target_records_reused": recovered + source_reused,
                }
            )
        records[index] = record
    manifest = {
        "schema_version": "mutaco-target-ledger-manifest-v1",
        "identity_policy": TARGET_IDENTITY_POLICY,
        "target_family_id": target_family.value,
        "target_indices": target_indices,
        "created_records": created,
        "recovered_records": recovered,
        "source_reused_records": source_reused,
        "source_ledger": str(source_root) if source_root is not None else None,
        "source_policy": source_policy,
        "records": {
            str(index): {
                "target_record_id": record["target_record_id"],
                "content_hash": record["content_hash"],
            }
            for index, record in records.items()
        },
    }
    write_pretty_json(ledger_root / "manifest.json", manifest)
    return records, manifest
