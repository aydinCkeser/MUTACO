"""Generic reference-only cohort orchestration."""

from __future__ import annotations

from pathlib import Path
import json
from time import perf_counter
from typing import Any, Mapping

import torch

from mutaco import canonical_json_bytes, stable_hash

from ..artifacts import (
    recover_target,
    write_target,
)

from ..contracts import SimulatorAdapter, TargetSelection
from ..methods import run_mutaco_target


def replay_indices(count: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, (count - 1) // 2, count - 1)))


def run_reference_cohort(
    *,
    adapter: SimulatorAdapter,
    config: Mapping[str, Any],
    theta_bank: torch.Tensor,
    target_indices: tuple[int, ...],
    campaign_science: Any,
    output: Path,
    device: torch.device,
    dtype: torch.dtype,
    restart: bool,
    verbose_events: bool,
    target_records: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    recovered_count = 0
    executed_count = 0
    total = len(target_indices)
    for position, index in enumerate(target_indices):
        directory = output / "targets" / f"target_{index:03d}"
        recovered = recover_target(directory, campaign_science.value) if restart else None
        if recovered is not None:
            row = recovered
            recovered_count += 1
        else:
            executed_count += 1
            callback = (
                lambda event, i=index: print(f"[target {i:03d}] {event.message}")
                if verbose_events
                else None
            )
            row = run_mutaco_target(
                adapter=adapter,
                index=index,
                theta_star=theta_bank[index],
                config=config,
                campaign_science=campaign_science,
                device=device,
                dtype=dtype,
                verbose_callback=callback,
                supplied_target_response=(
                    torch.tensor(
                        target_records[index]["target_response"],
                        device=device,
                        dtype=dtype,
                    ).contiguous()
                    if target_records is not None
                    else None
                ),
                supplied_target_seed=(
                    int(target_records[index]["target_seed"])
                    if target_records is not None
                    else None
                ),
                supplied_target_identity=(
                    str(target_records[index]["target_record_id"])
                    if target_records is not None
                    else None
                ),
            )
            write_target(directory, row)
            if config.get("artifact_policy") == "compact_paper_records_v1":
                cap = int(config["runtime_limits"]["artifact_cap_bytes"])
                current = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                completed = position + 1
                projected = current + (current / completed) * (total - completed)
                if projected > cap:
                    raise RuntimeError(
                        "ProjectedArtifactCapExceeded: full MUTACO records would "
                        f"require approximately {projected:.0f} bytes (cap={cap})"
                    )
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "target_index": index,
                            "method": "mutaco",
                            "status": row["status"],
                            "elapsed_seconds": perf_counter() - start,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        rows.append(row)
        elapsed = perf_counter() - start
        eta = elapsed / (position + 1) * (total - position - 1)
        print(
            f"[mutaco | target {position + 1:03d}/{total:03d}] {row['status']} "
            f"target_l2={row.get('target_error_l2')} "
            f"design_l2={row.get('design_error_l2')} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )
    return rows, {
        "executed_target_count": executed_count,
        "recovered_target_count": recovered_count,
        "target_processing_wall_seconds": perf_counter() - start,
        "sum_recorded_target_runtime_seconds": sum(
            float(row["runtime_seconds"]) for row in rows
        ),
    }


def replay_reference(
    *,
    adapter: SimulatorAdapter,
    config: Mapping[str, Any],
    theta_bank: torch.Tensor,
    target_indices: tuple[int, ...],
    campaign_science: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    target_records: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    positions = replay_indices(len(rows))
    evidence = []
    for position in positions:
        index = target_indices[position]
        replay = run_mutaco_target(
            adapter=adapter,
            index=index,
            theta_star=theta_bank[index],
            config=config,
            campaign_science=campaign_science,
            device=device,
            dtype=dtype,
            supplied_target_response=(
                torch.tensor(
                    target_records[index]["target_response"],
                    device=device,
                    dtype=dtype,
                ).contiguous()
                if target_records is not None
                else None
            ),
            supplied_target_seed=(
                int(target_records[index]["target_seed"])
                if target_records is not None
                else None
            ),
            supplied_target_identity=(
                str(target_records[index]["target_record_id"])
                if target_records is not None
                else None
            ),
        )
        primary_science = _replay_projection(rows[position])
        replay_science = _replay_projection(replay)
        equal = canonical_json_bytes(primary_science) == canonical_json_bytes(
            replay_science
        )
        evidence.append(
            {
                "target_index": index,
                "equal": equal,
                "primary_hash": stable_hash("cohort-replay-v1", primary_science),
                "replay_hash": stable_hash(
                    "cohort-replay-v1", replay_science
                ),
            }
        )
    return {
        "policy": "first_middle_last",
        "indices": tuple(target_indices[position] for position in positions),
        "results": evidence,
    }


def _replay_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Stable completed or typed-failure projection; timings are observational."""

    if row.get("scientific_projection") is not None:
        return {
            "status": row.get("status"),
            "target_record_id": row.get("target_record_id"),
            "scientific_projection": row.get("scientific_projection"),
            "assessment_response": row.get("assessment_response"),
        }
    return {
        "status": row.get("status"),
        "failure_explanation": row.get("failure_explanation"),
        "target_record_id": row.get("target_record_id"),
        "target_seed": row.get("target_seed"),
        "target_response": row.get("target_response"),
        "inference_calls": row.get("inference_calls"),
    }
