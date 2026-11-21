"""Zero-call Prompt 5D training-diagnostic recovery and artifact finalization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .artifacts import file_hash, read_json, write_pretty_json
from .methods.direct_posterior_diagnostics import tensorboard_training_diagnostics
from .prompt5d import _repository_provenance


_COLD_TRAINING = re.compile(
    r"^(?P<regime>[^/]+)/training/t(?P<target>\d+)_(?P<method>sbi_npe|sbi_snpe)$"
)


def _event_rounds(event_path: Path) -> list[dict[str, Any]]:
    audit = tensorboard_training_diagnostics(
        event_path, max_num_epochs=200, stop_after_epochs=20
    )
    return [
        {
            key: round_record[key]
            for key in (
                "round_index", "actual_epochs", "training_loss",
                "validation_loss", "epoch_durations_seconds", "best_epoch",
                "best_validation_loss", "reported_best_validation_loss",
                "stopping_reason",
            )
        }
        for round_record in audit["rounds"]
    ]


def recover_training_diagnostics(root: str | Path) -> dict[str, Any]:
    """Recover actual SBI 0.26.1 histories from preserved TensorBoard events."""

    campaign_root = Path(root).resolve()
    records = []
    for event_path in sorted(campaign_root.rglob("events.out.tfevents*")):
        directory = event_path.parent.relative_to(campaign_root).as_posix()
        match = _COLD_TRAINING.match(directory)
        if match:
            regime = match.group("regime")
            method = match.group("method")
            target_index: int | None = int(match.group("target"))
        elif directory == "amortized_shared_bank_336/training":
            regime = "amortized_shared_bank_336"
            method = "sbi_npe"
            target_index = None
        else:
            raise ValueError(f"unrecognized Prompt 5D training-event path: {directory}")
        rounds = _event_rounds(event_path)
        expected_rounds = 3 if method == "sbi_snpe" else 1
        if len(rounds) != expected_rounds:
            raise ValueError(
                f"{method} has {len(rounds)} histories; expected {expected_rounds}"
            )
        records.append({
            "regime": regime,
            "method": method,
            "target_index": target_index,
            "event_file": event_path.relative_to(campaign_root).as_posix(),
            "rounds": rounds,
        })
    if len(records) != 85:
        raise ValueError(f"expected 85 physical SBI training records, found {len(records)}")
    recovered = {
        "schema_version": "mutaco-prompt5d-recovered-training-v1",
        "interpretation": (
            "zero-call recovery from SBI 0.26.1 TensorBoard event scalars; "
            "recommendations and scientific records are unchanged"
        ),
        "physical_training_records": len(records),
        "round_histories": sum(len(record["rounds"]) for record in records),
        "records": records,
    }
    write_pretty_json(
        campaign_root / "training_diagnostics_recovered.json", recovered
    )
    (campaign_root / "training_diagnostics_recovered.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return recovered


def finalize_manifest(root: str | Path, *, repository_root: Path) -> dict[str, Any]:
    """Inventory final external logs and recovered diagnostics, then validate hashes."""

    campaign_root = Path(root).resolve()
    manifest_path = campaign_root / "manifest.json"
    manifest = read_json(manifest_path)
    artifacts = {
        path.relative_to(campaign_root).as_posix(): {
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        for path in campaign_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    manifest["artifacts"] = artifacts
    manifest["stored_artifact_bytes"] = sum(
        item["bytes"] for item in artifacts.values()
    )
    manifest["postprocessing"] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "repository": _repository_provenance(repository_root),
        "training_diagnostics_recovered": True,
        "scientific_calls": 0,
        "inference_calls": 0,
    }
    write_pretty_json(manifest_path, manifest)
    for relative, expected in artifacts.items():
        path = campaign_root / relative
        if (
            not path.is_file()
            or file_hash(path) != expected["sha256"]
            or path.stat().st_size != expected["bytes"]
        ):
            raise ValueError(f"final artifact validation failed: {relative}")
    return manifest


def postprocess_prompt5d(root: str | Path) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    recovered = recover_training_diagnostics(root)
    manifest = finalize_manifest(root, repository_root=repository_root)
    return {
        "training_records": recovered["physical_training_records"],
        "round_histories": recovered["round_histories"],
        "artifact_count": len(manifest["artifacts"]),
        "stored_artifact_bytes": manifest["stored_artifact_bytes"],
    }
