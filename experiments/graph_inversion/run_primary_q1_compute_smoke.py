"""Run the approved zero-simulator Q1 real-GP compute smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from mutaco import ParameterSpec, stable_hash
from experiments.benchmarking.artifacts import encoded_nonfinite_kind, read_json
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark.qualification import (
    NEIQualificationDeclaration,
    load_nei_qualification_declaration,
)
from experiments.paper.primary_benchmark.qualification_execution import append_hashed_json
from .primary_q1_driver import GraphPrimaryQ1Driver


SCHEMA_VERSION = "mutaco-primary-q1-real-gp-compute-smoke-v1"


class SyntheticSmokeStop(RuntimeError):
    """Expected stop before the first sequential response can be evaluated."""


def smoke_preview(declaration: NEIQualificationDeclaration) -> dict[str, Any]:
    """Describe the bounded smoke without constructing a GP or evaluating anything."""

    return {
        "schema_version": f"{SCHEMA_VERSION}-preview",
        "declaration_identity": declaration.document()["declaration_identity"],
        "target_record_identity": declaration.target_record_identities[0],
        "budget": 40,
        "snapshot_positions": [16],
        "expected_tier_observations": len(declaration.tiers),
        "synthetic_observation_evaluations": 0,
        "simulator_calls": 0,
        "scientific_calls": 0,
        "scientific_calls_executed": 0,
        "job_submitted": False,
    }


def _synthetic_response(
    theta: Sequence[float],
    *,
    parameter_spec: ParameterSpec,
    response_dimension: int,
) -> list[float]:
    """Return a deterministic smooth response without owning a simulator."""

    value = torch.tensor(theta, dtype=torch.float32)
    lower = torch.tensor(parameter_spec.lower, dtype=torch.float32)
    upper = torch.tensor(parameter_spec.upper, dtype=torch.float32)
    normalized = (value - lower) / (upper - lower)
    response_index = torch.arange(1, response_dimension + 1, dtype=torch.float32)
    weighted = torch.dot(
        normalized,
        torch.tensor((0.7, 1.1, 1.3, 1.7), dtype=torch.float32),
    )
    response = (
        torch.sin(response_index * (0.035 + 0.021 * weighted))
        + 0.35 * torch.cos(response_index * 0.11 + normalized[0])
        + 0.2 * normalized[(response_index.to(torch.int64) - 1) % normalized.numel()]
        + 0.005 * response_index * normalized.mean()
    )
    return response.contiguous().tolist()


def _nonfinite_counts(value: Any) -> dict[str, int]:
    counts = {"negative_infinity": 0, "positive_infinity": 0}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            marker = encoded_nonfinite_kind(item)
            if marker in counts:
                counts[str(marker)] += 1
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return counts


def _fit_summary(diagnostics: Any) -> list[dict[str, Any]]:
    if not isinstance(diagnostics, list):
        return []
    summaries = []
    for row in diagnostics:
        if not isinstance(row, Mapping):
            continue
        attempts = row.get("recovery_attempts")
        selected = None
        if isinstance(attempts, (list, tuple)):
            selected = next(
                (attempt for attempt in attempts if attempt.get("selected") is True),
                None,
            )
        summaries.append({
            "training_version": row.get("training_version"),
            "status": row.get("status"),
            "fit_id": row.get("fit_id"),
            "model_id": row.get("model_id"),
            "training_data_id": row.get("training_data_id"),
            "failure_kind": row.get("failure_kind"),
            "failure_message": row.get("failure_message"),
            "recovery_attempt_count": (
                len(attempts) if isinstance(attempts, (list, tuple)) else None
            ),
            "selected_recovery_stage": (
                selected.get("recovery_stage")
                if isinstance(selected, Mapping)
                else None
            ),
        })
    return summaries


def _source_commit(root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).strip()
    if dirty:
        raise PaperContractError("Q1 compute smoke requires a clean exact source")
    return commit


def execute_smoke(
    *,
    declaration: NEIQualificationDeclaration,
    configuration_path: Path,
    output: Path,
    expected_commit: str,
    torch_intraop_threads: int,
    torch_interop_threads: int,
) -> dict[str, Any]:
    """Fit the real GP and serialize one full snapshot using synthetic responses."""

    if output.exists():
        raise PaperContractError("Q1 compute-smoke output must be a new append-only path")
    root = Path(__file__).resolve().parents[2]
    commit = _source_commit(root)
    if commit != expected_commit:
        raise PaperContractError("Q1 compute smoke source commit mismatch")
    if torch_intraop_threads < 1 or torch_interop_threads != 1:
        raise PaperContractError("Q1 compute smoke requires positive intraop and one interop thread")
    torch.set_num_threads(torch_intraop_threads)
    torch.set_num_interop_threads(torch_interop_threads)

    driver = GraphPrimaryQ1Driver(
        declaration=declaration,
        configuration_path=configuration_path,
    )
    target_unit = torch.tensor((0.37, 0.53, 0.71, 0.29), dtype=torch.float32)
    lower = torch.tensor(driver.parameter_spec.lower, dtype=torch.float32)
    upper = torch.tensor(driver.parameter_spec.upper, dtype=torch.float32)
    target_theta = lower + target_unit * (upper - lower)
    target = {
        "target_record_id": declaration.target_record_identities[0],
        "target_index": 0,
        "target_response": _synthetic_response(
            target_theta.tolist(),
            parameter_spec=driver.parameter_spec,
            response_dimension=driver.response_spec.dimension,
        ),
    }
    synthetic_evaluations = 0

    def evaluate(theta: Sequence[float], evaluation_index: int) -> Sequence[float]:
        nonlocal synthetic_evaluations
        if evaluation_index != synthetic_evaluations + 1:
            raise PaperContractError("synthetic smoke evaluation order changed")
        if evaluation_index > 16:
            raise SyntheticSmokeStop(
                "expected stop after the position-16 qualification snapshot"
            )
        synthetic_evaluations += 1
        return _synthetic_response(
            theta,
            parameter_spec=driver.parameter_spec,
            response_dimension=driver.response_spec.dimension,
        )

    result = dict(driver.run_trajectory(
        target,
        40,
        str(declaration.trajectory_tier_id),
        (16,),
        evaluate,
    ))
    snapshots = result.get("snapshots")
    snapshot = snapshots[0] if isinstance(snapshots, list) and len(snapshots) == 1 else None
    tier_observations = (
        snapshot.get("tier_observations") if isinstance(snapshot, Mapping) else None
    )
    counts = _nonfinite_counts(snapshot)
    passed = (
        result.get("status") == "failed"
        and result.get("failure_stage") == "simulation"
        and result.get("failure_type") == "SyntheticSmokeStop"
        and synthetic_evaluations == 16
        and isinstance(snapshot, Mapping)
        and snapshot.get("snapshot_position") == 16
        and snapshot.get("baseline_size") == 16
        and snapshot.get("simulator_calls") == 0
        and snapshot.get("nonfinite_float_encoding")
        == "explicit_signed_infinity_v1"
        and isinstance(tier_observations, list)
        and len(tier_observations) == len(declaration.tiers)
        and counts["positive_infinity"] > 0
    )
    tier_summary = []
    if isinstance(tier_observations, list):
        tier_summary = [
            {
                "tier_id": row.get("tier_id"),
                "candidate_source": row.get("candidate_source"),
                "optimizer_success": row.get("optimizer_success"),
                "normalized_regret": row.get("normalized_regret"),
                "failure_flags": row.get("failure_flags"),
            }
            for row in tier_observations
        ]
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "source_commit": commit,
        "declaration_identity": declaration.document()["declaration_identity"],
        "configuration": str(configuration_path.as_posix()),
        "target_record_identity": target["target_record_id"],
        "budget": 40,
        "snapshot_position": 16,
        "snapshot_identity": (
            snapshot.get("snapshot_identity") if isinstance(snapshot, Mapping) else None
        ),
        "snapshot_state_identity": (
            snapshot.get("gp_snapshot", {}).get("snapshot_state_identity")
            if isinstance(snapshot, Mapping)
            else None
        ),
        "nonfinite_float_encoding": "explicit_signed_infinity_v1",
        "encoded_nonfinite_counts": counts,
        "tier_observations": tier_summary,
        "fit_diagnostics": _fit_summary(result.get("fit_diagnostics")),
        "expected_stop": {
            "failure_stage": result.get("failure_stage"),
            "failure_type": result.get("failure_type"),
            "failure_explanation": result.get("failure_explanation"),
        },
        "synthetic_observation_evaluations": synthetic_evaluations,
        "simulator_calls": 0,
        "scientific_calls": 0,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
    }
    document = {
        **body,
        "smoke_identity": stable_hash(SCHEMA_VERSION, body),
    }
    append_hashed_json(output / "compute_smoke.json", document)
    if not passed:
        raise PaperContractError(
            "Q1 real-GP compute smoke failed; append-only evidence was preserved"
        )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--torch-intraop-threads", type=int, required=True)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    declaration = load_nei_qualification_declaration(read_json(args.declaration))
    preview = smoke_preview(declaration)
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.execute:
        return
    if not args.confirm_approved:
        raise SystemExit("Q1 compute smoke requires --confirm-approved")
    document = execute_smoke(
        declaration=declaration,
        configuration_path=args.config,
        output=args.output,
        expected_commit=args.expected_commit,
        torch_intraop_threads=args.torch_intraop_threads,
        torch_interop_threads=args.torch_interop_threads,
    )
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
