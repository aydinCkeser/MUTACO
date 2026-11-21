"""Append-only, capped Q1 orchestration independent of simulator ownership."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver, stable_hash
from experiments.benchmarking.artifacts import (
    canonicalize_evidence,
    file_hash,
    read_json,
    safe_evidence_repr,
    write_pretty_json,
)
from experiments.paper.protocol.population_v3 import shard_indices
from experiments.paper.protocol.validation import PaperContractError
from .qualification import NEIQualificationDeclaration


def append_hashed_json(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(document)
    sidecar = path.with_suffix(".sha256")
    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file():
            raise PaperContractError(f"incomplete append-only artifact: {path}")
        if file_hash(path) != sidecar.read_text(encoding="ascii").strip():
            raise PaperContractError(f"append-only artifact hash mismatch: {path}")
        if read_json(path) != value:
            raise PaperContractError(f"append-only artifact differs: {path}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pretty_json(path, value)
    sidecar.write_text(file_hash(path), encoding="ascii")
    return value


def load_hashed_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise PaperContractError(f"missing hash-paired artifact: {path}")
    if file_hash(path) != sidecar.read_text(encoding="ascii").strip():
        raise PaperContractError(f"artifact hash mismatch: {path}")
    return read_json(path)


def persist_qualification_attempt(
    *,
    output: Path,
    attempt_body: Mapping[str, Any],
    target_position: int,
    budget: int,
    worker_rank: int,
    trajectory_identifier: str,
) -> dict[str, Any]:
    """Canonicalize once, then hash and persist the same Q1 attempt object."""

    stage = "evidence_canonicalization"
    try:
        canonical_body = canonicalize_evidence(attempt_body)
        stage = "evidence_semantic_hash"
        attempt = {
            **canonical_body,
            "attempt_identity": stable_hash(
                "nei-q1-trajectory-attempt-v1", canonical_body
            ),
        }
    except Exception as exc:
        envelope_body = {
            "schema_version": "mutaco-nei-q1-evidence-failure-v1",
            "trajectory_identifier": trajectory_identifier,
            "target_record_identity": str(
                attempt_body.get("target_record_identity", "unknown")
            ),
            "method": "nei_bo",
            "budget": int(budget),
            "target_position": int(target_position),
            "worker_identifier": f"worker-{worker_rank:02d}",
            "failure_stage": stage,
            "exception_class": type(exc).__name__,
            "offending_field_path": getattr(exc, "path", None),
            "offending_value_type": getattr(exc, "value_type", None),
            "offending_value_repr": getattr(
                exc, "value_repr", safe_evidence_repr(exc)
            ),
            "diagnostic": safe_evidence_repr(str(exc), limit=1000),
        }
        safe_body = canonicalize_evidence(envelope_body)
        envelope = {
            **safe_body,
            "failure_identity": stable_hash(
                "nei-q1-evidence-failure-v1", safe_body
            ),
        }
        append_hashed_json(
            output
            / "trajectory_failures"
            / f"target-{target_position:02d}-b{budget}.json",
            envelope,
        )
        raise
    append_hashed_json(
        output / "trajectory_attempts" / f"target-{target_position:02d}-b{budget}.json",
        attempt,
    )
    return attempt


def qualification_worker_preview(
    declaration: NEIQualificationDeclaration,
    *,
    rank: int,
    workers: int,
    maximum_calls: int,
) -> dict[str, Any]:
    if isinstance(rank, bool) or isinstance(workers, bool) or workers < 1 or rank not in range(workers):
        raise PaperContractError("Q1 worker geometry is invalid")
    positions = shard_indices(5, rank=rank, workers=workers)
    planned = len(positions) * 140
    if planned > maximum_calls:
        raise PaperContractError("Q1 worker call cap rejects declared shard")
    return {
        "schema_version": "mutaco-nei-q1-worker-preview-v1",
        "declaration_identity": declaration.document()["declaration_identity"],
        "rank": rank,
        "workers": workers,
        "target_positions": list(positions),
        "planned_search_calls": planned,
        "maximum_calls_argument": maximum_calls,
        "heldout_calls": 0,
        "scientific_calls_executed": 0,
    }


@dataclass
class QualificationCallCap:
    maximum: int
    attempted: int = 0
    successful: int = 0
    failed: int = 0

    def invoke(self, operation: Callable[[], Sequence[float]]) -> Sequence[float]:
        if self.attempted >= self.maximum:
            raise PaperContractError("Q1 physical-call cap exhausted")
        self.attempted += 1
        try:
            value = operation()
        except Exception:
            self.failed += 1
            raise
        self.successful += 1
        return value

    def document(self) -> dict[str, int]:
        return {
            "maximum_calls": self.maximum,
            "attempted_calls": self.attempted,
            "successful_calls": self.successful,
            "failed_calls": self.failed,
        }


QualificationTrajectory = Callable[
    [str, int, str, tuple[int, ...], Callable[[Sequence[float], int], Sequence[float]]],
    Mapping[str, Any],
]


def execute_qualification_worker(
    *,
    declaration: NEIQualificationDeclaration,
    output: Path,
    rank: int,
    workers: int,
    maximum_calls: int,
    source_commit: str,
    run_trajectory: QualificationTrajectory,
    evaluate_search: Callable[
        [str, int, Sequence[float], Mapping[str, Any]], Sequence[float]
    ],
) -> dict[str, Any]:
    """Execute one Q1 shard; all physical calls cross the supplied hard cap."""

    preview = qualification_worker_preview(
        declaration, rank=rank, workers=workers, maximum_calls=maximum_calls
    )
    if output.exists():
        raise PaperContractError("Q1 worker output must be a new append-only path")
    identity = declaration.document()["declaration_identity"]
    start = {
        "schema_version": "mutaco-nei-q1-worker-start-v1",
        "source_commit": source_commit,
        "declaration_identity": identity,
        "preview": preview,
    }
    append_hashed_json(output / "worker_start.json", start)
    cap = QualificationCallCap(maximum_calls)
    records: list[dict[str, Any]] = []
    status = "completed"
    error: dict[str, str] | None = None
    try:
        for position in preview["target_positions"]:
            target = declaration.target_record_identities[int(position)]
            for budget, snapshots in (
                (40, declaration.snapshot_positions_b40),
                (100, declaration.snapshot_positions_b100),
            ):
                before = cap.document()
                family = stable_hash("nei-q1-search-family-v1", {
                    "campaign_id": declaration.campaign_id,
                    "target_record_identity": target,
                    "budget": budget,
                    "method": "nei_bo",
                    "declaration_identity": identity,
                })
                deriver = SeedDeriver(
                    ScientificIdentity(family), declaration.inference_campaign_seed
                )

                def evaluate(theta: Sequence[float], evaluation_index: int) -> Sequence[float]:
                    coordinate = SeedCoordinate(
                        "simulation",
                        (
                            declaration.campaign_id,
                            target,
                            budget,
                            "nei_bo",
                            evaluation_index,
                            "simulation",
                        ),
                    )
                    context = {
                        "family_identity": family,
                        "evaluation_seed": deriver.seed_for(coordinate),
                        "evaluation_index": evaluation_index,
                        "coordinate": {
                            "namespace": coordinate.namespace,
                            "coordinates": list(coordinate.coordinates),
                        },
                    }
                    return cap.invoke(
                        lambda: evaluate_search(target, budget, theta, context)
                    )

                attempt_failure: dict[str, str] | None = None
                result: dict[str, Any] | None = None
                try:
                    result = dict(run_trajectory(
                        target,
                        budget,
                        str(declaration.trajectory_tier_id),
                        snapshots,
                        evaluate,
                    ))
                    if result.get("status") == "failed":
                        attempt_failure = {
                            "failure_stage": str(result.get("failure_stage")),
                            "failure_type": str(result.get("failure_type")),
                            "failure_explanation": str(result.get("failure_explanation")),
                        }
                except Exception as exc:
                    attempt_failure = {
                        "failure_stage": "trajectory_callback",
                        "failure_type": type(exc).__name__,
                        "failure_explanation": str(exc),
                    }
                after = cap.document()
                attempt_body = {
                    "schema_version": "mutaco-nei-q1-trajectory-attempt-v1",
                    "declaration_identity": identity,
                    "target_record_identity": target,
                    "target_position": int(position),
                    "budget": budget,
                    "trajectory_tier_id": declaration.trajectory_tier_id,
                    "snapshot_positions": list(snapshots),
                    "physical_calls": {
                        key: int(after[key]) - int(before[key])
                        for key in ("attempted_calls", "successful_calls", "failed_calls")
                    },
                    "result": result,
                    "failure": attempt_failure,
                }
                attempt = persist_qualification_attempt(
                    output=output,
                    attempt_body=attempt_body,
                    target_position=int(position),
                    budget=budget,
                    worker_rank=rank,
                    trajectory_identifier=family,
                )
                if attempt_failure is not None:
                    raise PaperContractError(
                        "Q1 trajectory failed at "
                        f"{attempt_failure['failure_stage']}: "
                        f"{attempt_failure['failure_type']}: "
                        f"{attempt_failure['failure_explanation']}"
                    )
                if result is None:
                    raise PaperContractError("Q1 trajectory returned no result")
                canonical_result = attempt.get("result")
                if not isinstance(canonical_result, Mapping):
                    raise PaperContractError("Q1 canonical trajectory result is invalid")
                if cap.attempted - int(before["attempted_calls"]) != budget:
                    raise PaperContractError("Q1 trajectory did not use its exact budget")
                observations = canonical_result.get("tier_observations")
                if not isinstance(observations, list):
                    raise PaperContractError("Q1 trajectory omitted tier observations")
                body = {
                    "schema_version": "mutaco-nei-q1-trajectory-record-v1",
                    "declaration_identity": identity,
                    "target_record_identity": target,
                    "budget": budget,
                    "trajectory_tier_id": declaration.trajectory_tier_id,
                    "snapshot_positions": list(snapshots),
                    "search_calls": budget,
                    "heldout_calls": 0,
                    "result": canonical_result,
                }
                canonical_body = canonicalize_evidence(body)
                record = {
                    **canonical_body,
                    "record_identity": stable_hash(
                        "nei-q1-trajectory-record-v1", canonical_body
                    ),
                }
                append_hashed_json(
                    output / "records" / f"target-{position:02d}-b{budget}.json",
                    record,
                )
                records.append(record)
    except Exception as exc:
        status = "failed"
        error = {"failure_type": type(exc).__name__, "explanation": str(exc)}
    terminal = {
        "schema_version": "mutaco-nei-q1-worker-terminal-v1",
        "source_commit": source_commit,
        "declaration_identity": identity,
        "status": status,
        "completed_records": len(records),
        "physical_calls": cap.document(),
        "failure": error,
    }
    append_hashed_json(output / "worker_terminal.json", terminal)
    if status != "completed":
        assert error is not None
        raise PaperContractError(
            "Q1 worker failed; terminal evidence was preserved: "
            f"{error['failure_type']}: {error['explanation']}"
        )
    if cap.attempted != int(preview["planned_search_calls"]):
        raise PaperContractError("Q1 completed worker call ledger differs from preview")
    return terminal
