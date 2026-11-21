from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from experiments.benchmarking.artifacts import (
    EvidenceCanonicalizationError,
    NEGATIVE_INFINITY,
    NONFINITE_TAG,
    POSITIVE_INFINITY,
    canonicalize_evidence,
)
from experiments.paper.primary_benchmark import (
    NEIQualificationDeclaration,
    NEINumericalTier,
    execute_qualification_worker,
    load_hashed_json,
    persist_qualification_attempt,
)
import experiments.paper.primary_benchmark.qualification_execution as execution
from mutaco import stable_hash


def _declaration() -> NEIQualificationDeclaration:
    return NEIQualificationDeclaration(
        campaign_id="q1-evidence-test",
        study_identity="study",
        target_ledger_hash="ledger",
        target_record_identities=("a", "b", "c", "d", "e"),
        tiers=(
            NEINumericalTier("low", 4, 4, 1, 1, 0.05),
            NEINumericalTier("high", 8, 8, 1, 1, 0.05),
        ),
        snapshot_positions_b40=(16, 40),
        snapshot_positions_b100=(16, 100),
        output_root="q1",
        trajectory_tier_id="high",
    )


def _complete_result(target: str, budget: int, snapshots: tuple[int, ...]) -> dict:
    snapshot_rows = [
        {
            "schema_version": "mutaco-nei-q1-snapshot-v1",
            "snapshot_position": position,
            "snapshot_identity": f"snapshot-{position}",
            "gp_snapshot": {
                "snapshot_state_identity": f"state-{position}",
                "parameter_state": {
                    "constraint": {"values": [float("-inf"), 1.0, float("inf")]}
                }
            },
            "tier_observations": [],
            "reference_batches": [{"values": [0.2, float("inf")]}],
        }
        for position in snapshots
    ]
    observations = [
        {
            "target_record_identity": target,
            "budget": budget,
            "snapshot_position": position,
            "tier_id": tier,
            "optimizer_success": True,
            "normalized_regret": 0.0,
            "failure_flags": [],
        }
        for position in snapshots
        for tier in ("low", "high")
    ]
    for snapshot in snapshot_rows:
        snapshot["tier_observations"] = [
            observation
            for observation in observations
            if observation["snapshot_position"] == snapshot["snapshot_position"]
        ]
    return {
        "status": "recommended",
        "search_calls": budget,
        "theta": [[0.1, 0.2, 0.3, 0.4] for _ in range(budget)],
        "responses_physical": [[0.5, 0.6] for _ in range(budget)],
        "fit_diagnostics": [
            {"status": "success", "optimizer": {"constraint_bound": float("inf")}}
        ],
        "proposal_diagnostics": [
            {
                "raw_values": [0.2, float("-inf")],
                "raw_valid": [True, False],
                "raw_failures": [None, "exact_evaluated_duplicate"],
            }
        ],
        "snapshots": snapshot_rows,
        "tier_observations": observations,
        "terminal_recommendation": {
            "selected_theta": [0.1, 0.2, 0.3, 0.4],
            "score": 0.1,
            "optimization": {"rejected_objective": float("inf")},
        },
        "trajectory_summary": {"worst_diagnostic": float("-inf")},
        "failure_stage": None,
        "failure_type": None,
        "failure_explanation": None,
    }


@dataclass(frozen=True)
class _Example:
    values: tuple[float, ...]


def test_evidence_canonicalizer_handles_python_numpy_and_recursive_shapes() -> None:
    value = {
        "python": [1.25, float("inf"), float("-inf")],
        "numpy": np.array([np.float32(2.5), np.float32(float("inf"))]),
        "scalar": np.float64(float("-inf")),
        "dataclass": _Example((3.0,)),
        "integer": np.int64(7),
    }
    canonical = canonicalize_evidence(value)
    assert canonical["python"] == [
        1.25,
        {NONFINITE_TAG: POSITIVE_INFINITY},
        {NONFINITE_TAG: NEGATIVE_INFINITY},
    ]
    assert canonical["numpy"][1] == {NONFINITE_TAG: POSITIVE_INFINITY}
    assert canonical["scalar"] == {NONFINITE_TAG: NEGATIVE_INFINITY}
    assert canonical["dataclass"] == {"values": [3.0]}
    assert canonical["integer"] == 7


def test_evidence_canonicalizer_rejects_nan_with_exact_nested_path() -> None:
    with pytest.raises(EvidenceCanonicalizationError) as caught:
        canonicalize_evidence(
            {"trajectory": {"fit_history": [{}, {"diagnostics": {"value": np.nan}}]}}
        )
    assert caught.value.path == "$.trajectory.fit_history[1].diagnostics.value"
    assert "forbids NaN" in str(caught.value)


def test_evidence_canonicalizer_rejects_reserved_tag_collision() -> None:
    with pytest.raises(EvidenceCanonicalizationError) as caught:
        canonicalize_evidence({"scientific": {NONFINITE_TAG: POSITIVE_INFINITY}})
    assert caught.value.path == f"$.scientific.{NONFINITE_TAG}"
    assert "collides" in str(caught.value)


def test_finite_only_semantic_identity_is_unchanged() -> None:
    body = {
        "schema_version": "mutaco-nei-q1-trajectory-attempt-v1",
        "result": {"values": (1.0, 2.0), "status": "recommended"},
    }
    assert stable_hash("nei-q1-trajectory-attempt-v1", body) == stable_hash(
        "nei-q1-trajectory-attempt-v1", canonicalize_evidence(body)
    )


def test_complete_b40_b100_results_use_production_hash_and_persistence_path(
    tmp_path: Path,
) -> None:
    declaration = _declaration()

    def run_trajectory(target, budget, tier, snapshots, evaluate):
        assert tier == "high"
        for index in range(1, budget + 1):
            evaluate((float(index),), index)
        return _complete_result(target, budget, snapshots)

    execute_qualification_worker(
        declaration=declaration,
        output=tmp_path / "worker-00",
        rank=0,
        workers=5,
        maximum_calls=140,
        source_commit="a" * 40,
        run_trajectory=run_trajectory,
        evaluate_search=lambda target, budget, theta, context: (theta[0],),
    )
    for budget in (40, 100):
        attempt = load_hashed_json(
            tmp_path
            / "worker-00"
            / "trajectory_attempts"
            / f"target-00-b{budget}.json"
        )
        body = {key: value for key, value in attempt.items() if key != "attempt_identity"}
        assert attempt["attempt_identity"] == stable_hash(
            "nei-q1-trajectory-attempt-v1", body
        )
        assert attempt["result"]["proposal_diagnostics"][0]["raw_values"][1] == {
            NONFINITE_TAG: NEGATIVE_INFINITY
        }
        assert attempt["result"]["fit_diagnostics"][0]["optimizer"][
            "constraint_bound"
        ] == {NONFINITE_TAG: POSITIVE_INFINITY}
        assert attempt["result"]["snapshots"][0]["gp_snapshot"][
            "parameter_state"
        ]["constraint"]["values"][0] == {NONFINITE_TAG: NEGATIVE_INFINITY}
        assert attempt["result"]["terminal_recommendation"]["optimization"][
            "rejected_objective"
        ] == {NONFINITE_TAG: POSITIVE_INFINITY}
        assert attempt["result"]["trajectory_summary"]["worst_diagnostic"] == {
            NONFINITE_TAG: NEGATIVE_INFINITY
        }
        record = load_hashed_json(
            tmp_path / "worker-00" / "records" / f"target-00-b{budget}.json"
        )
        record_body = {
            key: value for key, value in record.items() if key != "record_identity"
        }
        assert record["record_identity"] == stable_hash(
            "nei-q1-trajectory-record-v1", record_body
        )


@pytest.mark.parametrize(
    ("section", "expected_path"),
    (
        ("fit", "$.result.fit_diagnostics[0].optimizer.objective"),
        ("proposal", "$.result.proposal_diagnostics[0].raw_values[1]"),
        ("snapshot", "$.result.snapshots[0].gp_snapshot.objective"),
        (
            "terminal",
            "$.result.terminal_recommendation.optimization.objective",
        ),
        ("summary", "$.result.trajectory_summary.objective"),
    ),
)
def test_nested_nan_reports_path_and_persists_safe_failure_envelope(
    tmp_path: Path, section: str, expected_path: str
) -> None:
    result = _complete_result("a", 40, (16, 40))
    if section == "fit":
        result["fit_diagnostics"][0]["optimizer"]["objective"] = float("nan")
    elif section == "proposal":
        result["proposal_diagnostics"][0]["raw_values"][1] = float("nan")
    elif section == "snapshot":
        result["snapshots"][0]["gp_snapshot"]["objective"] = float("nan")
    elif section == "terminal":
        result["terminal_recommendation"]["optimization"]["objective"] = float("nan")
    else:
        result["trajectory_summary"]["objective"] = float("nan")
    body = {
        "schema_version": "mutaco-nei-q1-trajectory-attempt-v1",
        "declaration_identity": "declaration",
        "target_record_identity": "a",
        "target_position": 0,
        "budget": 40,
        "trajectory_tier_id": "high",
        "snapshot_positions": [16, 40],
        "physical_calls": {
            "attempted_calls": 40,
            "successful_calls": 40,
            "failed_calls": 0,
        },
        "result": result,
        "failure": None,
    }
    output = tmp_path / section
    with pytest.raises(EvidenceCanonicalizationError) as caught:
        persist_qualification_attempt(
            output=output,
            attempt_body=body,
            target_position=0,
            budget=40,
            worker_rank=3,
            trajectory_identifier="family",
        )
    assert caught.value.path == expected_path
    assert not (output / "trajectory_attempts" / "target-00-b40.json").exists()
    envelope = load_hashed_json(
        output / "trajectory_failures" / "target-00-b40.json"
    )
    assert envelope["failure_stage"] == "evidence_canonicalization"
    assert envelope["offending_field_path"] == expected_path
    assert envelope["offending_value_type"] == "float"
    assert "nan" in envelope["offending_value_repr"].lower()
    assert "result" not in envelope
    envelope_body = {
        key: value for key, value in envelope.items() if key != "failure_identity"
    }
    assert envelope["failure_identity"] == stable_hash(
        "nei-q1-evidence-failure-v1", envelope_body
    )


def test_attempt_hash_failure_still_persists_safe_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = execution.stable_hash

    def fail_attempt_hash(prefix: str, value: object) -> str:
        if prefix == "nei-q1-trajectory-attempt-v1":
            raise RuntimeError("synthetic semantic hash failure")
        return original(prefix, value)

    monkeypatch.setattr(execution, "stable_hash", fail_attempt_hash)
    body = {
        "schema_version": "mutaco-nei-q1-trajectory-attempt-v1",
        "target_record_identity": "a",
        "result": {"status": "recommended", "value": 1.0},
    }
    with pytest.raises(RuntimeError, match="synthetic semantic hash failure"):
        persist_qualification_attempt(
            output=tmp_path,
            attempt_body=body,
            target_position=0,
            budget=40,
            worker_rank=2,
            trajectory_identifier="family",
        )
    envelope = load_hashed_json(
        tmp_path / "trajectory_failures" / "target-00-b40.json"
    )
    assert envelope["failure_stage"] == "evidence_semantic_hash"
    assert envelope["exception_class"] == "RuntimeError"
    assert envelope["offending_field_path"] is None
    assert "result" not in envelope
