from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.benchmarking.artifacts import (
    NEGATIVE_INFINITY,
    NONFINITE_TAG,
    POSITIVE_INFINITY,
    canonicalize_evidence,
    file_hash,
)
from experiments.paper.common.cohort_v2 import validate_cohort_declaration
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark import (
    NEIQualificationDeclaration,
    NEINumericalTier,
    append_hashed_json,
    execute_qualification_worker,
    load_hashed_json,
    qualification_worker_preview,
    select_nei_numerical_tier,
)
from infrastructure.truba.collect_primary_benchmark import collect_q1
import experiments.paper.primary_benchmark.numerical_qualification as numerical
from mutaco import FrozenResponseTransform, ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec, ScientificIdentity, SeedDeriver
from experiments.graph_inversion.primary_q1_driver import GraphPrimaryQ1Driver
from experiments.graph_inversion.run_primary_q1_compute_smoke import (
    _nonfinite_counts,
    smoke_preview,
)


ROOT = Path(__file__).resolve().parents[2]


def _declaration() -> NEIQualificationDeclaration:
    return NEIQualificationDeclaration(
        campaign_id="q1-test",
        study_identity="study",
        target_ledger_hash="ledger",
        target_record_identities=("a", "b", "c", "d", "e"),
        tiers=(
            NEINumericalTier("low", 128, 256, 4, 40, 0.05),
            NEINumericalTier("high", 512, 1024, 12, 80, 0.03),
        ),
        snapshot_positions_b40=(16, 28, 40),
        snapshot_positions_b100=(16, 40, 70, 100),
        output_root="q1",
        trajectory_tier_id="high",
    )


def _observations(declaration: NEIQualificationDeclaration):
    rows = []
    for target in declaration.target_record_identities:
        for budget, positions in (
            (40, declaration.snapshot_positions_b40),
            (100, declaration.snapshot_positions_b100),
        ):
            for position in positions:
                for tier in declaration.tiers:
                    rows.append({
                        "target_record_identity": target,
                        "budget": budget,
                        "snapshot_position": position,
                        "tier_id": tier.tier_id,
                        "optimizer_success": True,
                        "normalized_regret": 0.005 if tier.tier_id == "low" else 0.0,
                        "failure_flags": [],
                    })
    return rows


def test_lowest_effort_passing_tier_is_selected_deterministically() -> None:
    declaration = _declaration()
    result = select_nei_numerical_tier(declaration, _observations(declaration))
    assert result["status"] == "qualified"
    assert result["selected_tier_id"] == "low"
    assert result["scientific_calls_added_by_selection"] == 0


def test_tier_selection_rejects_incomplete_or_duplicate_slots() -> None:
    declaration = _declaration()
    observations = _observations(declaration)
    with pytest.raises(PaperContractError, match="cover"):
        select_nei_numerical_tier(declaration, observations[:-1])
    with pytest.raises(PaperContractError, match="duplicated"):
        select_nei_numerical_tier(declaration, observations + observations[:1])


def test_q1_worker_preview_has_exact_five_target_700_call_cap() -> None:
    declaration = _declaration()
    previews = [
        qualification_worker_preview(declaration, rank=rank, workers=5, maximum_calls=140)
        for rank in range(5)
    ]
    assert sum(row["planned_search_calls"] for row in previews) == 700
    assert all(row["heldout_calls"] == 0 for row in previews)
    with pytest.raises(PaperContractError, match="call cap"):
        qualification_worker_preview(declaration, rank=0, workers=5, maximum_calls=139)


def test_append_only_hashed_artifacts_are_idempotent_but_immutable(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    append_hashed_json(path, {"value": 1})
    append_hashed_json(path, {"value": 1})
    assert path.with_suffix(".sha256").read_text() == file_hash(path)
    with pytest.raises(PaperContractError, match="differs"):
        append_hashed_json(path, {"value": 2})


def test_offcohort_theta_declaration_is_exact_and_has_no_responses() -> None:
    path = ROOT / "experiments/graph_inversion/target_declarations/primary_nei_q1_offcohort_theta_v1.json"
    declaration = validate_cohort_declaration(json.loads(path.read_text(encoding="utf-8")))
    assert declaration["target_count"] == 5
    assert declaration["classification"] == "frozen_hidden_theta_declaration_no_responses"
    assert all("target_response" not in row for row in declaration["records"])
    assert path.with_suffix(".sha256").read_text() == file_hash(path)


def test_q1_preparation_is_frozen_but_all_scientific_gates_are_false() -> None:
    plan = json.loads((
        ROOT / "experiments/graph_inversion/configs/paper/primary_nei_q1_preparation_v1.json"
    ).read_text())
    assert plan["target_construction"] == {
        "required_calls": 5,
        "authorized": False,
        "response_record_identities": None,
    }
    assert plan["qualification"]["maximum_search_calls"] == 700
    assert plan["qualification"]["authorized"] is False
    assert plan["q2"]["maximum_calls"] == 123
    assert plan["q2"]["authorized"] is False
    assert plan["scientific_calls_executed"] == 0


def test_q1_fake_worker_and_collector_preserve_exact_700_call_ledger(tmp_path: Path) -> None:
    declaration = _declaration()

    def run_trajectory(target, budget, tier, snapshots, evaluate):
        assert tier == "high"
        for index in range(1, budget + 1):
            evaluate((float(index),), index)
        rows = []
        for position in snapshots:
            for numerical_tier in declaration.tiers:
                rows.append({
                    "target_record_identity": target,
                    "budget": budget,
                    "snapshot_position": position,
                    "tier_id": numerical_tier.tier_id,
                    "optimizer_success": True,
                    "normalized_regret": 0.005,
                    "failure_flags": [],
                })
        return {"tier_observations": rows, "snapshot_count": len(snapshots)}

    worker = tmp_path / "workers" / "worker-00"
    terminal = execute_qualification_worker(
        declaration=declaration,
        output=worker,
        rank=0,
        workers=1,
        maximum_calls=700,
        source_commit="a" * 40,
        run_trajectory=run_trajectory,
        evaluate_search=lambda target, budget, theta, context: (theta[0],),
    )
    assert terminal["physical_calls"]["attempted_calls"] == 700
    result = collect_q1(
        declaration=declaration,
        workers_root=tmp_path / "workers",
        output=tmp_path / "collection",
        workers=1,
        expected_commit="a" * 40,
    )
    assert result["selected_tier_id"] == "low"
    assert result["scientific_calls_added_by_selection"] == 0


def test_q1_failed_trajectory_persists_underlying_result_before_budget_guard(
    tmp_path: Path,
) -> None:
    declaration = _declaration()

    def run_trajectory(target, budget, tier, snapshots, evaluate):
        del target, budget, tier, snapshots
        for index in range(1, 17):
            evaluate((float(index),), index)
        return {
            "status": "failed",
            "theta": [[float(index)] for index in range(16)],
            "responses_physical": [[float(index)] for index in range(16)],
            "fit_diagnostics": [{"status": "success"}],
            "proposal_diagnostics": [],
            "snapshots": [],
            "tier_observations": [],
            "failure_stage": "initial_gp_fit",
            "failure_type": "RuntimeError",
            "failure_explanation": "synthetic snapshot failure",
        }

    worker = tmp_path / "worker-00"
    with pytest.raises(PaperContractError, match="synthetic snapshot failure"):
        execute_qualification_worker(
            declaration=declaration,
            output=worker,
            rank=0,
            workers=5,
            maximum_calls=140,
            source_commit="a" * 40,
            run_trajectory=run_trajectory,
            evaluate_search=lambda target, budget, theta, context: (theta[0],),
        )
    attempt = load_hashed_json(
        worker / "trajectory_attempts" / "target-00-b40.json"
    )
    terminal = load_hashed_json(worker / "worker_terminal.json")
    assert attempt["physical_calls"] == {
        "attempted_calls": 16,
        "successful_calls": 16,
        "failed_calls": 0,
    }
    assert attempt["result"]["theta"] == [[float(index)] for index in range(16)]
    assert attempt["failure"] == {
        "failure_stage": "initial_gp_fit",
        "failure_type": "RuntimeError",
        "failure_explanation": "synthetic snapshot failure",
    }
    assert terminal["physical_calls"]["successful_calls"] == 16
    assert "initial_gp_fit: RuntimeError: synthetic snapshot failure" in (
        terminal["failure"]["explanation"]
    )


def test_snapshot_engine_compares_all_tiers_with_four_common_base_batches(monkeypatch) -> None:
    declaration = _declaration()
    dtype = torch.float64
    responses = torch.tensor([[0.1, 0.9], [0.4, 0.6]], dtype=dtype)
    transform = FrozenResponseTransform.fit(
        responses,
        response_spec=ResponseSpec((ResponseAxis("a"), ResponseAxis("b"))),
        observation_ids=("a", "b"),
    )
    candidates = iter((0.2, 0.8, 0.5))

    def fake_propose(*args, **kwargs):
        value = next(candidates)
        candidate = torch.tensor([value], dtype=dtype)
        return SimpleNamespace(
            candidate=candidate,
            candidate_source="local_refinement",
            provenance=lambda: {"candidate": [value], "simulator_calls": 0},
        )

    batch_ids = []

    def fake_evaluate(*args, **kwargs):
        config = kwargs["config"]
        batch_ids.append(config.mc_coordinate.coordinates)
        return SimpleNamespace(
            values=torch.tensor([0.1, 0.2, 0.3], dtype=dtype),
            provenance=lambda: {
                "base_sample_identity": str(config.mc_coordinate.coordinates),
                "values": [0.1, 0.2, 0.3],
                "simulator_calls": 0,
            },
        )

    class FakeSurrogate:
        def warm_start_state(self):
            return SimpleNamespace(
                source_fit_id="fit",
                source_model_id="model",
                compatibility_id="compat",
                consecutive_fixed_conditioning_updates=0,
                source_attempt_kind="cold",
                parameter_state={"weight": torch.tensor([1.0], dtype=dtype)},
            )

    monkeypatch.setattr(numerical, "propose_composite_mc_nei", fake_propose)
    monkeypatch.setattr(numerical, "evaluate_composite_mc_nei_candidates", fake_evaluate)
    result = numerical.evaluate_qualification_snapshot(
        declaration=declaration,
        surrogate=FakeSurrogate(),
        baseline=torch.tensor([[0.1], [0.9]], dtype=dtype),
        transform=transform,
        target_physical=torch.tensor([0.5, 0.5], dtype=dtype),
        parameter_spec=ParameterSpec((ParameterAxis("x", 0.0, 1.0),)),
        seed_deriver=SeedDeriver(ScientificIdentity("q1-reference-test"), 7),
        target_record_identity="a",
        budget=40,
        snapshot_position=16,
        run_scale=0.5,
    )
    assert len(result["tier_observations"]) == 2
    assert len(result["reference_batches"]) == 4
    assert len(set(batch_ids)) == 4
    assert result["tier_observations"][1]["normalized_regret"] < result["tier_observations"][0]["normalized_regret"]
    assert result["gp_snapshot"]["snapshot_state_identity"].startswith("nei-q1-gp-snapshot-state-v1-")


def test_q1_snapshot_state_encodes_constraint_infinities_and_rejects_nan() -> None:
    class InfiniteConstraintSurrogate:
        def warm_start_state(self):
            return SimpleNamespace(
                source_fit_id="fit",
                source_model_id="model",
                compatibility_id="compat",
                consecutive_fixed_conditioning_updates=0,
                source_attempt_kind="cold",
                parameter_state={
                    "constraint": torch.tensor(
                        [float("-inf"), 1.0, float("inf")], dtype=torch.float32
                    )
                },
            )

    state = numerical._snapshot_state(InfiniteConstraintSurrogate())
    assert state["nonfinite_float_encoding"] == "explicit_signed_infinity_v1"
    assert state["parameter_state"]["constraint"]["values"] == [
        {NONFINITE_TAG: NEGATIVE_INFINITY},
        1.0,
        {NONFINITE_TAG: POSITIVE_INFINITY},
    ]

    with pytest.raises(ValueError, match="forbids NaN"):
        numerical._canonical_numerical_document({"invalid": float("nan")})


def test_graph_q1_driver_binds_canonical_cpu_float32_specs_without_calls() -> None:
    driver = GraphPrimaryQ1Driver(
        declaration=_declaration(),
        configuration_path=(
            ROOT / "experiments/graph_inversion/configs/paper/paper120_b100_mutaco_v1.json"
        ),
    )
    assert driver.parameter_spec.dimension == 4
    assert driver.response_spec.dimension == 20
    assert driver.dtype is torch.float32


def test_q1_real_gp_compute_smoke_preview_and_nonfinite_counter_are_zero_call() -> None:
    preview = smoke_preview(_declaration())
    assert preview["budget"] == 40
    assert preview["snapshot_positions"] == [16]
    assert preview["expected_tier_observations"] == 2
    assert preview["synthetic_observation_evaluations"] == 0
    assert preview["simulator_calls"] == 0
    assert preview["scientific_calls"] == 0
    assert preview["scientific_calls_executed"] == 0
    assert preview["job_submitted"] is False
    assert _nonfinite_counts(canonicalize_evidence({
        "lower": float("-inf"),
        "nested": [float("inf")],
    })) == {"negative_infinity": 1, "positive_infinity": 1}
