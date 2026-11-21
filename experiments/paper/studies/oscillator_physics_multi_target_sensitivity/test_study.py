from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.paper.oscillator_physics_multi_target_sensitivity import study


EXPECTED_TARGETS = [25, 64, 34, 48, 119, 86, 57, 13, 111, 96, 45, 58]


def test_selection_is_exact_and_uses_only_frozen_theta_and_bounds() -> None:
    declaration = study.load_declaration()
    assert declaration["selection"]["selected_target_indices"] == EXPECTED_TARGETS
    assert declaration["selection"]["target098_selected"] is False
    assert declaration["selection"]["information_inputs"] == [
        "frozen_theta_star", "p4_paper_baseline_parameter_bounds"
    ]
    assert declaration["scientific_calls_executed"] == 0
    assert declaration["jobs_submitted"] == 0


def test_boundary_aware_symmetric_float32_design_is_exact() -> None:
    declaration = study.load_declaration()
    reduced = []
    assert len(declaration["targets"]) == 12
    assert sum(len(target["ordered_points"]) for target in declaration["targets"]) == 96
    assert [point["global_point_index"] for target in declaration["targets"]
            for point in target["ordered_points"]] == list(range(96))
    for target in declaration["targets"]:
        assert len(target["coordinates"]) == 4
        assert len(target["ordered_points"]) == 8
        for coordinate in target["coordinates"]:
            z = coordinate["normalized_center"]
            boundary = min(z, 1.0 - z)
            h = min(0.05, 0.9 * boundary)
            assert coordinate["nearest_boundary_distance"] == boundary
            assert coordinate["nominal_normalized_half_step"] == h
            assert coordinate["boundary_reduced_from_0p05"] == (h < 0.05)
            a = coordinate["parameter_index"]
            minus = coordinate["persisted_float32_minus_theta"]
            plus = coordinate["persisted_float32_plus_theta"]
            assert coordinate["actual_physical_denominator"] == plus[a] - minus[a]
            for point in (minus, plus):
                assert all(lower <= value <= upper
                           for value, (lower, upper) in zip(point, study.BOUNDS, strict=True))
            if h < 0.05:
                reduced.append((target["target_index"], a))
    assert reduced == [
        (64, 2), (64, 3), (34, 0), (48, 0), (48, 1), (119, 2),
        (86, 1), (86, 3), (57, 1), (57, 3), (13, 2), (111, 0), (58, 2),
    ]


def test_crn_is_shared_within_target_unique_between_targets() -> None:
    targets = study.load_declaration()["targets"]
    seeds = [target["common_random_numbers"]["evaluation_seed"] for target in targets]
    run_ids = [target["common_random_numbers"]["scientific_run_id"] for target in targets]
    record_ids = [target["target_record_id"] for target in targets]
    assert len(set(seeds)) == len(set(run_ids)) == 12
    assert len(set(record_ids)) == 12
    assert study.TARGET098_DIAGNOSTIC_SEED not in seeds
    for target in targets:
        crn = target["common_random_numbers"]
        assert crn["scope"] == "all_eight_points_within_target_only"
        assert crn["shared_namespaces"] == [
            "quenched_disorder", "initial_state", "thermal_noise"
        ]


def test_call_contract_and_preflight_are_closed_and_nonmutating(tmp_path: Path) -> None:
    declaration = study.load_declaration()
    assert declaration["call_contract"] == {
        "targets": 12, "maximum_calls_per_target": 8,
        "required_successful_calls_per_target": 8,
        "maximum_simulator_calls_global": 96,
        "required_successful_calls_global": 96,
        "calls_per_point": 1, "center_calls": 0, "retry_calls": 0,
        "automatic_retry": False, "resubmission": False,
    }
    output = tmp_path / "fresh"
    receipt = study.validate_preflight(output)
    assert receipt["scientific_calls_executed"] == 0
    assert receipt["job_submitted"] is False
    assert not output.exists()
    with pytest.raises(study.StudyError, match="gate is closed"):
        study.execute_worker(output, study.AUTHORIZATION_PHRASE)
    assert not output.exists()


def test_boundary_violation_fails_closed() -> None:
    row = dict(study._cohort()[0])
    row["theta_star"] = [0.4, 1.0, 1.0, 0.5]
    with pytest.raises(study.StudyError, match="half-step is not positive"):
        study._target_design(row, 1, 0.0)


def test_mock_worker_has_exact_96_call_single_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    coefficients = torch.arange(1, 81, dtype=torch.float32).reshape(20, 4) / 80.0

    class FakeSimulator:
        def evaluate(self, theta: torch.Tensor, context):
            calls.append((theta.clone(), context.evaluation_seed))
            return SimpleNamespace(
                values=coefficients @ theta,
                metadata={"transport_call_count": 1,
                          "seed_manifest": {"context": context.evaluation_seed}},
            )

    monkeypatch.setattr(study, "_simulator", lambda declaration: FakeSimulator())
    monkeypatch.setenv(study.AUTHORIZATION_ENV, "1")
    output = tmp_path / "result"
    result = study.execute_worker(output, study.AUTHORIZATION_PHRASE)
    assert len(calls) == 96
    assert result["call_accounting"] == {
        "attempted": 96, "successful": 96, "failed": 0, "retry": 0, "center": 0
    }
    assert len({seed for _, seed in calls}) == 12
    assert all(len({seed for _, seed in calls[8*i:8*i+8]}) == 1 for i in range(12))
    assert len(list(output.glob("target_*/points/*.intent.json"))) == 96
    assert len(list(output.glob("target_*/points/*.outcome.json"))) == 96
    assert len(list(output.glob("target_*/sensitivity.json"))) == 12
    assert json.loads((output / "manifest.json").read_text()) == result


def test_truba_surface_is_single_nonarray_and_explicitly_gated() -> None:
    submit = (study.PACKAGE / "truba/submit.sh").read_text()
    worker = (study.PACKAGE / "truba/worker.sbatch").read_text()
    assert "--validate-only" in submit and "--submit" in submit
    assert study.AUTHORIZATION_ENV in submit and study.AUTHORIZATION_ENV in worker
    assert "--array" not in submit and "#SBATCH --array" not in worker
    assert submit.count("sbatch ") == 1
    assert "execute-worker" in worker and study.AUTHORIZATION_PHRASE in worker
    assert "sbatch" not in worker
    choices = study.build_parser()._subparsers._group_actions[0].choices
    assert choices.keys() == {"print-declaration", "preflight", "execute-worker"}
