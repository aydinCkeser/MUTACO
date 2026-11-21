from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.paper.oscillator_physics_post_analysis import analysis


def test_declaration_has_exact_eight_normalized_perturbations() -> None:
    declaration = analysis.load_declaration()
    points = declaration["finite_difference"]["ordered_points"]
    assert len(points) == 8
    assert declaration["normalized_half_step"] == 0.05
    assert declaration["call_contract"]["center_calls"] == 0
    assert declaration["call_contract"]["retry_calls"] == 0
    assert [point["theta"] for point in points] == [
        [1.8354227542877197, 1.115454077720642, 1.1611828804016113, 0.880811870098114],
        [1.9854227304458618, 1.115454077720642, 1.1611828804016113, 0.880811870098114],
        [1.910422682762146, 1.0404541492462158, 1.1611828804016113, 0.880811870098114],
        [1.910422682762146, 1.190454125404358, 1.1611828804016113, 0.880811870098114],
        [1.910422682762146, 1.115454077720642, 1.0861828327178955, 0.880811870098114],
        [1.910422682762146, 1.115454077720642, 1.2361829280853271, 0.880811870098114],
        [1.910422682762146, 1.115454077720642, 1.1611828804016113, 0.8308118581771851],
        [1.910422682762146, 1.115454077720642, 1.1611828804016113, 0.930811882019043],
    ]


def test_zero_call_analysis_uses_only_frozen_response() -> None:
    declaration = analysis.load_declaration()
    target = analysis.load_frozen_target(declaration)
    reconstruction = analysis.load_frozen_reconstruction(declaration)
    gauge, trajectories = analysis.build_zero_call_documents(
        declaration, target, reconstruction
    )
    assert gauge["scientific_calls"] == trajectories["scientific_calls"] == 0
    assert len(trajectories["rows"]) == 20
    assert {row["series"] for row in trajectories["rows"]} == {
        "target", "mutaco_b40_reconstruction"
    }
    assert len(gauge["rows"]) == 10
    assert gauge["rows"][0] == {
        "series": "target",
        "force": 0.0,
        "omega_bar_pi_minus_zero": 0.0004995409253751859,
        "M_pi_minus_zero": -0.16046404838562012,
    }
    assert gauge["rows"][5] == {
        "series": "mutaco_b40_reconstruction",
        "force": 0.0,
        "omega_bar_pi_minus_zero": 0.0003016288537764922,
        "M_pi_minus_zero": -0.17285972833633423,
    }


def test_preflight_is_non_mutating_and_reports_closed_gate(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    preview = analysis.validate_preflight(output)
    assert preview["scientific_calls_executed"] == 0
    assert preview["job_submitted"] is False
    assert preview["submission_authorized"] is False
    assert len(preview["pending_theta"]) == 8
    assert not output.exists()


def test_worker_gate_fails_before_creating_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUTACO_OSCILLATOR_EIGHT_CALLS_AUTHORIZED", raising=False)
    output = tmp_path / "closed"
    with pytest.raises(analysis.PostAnalysisError, match="gate is closed"):
        analysis.execute_worker(output, analysis.AUTHORIZATION_PHRASE)
    assert not output.exists()


def test_mocked_worker_makes_exactly_eight_no_retry_calls_and_builds_jacobian(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    manifest = {"quenched_disorder/0": 11, "initial_state/0": 12, "thermal_noise/0": 13}

    class FakeSimulator:
        def evaluate(self, theta: torch.Tensor, context):
            calls.append((theta.clone(), context))
            values = torch.arange(20, dtype=torch.float32) + theta.sum()
            return SimpleNamespace(
                values=values,
                metadata={"transport_call_count": 1, "seed_manifest": manifest},
            )

    monkeypatch.setattr(analysis, "_simulator", lambda declaration: FakeSimulator())
    monkeypatch.setenv("MUTACO_OSCILLATOR_EIGHT_CALLS_AUTHORIZED", "1")
    output = tmp_path / "executed"
    result = analysis.execute_worker(output, analysis.AUTHORIZATION_PHRASE)
    assert len(calls) == 8
    assert len({call[1].evaluation_seed for call in calls}) == 1
    assert result["call_accounting"] == {
        "attempted_calls": 8, "successful_calls": 8, "failed_calls": 0,
        "retry_calls": 0, "center_calls": 0,
    }
    assert len(list((output / "points").glob("*.intent.json"))) == 8
    assert len(list((output / "points").glob("*.outcome.json"))) == 8
    reloaded = json.loads((output / "sensitivity.json").read_text(encoding="utf-8"))
    assert reloaded == result
    assert len(result["jacobian_wrt_normalized"]) == 20
    assert all(len(row) == 4 for row in result["jacobian_wrt_normalized"])


def test_truba_wrapper_has_validation_and_explicit_submission_gate() -> None:
    package = Path(analysis.__file__).resolve().parent
    submit = (package / "truba/submit.sh").read_text(encoding="utf-8")
    worker = (package / "truba/worker.sbatch").read_text(encoding="utf-8")
    assert "--validate-only" in submit and "--submit" in submit
    assert "MUTACO_OSCILLATOR_EIGHT_CALLS_AUTHORIZED" in submit
    assert "--array" not in submit
    assert "execute-worker" in worker
    assert analysis.AUTHORIZATION_PHRASE in worker
    assert "sbatch" not in worker
