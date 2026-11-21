from __future__ import annotations

from copy import deepcopy
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from mutaco import FrozenResponseTransform, ResponseAxis, ResponseSpec, stable_hash
from experiments.paper.exploration import replay_divergence as diagnostics
from experiments.paper.exploration import scalar_primary_replay as replay


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bool])
def test_tensor_bytes_preserve_exact_bits_and_shape(dtype: torch.dtype) -> None:
    values = torch.tensor([[-0.0, 1.25], [float("inf"), -3.5]], dtype=dtype).T
    document = diagnostics.tensor_document(values)
    raw = base64.b64decode(document["data_base64"], validate=True)
    decoded = np.frombuffer(raw, dtype=document["numpy_dtype"]).reshape(document["shape"])
    assert document["byte_order"] == "little"
    assert hashlib.sha256(raw).hexdigest() == document["sha256_contiguous_bytes"]
    assert decoded.tobytes() == values.contiguous().numpy().tobytes()
    assert np.array_equal(decoded, values.numpy())
    if dtype != torch.bool:
        assert np.signbit(decoded[0, 0])  # signed zero survives serialization


def test_reference_packet_is_compact_deterministic_and_keeps_provenance(tmp_path: Path) -> None:
    capture = {
        "source": {"git_head": "diagnostic-source"},
        "reference": {"historical_execution_source_commit": "historical-source"},
        "environment": {"platform": "reference-computer"},
        "reconstructed": {"effective_candidate_17": diagnostics.tensor_document(torch.ones(4)),
                          "production_proposal": {"timings": {"total_seconds": 12},
                                                  "execution": {"interpreter_path": "/machine/python"}}},
        "replay_outcome": {"reconstructed_gp_fits": [{"elapsed_seconds": 7, "status": "success"}]},
    }
    changed_timers = deepcopy(capture)
    changed_timers["replay_outcome"]["reconstructed_gp_fits"][0]["elapsed_seconds"] = 99
    changed_timers["reconstructed"]["production_proposal"]["timings"]["total_seconds"] = 200
    paths = [tmp_path / "first.json.gz", tmp_path / "second.json.gz"]
    first = diagnostics.write_reference_packet(paths[0], capture)
    second = diagnostics.write_reference_packet(paths[1], changed_timers)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    packet = json.loads(gzip.decompress(paths[0].read_bytes()))
    assert packet == first == second
    assert packet["capture"]["source"] == capture["source"]
    assert packet["capture"]["reference"] == capture["reference"]
    assert packet["capture"]["environment"] == capture["environment"]
    assert "values" not in packet["capture"]["reconstructed"]["effective_candidate_17"]
    assert "elapsed_seconds" not in packet["capture"]["replay_outcome"]["reconstructed_gp_fits"][0]
    changed_machine = deepcopy(capture)
    changed_machine["environment"]["platform"] = "other-computer"
    changed_machine["reconstructed"]["production_proposal"]["execution"] = {"interpreter_path": "/other/python"}
    assert diagnostics.reference_packet(changed_machine)["numerical_state_sha256"] == first["numerical_state_sha256"]
    with pytest.raises(FileExistsError):
        diagnostics.write_reference_packet(paths[0], capture)


@pytest.mark.parametrize("entrypoint", ["evaluate", "evaluate_batch", "evaluate_condition_grid"])
def test_reference_capture_blocks_physical_simulator_entrypoints(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str,
) -> None:
    original = getattr(replay.OverdampedScientificSimulator, entrypoint)
    entered = []

    def would_run_physics(*args: object, **kwargs: object) -> None:
        entered.append(True)
        raise AssertionError("real simulator body must never be entered")

    monkeypatch.setattr(replay.OverdampedScientificSimulator, entrypoint, would_run_physics)

    def attempt_real_simulator(**kwargs: object) -> None:
        getattr(replay.OverdampedScientificSimulator, entrypoint)(object())

    monkeypatch.setattr(replay, "run_mutaco_target", attempt_real_simulator)
    capture = diagnostics.capture_first_adaptive(40, 16)
    assert not entered
    assert capture["replay_outcome"]["status"] == "invariant_failure"
    assert capture["replay_outcome"]["physical_simulator_calls"] == 1  # blocked attempt
    assert capture["replay_outcome"]["archived_observations_consumed"] == 0
    assert "capture_failure" in capture["reconstructed"]
    assert getattr(replay.OverdampedScientificSimulator, entrypoint) is would_run_physics
    assert original is not would_run_physics


def _transform(physical: torch.Tensor) -> dict:
    return FrozenResponseTransform.fit(
        physical,
        response_spec=ResponseSpec(tuple(ResponseAxis(f"response_{i}") for i in range(physical.shape[1]))),
        observation_ids=tuple(f"observation_{i}" for i in range(physical.shape[0])),
    ).metadata.provenance_document()


def _semantic(document: dict) -> dict:
    return {key: value for key, value in document.items() if key not in {"transform_id", "fit_device"}}


@pytest.fixture
def reduction_example() -> tuple[torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(19)
    physical = torch.randn((16, 20), generator=generator) * torch.logspace(-4, 4, 20)
    assert physical.dtype == torch.float32
    assert not torch.equal(physical.mean(0), physical.T.contiguous().mean(1))
    return physical, _transform(physical)


@pytest.mark.parametrize("json_roundtrip", [False, True])
def test_transform_probe_verifies_numerical_mean_change_against_complete_identity(
    reduction_example: tuple[torch.Tensor, dict], json_roundtrip: bool,
) -> None:
    physical, metadata = reduction_example
    before = physical.clone()
    metadata_before = deepcopy(metadata)
    historical = {**_semantic(metadata), "mean": physical.T.contiguous().mean(1).tolist()}
    historical_id = stable_hash("response-transform-v1", historical)
    if json_roundtrip:
        metadata = json.loads(json.dumps(metadata))
    result = diagnostics.transform_investigation(physical, metadata, historical_id)

    assert result["native_hash_self_consistent"]
    assert result["native_mean_matches_executed_transform"]
    assert result["alternate_full_semantic_hash_matches_history"]
    assert result["historical_values_origin"] == "hash_verified_reconstruction"
    assert result["changed_semantic_fields"] == ["mean"]
    assert result["mean_differences"]
    assert result["standardized_data_difference"]["different_elements"] > 0
    assert result["standardized_data_difference"]["max_abs_difference"] > 0
    assert result["alternate_id"] != result["native_id"]
    assert not result["production_replay_inputs_changed"]
    assert torch.equal(physical, before)
    assert _semantic(metadata) == (_semantic(json.loads(json.dumps(metadata_before))) if json_roundtrip else _semantic(metadata_before))


def test_transform_probe_recognizes_native_historical_match_when_alternative_differs(
    reduction_example: tuple[torch.Tensor, dict],
) -> None:
    physical, metadata = reduction_example
    result = diagnostics.transform_investigation(physical, metadata, metadata["transform_id"])

    assert result["native_hash_self_consistent"]
    assert result["native_full_semantic_hash_matches_history"]
    assert result["native_mean_matches_executed_transform"]
    assert result["historical_values_origin"] == "native_execution_hash_verified_against_history"
    assert not result["alternate_full_semantic_hash_matches_history"]
    assert result["alternate_id"] != result["historical_id"]
    assert result["mean_differences"]
    assert not result["production_replay_inputs_changed"]


def test_transform_probe_does_not_attribute_semantic_identity_change_to_numerics() -> None:
    physical = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    metadata = _transform(physical)
    historical = {**_semantic(metadata), "fitted_observation_ids": tuple(f"other_{i}" for i in range(16))}
    historical_id = stable_hash("response-transform-v1", historical)
    result = diagnostics.transform_investigation(physical, metadata, historical_id)

    assert result["native_hash_self_consistent"]
    assert result["native_mean_matches_executed_transform"]
    assert result["changed_semantic_fields"] == []
    assert result["mean_differences"] == []
    assert result["standardized_data_difference"]["exact"]
    assert result["standardized_data_difference"]["max_abs_difference"] == 0
    assert not result["alternate_full_semantic_hash_matches_history"]
    assert result["historical_values_origin"] == "unresolved"


def test_transform_probe_requires_all_historical_identity_fields(
    reduction_example: tuple[torch.Tensor, dict],
) -> None:
    physical, metadata = reduction_example
    historical = {
        **_semantic(metadata), "mean": physical.T.contiguous().mean(1).tolist(),
        "scale_floor": metadata["scale_floor"] * 2,
    }
    result = diagnostics.transform_investigation(
        physical, metadata, stable_hash("response-transform-v1", historical),
    )
    assert result["mean_differences"]
    assert not result["alternate_full_semantic_hash_matches_history"]
    assert result["historical_values_origin"] == "unresolved"


@pytest.mark.parametrize("case", ["matching", "candidate_mismatch", "context_mismatch"])
def test_bounded_capture_never_returns_observation_17(
    monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    class AdapterExercised(Exception):
        pass

    trajectory = [
        {"theta": [float(i)], "response_physical": [float(i + 100)],
         "simulator_context": {"evaluation_index": i + 1}}
        for i in range(17)
    ]
    original_adapter = replay.ArchivedResponses
    monkeypatch.setattr(replay, "load_reference", lambda *_: {})

    def fake_replay(*_: object) -> None:
        callback = replay.ArchivedResponses(trajectory)
        for row in trajectory[:16]:
            assert callback(row["theta"], row["simulator_context"]) == row["response_physical"]
        row = trajectory[16]
        theta, context = list(row["theta"]), dict(row["simulator_context"])
        if case == "matching":
            with pytest.raises(RuntimeError, match="stopped before returning observation 17"):
                callback(theta, context)
            assert callback.capture_complete
            assert callback.first_divergence is None
        else:
            if case == "candidate_mismatch":
                theta[0] += 1
                kind = "candidate_vector"
            else:
                context["evaluation_index"] += 1
                kind = "effective_simulation_context"
            with pytest.raises(replay.ReplayMismatch):
                callback(theta, context)
            assert callback.first_divergence["kind"] == kind
            assert callback.first_divergence["evaluation_index"] == 17
            assert not callback.capture_complete
        assert callback.consumed == 16
        # Stop the fake execution before the diagnostic's model-reporting stage.
        raise AdapterExercised

    monkeypatch.setattr(replay, "replay_reference", fake_replay)
    with pytest.raises(AdapterExercised):
        diagnostics.capture_first_adaptive(40, 16)
    assert replay.ArchivedResponses is original_adapter


def test_failed_first_fit_returns_partial_diagnostics_with_original_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = {"transform_id": "archived-transform", "raw_rankings": []}
    record = {"decision": {"trajectory": [{} for _ in range(16)] + [
        {"acquisition_diagnostics": historical},
    ]}}
    original_outcome = {
        "status": "diverged", "archived_observations_consumed": 16,
        "physical_simulator_calls": 0, "ambient_rng_preserved": True,
        "first_divergence": {"kind": "production_runner_failure", "evaluation_index": 17,
                             "explanation": "synthetic first-fit failure"},
    }
    calls = []

    def failed_run(**kwargs: object) -> None:
        calls.append("failed_run")
        kwargs["diagnostic_result_callback"](SimpleNamespace(transform=None))

    def fake_replay(*_: object) -> dict:
        replay.run_mutaco_target(diagnostic_result_callback=lambda _: None)
        return deepcopy(original_outcome)

    monkeypatch.setattr(replay, "load_reference", lambda *_: record)
    monkeypatch.setattr(replay, "run_mutaco_target", failed_run)
    monkeypatch.setattr(replay, "replay_reference", fake_replay)
    result = diagnostics.capture_first_adaptive(40, 16)

    assert calls == ["failed_run"]
    assert result["archived_first_proposal"] == historical
    assert {key: result["replay_outcome"][key] for key in original_outcome} == original_outcome
    assert result["replay_outcome"]["full_historical_replay_executed"] is False
    assert result["replay_outcome"]["golden_gate_verdict"] == "not_evaluated_by_bounded_capture"
    partial = result["reconstructed"]
    assert "did not reach a successful first proposal" in partial["capture_failure"]
    assert partial["model_builds"] == partial["restarts"] == partial["score_calls"] == []
    assert "production_proposal" not in partial
    assert partial["complete_diagnostic_ambient_rng_preserved"]


def _without_timings(value: object) -> object:
    if isinstance(value, dict):
        return {key: _without_timings(item) for key, item in value.items()
                if key != "timings" and not key.endswith("_seconds")}
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


@pytest.mark.skipif(
    os.environ.get("MUTACO_RUN_REPLAY_DIAGNOSTICS") != "1",
    reason="production fit/proposal diagnostic comparison is an explicit zero-call gate",
)
@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="qualification comparison targets the current macOS ARM64 execution",
)
@pytest.mark.parametrize("budget,target", [(40, 16), (100, 0)])
def test_capture_preserves_strict_replays_first_fit_and_adaptive_decision(
    budget: int, target: int, tmp_path: Path,
) -> None:
    reports = {}
    for module in ("scalar_primary_replay", "replay_divergence"):
        output = tmp_path / f"{module}-b{budget}.json"
        completed = subprocess.run([
            sys.executable, "-m", f"experiments.paper.exploration.{module}",
            "--budget", str(budget), "--target", str(target), "--output", str(output),
            "--torch-intraop-threads", "20", "--torch-interop-threads", "1",
        ], cwd=replay.ROOT, text=True, capture_output=True)
        assert output.is_file(), completed.stdout + completed.stderr
        assert completed.returncode == (2 if module == "scalar_primary_replay" else 0), completed.stdout + completed.stderr
        reports[module] = json.loads(output.read_text())

    strict = reports["scalar_primary_replay"]
    diagnostic = reports["replay_divergence"]
    observed = diagnostic["replay_outcome"]
    captured = diagnostic["reconstructed"]
    assert strict["status"] == observed["status"] == "diverged"
    assert strict["first_divergence"]["evaluation_index"] == 17
    assert strict["first_divergence"]["kind"] == "candidate_vector"
    assert strict["first_divergence"] == observed["first_divergence"]
    assert strict["frozen_transform_identity"] == observed["frozen_transform_identity"]
    assert strict["latest_proposal_reconstructed"] == observed["latest_proposal_reconstructed"]
    assert _without_timings(strict["reconstructed_gp_fits"]) == _without_timings(observed["reconstructed_gp_fits"])
    assert _without_timings(strict["reconstructed_proposals"]) == _without_timings(observed["reconstructed_proposals"])
    assert strict["logical_replay_accounting"] == observed["logical_replay_accounting"]
    assert strict["archived_observations_consumed"] == observed["archived_observations_consumed"] == 16
    assert strict["physical_simulator_calls"] == observed["physical_simulator_calls"] == 0
    assert strict["ambient_rng_preserved"] and observed["ambient_rng_preserved"]
    assert captured["complete_diagnostic_ambient_rng_preserved"]
    assert observed["full_historical_replay_executed"] is False
    assert observed["golden_gate_verdict"] == "not_evaluated_by_bounded_capture"
    assert {build["role"] for build in captured["model_builds"]} == {
        "fit_model_initialization", "fixed_parameter_float64_covariance_mirror",
    }
    assert captured["initial_theta_comparison"]["exact"]
    assert captured["initial_response_comparison"]["exact"]
    assert captured["executed_standardized_data_matches_direct_calculation"]
