from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.paper.exploration import scalar_primary_replay as replay


def test_json_metadata_container_normalization_preserves_exact_numbers():
    assert replay.json_metadata_equal({'indices':(), 'rankings':({'score':1.25},)},
                                      {'indices':[], 'rankings':[{'score':1.25}]})
    assert not replay.json_metadata_equal({'score':1.25},{'score':float(np.nextafter(1.25,2.0))})
    with pytest.raises(ValueError):replay.json_metadata_equal({'score':float('nan')},{'score':None})


@pytest.mark.parametrize("budget,target,index", [(40, 16, 48), (100, 0, 0)])
def test_committed_reference_archives_and_order(budget: int, target: int, index: int) -> None:
    record = replay.load_reference(budget, target)
    assert replay.REFERENCES[(budget, target)][0] == index
    trajectory = record["decision"]["trajectory"]
    assert len(trajectory) == budget
    assert record["decision"]["terminal_metadata"]["simulator_calls"] == 0
    assert {row["acquisition_diagnostics"]["branch"] for row in trajectory[16:]} == {"barrier", "exploration"}
    if budget == 40:
        assert trajectory[-1]["candidate_source"] == "raw_duplicate_recovery"


@pytest.fixture
def observation() -> dict:
    return {"evaluation_index": 1, "theta": [0.5, 1.0],
            "response_physical": [2.0, 3.0],
            "simulator_context": {"evaluation_index": 1, "logical_index": 0,
                                  "evaluation_seed": 17, "purpose": "primary_search_simulation",
                                  "coordinate": {"namespace": "simulation", "coordinates": [1]}}}


def test_observation_return_requires_exact_vector_and_owns_response(observation: dict) -> None:
    callback = replay.ArchivedResponses([observation])
    changed = [float(np.nextafter(np.float32(0.5), np.float32(1))), 1.0]
    with pytest.raises(replay.ReplayMismatch, match="candidate_vector"):
        callback(changed, observation["simulator_context"])
    assert callback.consumed == 0
    with pytest.raises(replay.ReplayMismatch, match="candidate_vector"):
        callback(observation["theta"], observation["simulator_context"])
    assert callback.consumed == 0
    callback = replay.ArchivedResponses([observation])
    result = callback(observation["theta"], observation["simulator_context"])
    result[0] = -1
    assert observation["response_physical"] == [2.0, 3.0]
    assert callback.consumed == 1
    with pytest.raises(replay.ReplayMismatch, match="observation_budget_exceeded"):
        callback(observation["theta"], observation["simulator_context"])


@pytest.mark.parametrize("field,value", [("evaluation_index", 2), ("logical_index", 1),
                                       ("evaluation_seed", 18), ("purpose", "heldout"),
                                       ("coordinate", {"namespace": "wrong", "coordinates": [1]})])
def test_context_or_order_mismatch_never_consumes_an_observation(observation: dict, field: str, value: object) -> None:
    callback = replay.ArchivedResponses([observation])
    context = deepcopy(observation["simulator_context"])
    context[field] = value
    with pytest.raises(replay.ReplayMismatch, match="effective_simulation_context"):
        callback(observation["theta"], context)
    assert callback.consumed == 0


def test_corrupted_archive_is_rejected_before_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "scalar-primary-b40-v3r2"
    directory = tmp_path / "b40-v3r2"
    directory.mkdir()
    original = replay.ARCHIVES / "b40-v3r2"
    for suffix in (".manifest.json", ".manifest.sha256", ".tar.sha256"):
        (directory / (name + suffix)).write_bytes((original / (name + suffix)).read_bytes())
    (directory / (name + ".tar.gz")).write_bytes(b"corrupted archive")
    monkeypatch.setattr(replay, "ARCHIVES", tmp_path)
    with pytest.raises(ValueError, match="compressed archive SHA-256 mismatch"):
        replay.verify_archive(40)


@pytest.mark.skipif(os.environ.get("MUTACO_RUN_GOLDEN_REPLAY") != "1",
                    reason="exact historical numerical replay is an explicit qualification gate")
@pytest.mark.parametrize("budget,target", [(40, 16), (100, 0)])
def test_exact_historical_production_replay(budget: int, target: int, tmp_path: Path) -> None:
    output = tmp_path / f"b{budget}-target{target}.json"
    completed = subprocess.run([
        sys.executable, "-m", "experiments.paper.exploration.scalar_primary_replay",
        "--budget", str(budget), "--target", str(target), "--output", str(output),
        "--torch-intraop-threads", "20", "--torch-interop-threads", "1",
    ], cwd=replay.ROOT, text=True, capture_output=True)
    assert output.is_file(), completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    assert report["physical_simulator_calls"] == 0
    assert report["ambient_rng_preserved"]
    assert completed.returncode == 0 and report["status"] == "passed", report.get("first_divergence", report)
