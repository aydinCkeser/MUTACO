"""Pre-migration persisted inputs and explicitly qualified fallback render bytes."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from figure_rendering_helpers import capture_rendering, load_plot_data, render_profile, RENDERERS


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = json.loads((ROOT / "tests/fixtures/offline_figure_rendering_before_2b.json").read_text())


@pytest.fixture(scope="module", params=sorted(EXPECTED["renderers"]))
def rendering(request, offline_figure_root, tmp_path_factory):
    module = importlib.import_module("experiments.paper.exploration." + request.param)
    result = capture_rendering(module, offline_figure_root, tmp_path_factory.mktemp(request.param))
    return request.param, result


def test_exact_plot_data_matches_pre_move_capture(rendering) -> None:
    name, result = rendering
    assert result["plot_data_sha256"] == EXPECTED["renderers"][name]["plot_data_sha256"]


def test_plot_artists_match_pre_move_capture(rendering) -> None:
    name, result = rendering
    if render_profile()["matplotlib"] != EXPECTED["profile"]["matplotlib"]:
        pytest.skip("artist snapshot is qualified for the recorded Matplotlib version")
    assert result["artists_sha256"] == EXPECTED["renderers"][name]["artists_sha256"]


def test_fallback_image_bytes_match_pre_move_capture(rendering) -> None:
    name, result = rendering
    if render_profile() != EXPECTED["profile"]:
        pytest.skip("fallback image bytes require the explicitly recorded renderer/font profile")
    assert result["fallback_files"] == EXPECTED["renderers"][name]["fallback_files"]


@pytest.mark.parametrize("name", sorted(EXPECTED["renderers"]))
def test_missing_and_corrupt_inputs_fail_before_rendering(name, tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("experiments.paper.exploration." + name)
    with pytest.raises(FileNotFoundError):
        load_plot_data(module, tmp_path)
    if name == RENDERERS[5]:
        path = tmp_path / "wrong.json"
        path.write_text("{}")
        monkeypatch.setattr(module, "DATA", path)
        with pytest.raises(AssertionError, match="Frozen plotting data changed"):
            module.main()
    else:
        path = tmp_path / "wrong.json"
        path.write_text("{}")
        with pytest.raises(RuntimeError, match="Frozen input hash mismatch"):
            module._require_hash(path, "0" * 64)
