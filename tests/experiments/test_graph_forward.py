from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from experiments.graph_forward import load_forward_artifacts
from experiments.graph_forward.plotting import render_forward_plots
from experiments.graph_forward.run import execute
from experiments.graph_forward.validate_cuda import execute as execute_cuda


def test_forward_experiment_artifacts_load_and_replot(tmp_path: Path) -> None:
    output = tmp_path / "forward"
    manifest = execute(output=output, device="cpu", seed=73007)

    loaded = load_forward_artifacts(output)

    assert manifest["cuda_parity_claimed"] is False
    assert loaded["result"]["model_identity"] == (
        "graph-oscillator.overdamped-scientific.v1"
    )
    assert len(loaded["result"]["response"]) == 20
    assert loaded["result"]["configuration"]["burn_in_steps"] == 100
    assert loaded["result"]["configuration"]["sample_steps"] == 900
    assert loaded["result"]["runtime"]["dtype"] == "float32"
    assert loaded["result"]["residual_transient_limitation"] is False
    for suffix in ("pdf", "svg"):
        assert (output / f"forward_response.{suffix}").stat().st_size > 100
        assert (output / f"burn_in_sensitivity.{suffix}").stat().st_size > 100
    replotted = render_forward_plots(loaded["result"], tmp_path / "replotted")
    assert len(replotted) == 4
    assert all(path.stat().st_size > 100 for path in replotted)


def test_forward_loader_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "forward"
    execute(output=output, device="cpu", seed=73007)
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["response"][0] += 1.0
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_forward_artifacts(output)


def test_cuda_validation_is_real_or_fails_explicitly(tmp_path: Path) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("real CUDA validation is intentionally executed by the bounded command")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        execute_cuda(tmp_path)
