from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from pypdf import PdfReader
import pytest
import torch

from mutaco import resolve_runtime, RuntimeRequest
from experiments.analytic_target_matching.noisy_cohort._artifacts import (
    recover_target,
)
from experiments.analytic_target_matching.noisy_cohort._campaign import (
    campaign_identity,
    replay_indices,
    target_bank,
)
from experiments.analytic_target_matching.noisy_cohort._metrics import (
    summarize,
    vector_errors,
)
from experiments.analytic_target_matching.noisy_cohort.run import execute


ROOT = Path.cwd()
SMOKE = ROOT / "experiments/analytic_target_matching/configs/noisy_cohort_smoke.json"
FULL = ROOT / "experiments/analytic_target_matching/configs/noisy_cohort_100.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cohort_size_bank_prefix_and_replay_policy() -> None:
    runtime = resolve_runtime(RuntimeRequest(device="cpu", dtype="float64"))
    small = _load(SMOKE)
    full = _load(FULL)
    theta_small, _, _ = target_bank(
        small, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    theta_full, _, _ = target_bank(
        full, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    assert theta_full.shape == (100, 2)
    assert torch.equal(theta_small, theta_full[:4])
    assert torch.unique(theta_full, dim=0).shape[0] == 100
    assert bool(((theta_full >= -1) & (theta_full <= 1)).all())
    assert replay_indices(1) == (0,)
    assert replay_indices(4) == (0, 1, 3)
    assert replay_indices(20) == (0, 9, 19)
    assert replay_indices(100) == (0, 49, 99)
    invalid = deepcopy(small)
    invalid["campaign"]["target_count"] = 0
    with pytest.raises(ValueError):
        target_bank(invalid, device=runtime.torch_device, dtype=runtime.torch_dtype)


def test_metrics_and_box_statistics_match_hand_calculation() -> None:
    actual = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)
    expected = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    errors = vector_errors(actual, expected)
    assert errors["l1"] == 3.0
    assert errors["l2"] == pytest.approx(5 ** 0.5)
    stats = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats["median"] == 3.0
    assert stats["q1"] == 2.0
    assert stats["q3"] == 4.0
    assert stats["outliers"] == [100.0]


def test_smoke_campaign_artifacts_replay_recovery_and_plots(tmp_path: Path) -> None:
    output = tmp_path / "cohort"
    output.mkdir()
    (output / "console.stdout.log").write_text("externally managed\n", encoding="utf-8")
    manifest = execute(config_path=SMOKE, output=output, device_request="cpu")
    assert manifest["summary"]["target_count"] == 4
    assert manifest["summary"]["completed_count"] == 4
    assert manifest["accounting"]["target_generation_calls"] == 4
    assert manifest["accounting"]["inference_calls"] == 20
    assert manifest["accounting"]["assessment_calls"] == 4
    assert manifest["accounting"]["gp_fits"] == 8
    assert manifest["accounting"]["replay_indices"] == (0, 1, 3)
    assert manifest["execution_mode"] == "fresh"
    assert manifest["cohort_execution"]["executed_target_count"] == 4
    assert manifest["cohort_execution"]["recovered_target_count"] == 0
    assert "console.stdout.log" not in manifest["artifacts"]
    assert "manifest.json" not in manifest["artifacts"]
    assert (output / "progress.jsonl").is_file()
    for index in range(4):
        directory = output / "targets" / f"target_{index:03d}"
        assert recover_target(directory, manifest["campaign_id"]) is not None
        for name in ("result", "assessment", "scientific_history"):
            assert (directory / f"{name}.json").is_file()
            assert (directory / f"{name}.sha256").is_file()
    for stem in ("primary_error_boxplots", "latent_response_boxplots"):
        ET.parse(output / "plots" / f"{stem}.svg")
        assert len(PdfReader(output / "plots" / f"{stem}.pdf").pages) == 1
    progress_before_restart = (output / "progress.jsonl").read_bytes()
    restarted = execute(
        config_path=SMOKE, output=output, device_request="cpu", restart=True
    )
    assert restarted["campaign_id"] == manifest["campaign_id"]
    assert restarted["execution_mode"] == "restart"
    assert restarted["cohort_execution"]["executed_target_count"] == 0
    assert restarted["cohort_execution"]["recovered_target_count"] == 4
    assert "manifest.json" not in restarted["artifacts"]
    assert (output / "progress.jsonl").read_bytes() == progress_before_restart
    assert not tuple(output.rglob("*.tmp"))
    bank = json.loads((output / "target_bank.json").read_text(encoding="utf-8"))
    bank["theta"][0][0] += 0.1
    (output / "target_bank.json").write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError, match="target bank"):
        execute(config_path=SMOKE, output=output, device_request="cpu", restart=True)


def test_direct_entry_points_resolve_outside_repository(tmp_path: Path) -> None:
    directory = ROOT / "experiments/analytic_target_matching/noisy_cohort"
    for filename in ("run.py", "plot.py"):
        result = subprocess.run(
            [sys.executable, str(directory / filename), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
