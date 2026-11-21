"""Safe help, legacy fail-before-science and persisted-manifest validation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.graph_inversion import run_experiment as legacy


ROOT = Path(__file__).resolve().parents[2]
HELP_MODULES = (
    "experiments.graph_inversion.run_experiment",
    "experiments.graph_inversion.run_paper_v3_worker",
    "experiments.paper.analysis.evidence_analysis_cli",
    "experiments.paper.figures.plot_paper120_primary_boxplot_v1",
    "reproducibility.verify",
)
PROBE = r'''
from contextlib import ExitStack
import importlib.abc, runpy, subprocess, sys
from pathlib import Path
from unittest.mock import patch
from mutaco.surrogate import exact_multitask as gp
from mutaco_graph import OverdampedScientificSimulator
class NoSBI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == "sbi":
            raise ModuleNotFoundError("optional SBI forbidden for safe help", name=fullname)
sys.meta_path.insert(0, NoSBI())
def forbidden(*a, **k):
    raise AssertionError("GP, physical or external execution forbidden for help")
native_process = subprocess.Popen
def font_inventory_only(command, *args, **kwargs):
    executable = Path(command[0]).name if isinstance(command, (list, tuple)) else ""
    if executable not in {"fc-list", "fc-match"} and list(command) != ["system_profiler", "-xml", "SPFontsDataType"]:
        forbidden()
    return native_process(command, *args, **kwargs)
with ExitStack() as stack:
    for name in ("_build_model", "fit_exact_multitask_gp", "fit_exact_multitask_gp_with_recovery"):
        stack.enter_context(patch.object(gp, name, forbidden))
    for name in ("__post_init__", "evaluate", "evaluate_batch", "evaluate_condition_grid"):
        stack.enter_context(patch.object(OverdampedScientificSimulator, name, forbidden))
    stack.enter_context(patch.object(subprocess, "Popen", font_inventory_only))
    module = sys.argv[1]
    sys.argv = [module, "--help"]
    runpy.run_module(module, run_name="__main__")
'''


@pytest.mark.parametrize("module", HELP_MODULES)
def test_fresh_help_is_safe_without_optional_SBI(module, tmp_path) -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    result = subprocess.run([sys.executable, "-c", PROBE, module], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_legacy_execute_fails_before_loading_configuration_or_running_science(monkeypatch, tmp_path) -> None:
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append("forbidden")
        raise AssertionError("legacy command must not start science")
    monkeypatch.setattr(legacy, "load_campaign", forbidden)
    monkeypatch.setattr(legacy, "run_campaign", forbidden)
    with pytest.raises(ModuleNotFoundError) as caught:
        legacy.execute(config=tmp_path / "not-loaded.json")
    assert caught.value.name == "experiments.graph_inversion.plot_results"
    assert attempts == []


@pytest.mark.parametrize("mutation, message", (
    (None, None), ("configuration", "configuration mismatch"),
    ("inventory", "no artifact inventory"), ("missing", "artifact is missing"),
    ("size", "artifact size mismatch"), ("hash", "artifact hash mismatch"),
))
def test_persisted_manifest_validation_keeps_exact_checks(tmp_path, mutation, message) -> None:
    # Fixed stored bytes; no GP, simulator, report producer or figure renderer.
    data = b'{"fixed": [1, 2, 3]}\n'
    path = tmp_path / "result.json"
    path.write_bytes(data)
    config = {"fixed": "configuration"}
    manifest = {"configuration": config,
                "artifacts": {"result.json": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
    if mutation == "configuration": manifest["configuration"] = {}
    elif mutation == "inventory": manifest["artifacts"] = {}
    elif mutation == "missing": path.unlink()
    elif mutation == "size": path.write_bytes(data + b" ")
    elif mutation == "hash": path.write_bytes(data.replace(b"1", b"9"))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if message:
        with pytest.raises(ValueError, match=message):
            legacy._validated_complete_manifest(tmp_path, config)
    else:
        assert legacy._validated_complete_manifest(tmp_path, config) == manifest
