from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "infrastructure/local/run_mutaco.sh"
PYTEST_RUNNER = ROOT / "infrastructure/local/run_pytest.sh"
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS launcher qualification")


def launch(script: Path, *args: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(script), *args], cwd=ROOT.parent, text=True, capture_output=True,
                          env={**os.environ, "MUTACO_SBI_PYTHON": sys.executable, **environment})


def test_python_arguments_cwd_and_local_imports_are_preserved() -> None:
    arguments = ['a "quoted" value', "path with spaces", "$(literal)"]
    result = launch(RUNNER, "-c", "import json,sys,os,mutaco; print(json.dumps([sys.argv[1:],os.getcwd(),mutaco.__file__]))", *arguments)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout.splitlines()[-1])
    assert values[0] == arguments
    assert Path(values[1]) == ROOT
    assert Path(values[2]).is_relative_to(ROOT / "src")


def test_command_failure_status_is_preserved() -> None:
    assert launch(RUNNER, "-c", "raise SystemExit(23)").returncode == 23


def test_missing_interpreter_is_rejected(tmp_path: Path) -> None:
    result = launch(RUNNER, "-c", "raise SystemExit(0)", MUTACO_SBI_PYTHON=str(tmp_path / "missing"))
    assert result.returncode != 0
    assert "interpreter is missing" in result.stderr


def test_wrong_environment_is_rejected(tmp_path: Path) -> None:
    # A symlink retains the real interpreter prefix; use an actual wrong prefix.
    candidate = "/usr/bin/python3"
    result = launch(RUNNER, "-c", "raise SystemExit(0)", MUTACO_SBI_PYTHON=candidate)
    assert result.returncode != 0
    assert "refusing interpreter prefix" in result.stderr


def test_pytest_temp_inside_repository_is_rejected_before_creation() -> None:
    temporary = ROOT / ".pytest_rejected_macos_launcher"
    result = launch(PYTEST_RUNNER, "--collect-only", MUTACO_PYTEST_TEMP_ROOT=str(temporary))
    assert result.returncode != 0
    assert "inside the Dropbox repository" in result.stderr
    assert not temporary.exists()
