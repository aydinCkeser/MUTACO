from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from infrastructure.local.runtime_hygiene import (
    forbidden_repository_temp_directories,
    validate_external_pytest_temp,
)
from infrastructure.local.exec_python import launch_arguments


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("source", ("basetemp", "debug", "system"))
def test_pytest_hygiene_rejects_every_repository_local_temp_source(
    tmp_path: Path, source: str
) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir()
    values = {
        "basetemp": None,
        "debug_temproot": None,
        "system_temp": tmp_path / "system-temp",
    }
    if source == "basetemp":
        values["basetemp"] = ".pytest_agent"
    elif source == "debug":
        values["debug_temproot"] = repository / ".pytest_agent"
    else:
        values["system_temp"] = repository / "Temp"
    with pytest.raises(ValueError, match="inside the Dropbox repository"):
        validate_external_pytest_temp(repository, **values)


def test_pytest_hygiene_accepts_external_temp_roots(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    external = tmp_path / "external"
    repository.mkdir()
    assert validate_external_pytest_temp(
        repository,
        basetemp=None,
        debug_temproot=external / "debug",
        system_temp=external / "system",
    ) == ((external / "debug").resolve(), (external / "system").resolve())


def test_repository_root_contains_no_transient_test_directories() -> None:
    assert forbidden_repository_temp_directories(ROOT) == ()


def test_launchers_pin_sbi_and_external_pytest_temp() -> None:
    runner = (ROOT / "infrastructure/local/run_mutaco.ps1").read_text(encoding="utf-8")
    pytest_runner = (ROOT / "infrastructure/local/run_pytest.ps1").read_text(encoding="utf-8")
    assert "mutaco-sbi" in runner
    assert "import sbi" in runner
    assert "refusing interpreter prefix" in runner
    assert "PYTEST_DEBUG_TEMPROOT" in pytest_runner
    assert "LocalApplicationData" in pytest_runner
    assert "--basetemp" not in pytest_runner
    assert "PYTHONUTF8" in runner
    assert "MUTACO_LAUNCH_ARGUMENTS_JSON" in runner
    assert "-p no:benchmark" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_exact_python_argument_trampoline_preserves_quotes_and_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = ["-c", 'print("quoted value")', "path with spaces"]
    monkeypatch.setenv("MUTACO_LAUNCH_ARGUMENTS_JSON", json.dumps(arguments))
    assert launch_arguments() == tuple(arguments)
    assert "MUTACO_LAUNCH_ARGUMENTS_JSON" not in os.environ
