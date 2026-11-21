"""Repository-wide pytest guard against Dropbox-hosted temporary output."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest

from infrastructure.local.runtime_hygiene import (
    forbidden_repository_temp_directories,
    format_paths,
    validate_external_pytest_temp,
)


ROOT = Path(__file__).resolve().parent


def pytest_configure(config: pytest.Config) -> None:
    try:
        validate_external_pytest_temp(
            ROOT,
            basetemp=config.getoption("basetemp"),
            debug_temproot=os.environ.get("PYTEST_DEBUG_TEMPROOT"),
            system_temp=tempfile.gettempdir(),
        )
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    forbidden = forbidden_repository_temp_directories(ROOT)
    if not forbidden:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep(
            "=",
            "repository-local test output detected: " + format_paths(forbidden),
            red=True,
        )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
