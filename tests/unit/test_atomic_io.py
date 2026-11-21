from __future__ import annotations

import errno
import os

import pytest

from mutaco import atomic_io


def test_atomic_write_retries_transient_windows_lock_without_recomputing_payload(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.json"
    real_replace = os.replace
    attempts = 0
    sleeps: list[float] = []

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(errno.EACCES, "simulated Dropbox lock", target)
        real_replace(source, target)

    monkeypatch.setattr(atomic_io, "_WINDOWS", True)
    monkeypatch.setattr(atomic_io.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", sleeps.append)
    atomic_io.atomic_write_bytes(destination, b"one computation\n")

    assert destination.read_bytes() == b"one computation\n"
    assert attempts == 3
    assert sleeps == [0.025, 0.05]
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_atomic_write_does_not_retry_nonlocking_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.json"
    attempts = 0

    def fail_replace(source, target):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.ENOSPC, "disk full", target)

    monkeypatch.setattr(atomic_io, "_WINDOWS", True)
    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_io.atomic_write_bytes(destination, b"payload")
    assert attempts == 1
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))
