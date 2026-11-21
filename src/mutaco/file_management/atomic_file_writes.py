"""Small atomic-write primitives hardened against transient Windows locks."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import time
from typing import Callable
from uuid import uuid4


_WINDOWS = os.name == "nt"
_WINDOWS_SHARING_ERRORS = {5, 32, 33}
_REPLACE_ATTEMPTS = 9
_INITIAL_RETRY_SECONDS = 0.025
_MAX_RETRY_SECONDS = 0.8


def _retryable_windows_lock(error: OSError) -> bool:
    return _WINDOWS and (
        isinstance(error, PermissionError)
        or getattr(error, "winerror", None) in _WINDOWS_SHARING_ERRORS
        or error.errno in {errno.EACCES, errno.EPERM}
    )


def _with_windows_lock_retry(operation: Callable[[], None]) -> None:
    delay = _INITIAL_RETRY_SECONDS
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            operation()
            return
        except OSError as error:
            if not _retryable_windows_lock(error) or attempt + 1 == _REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, _MAX_RETRY_SECONDS)


def _replace(source: Path, destination: Path) -> None:
    _with_windows_lock_retry(lambda: os.replace(source, destination))


def _unlink(path: Path) -> None:
    if not path.exists():
        return
    _with_windows_lock_retry(lambda: path.unlink(missing_ok=True))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically write bytes, retrying only transient Windows sharing locks."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    primary_error: BaseException | None = None
    try:
        temporary.write_bytes(payload)
        _replace(temporary, destination)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _unlink(temporary)
        except OSError:
            if primary_error is None:
                raise


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically write text through the same byte-exact persistence path."""

    atomic_write_bytes(Path(path), text.encode(encoding))


# Trusted historical pickles refer to this original Python module name.
for _legacy_export in (
    _retryable_windows_lock,
    _with_windows_lock_retry,
    _replace,
    _unlink,
    atomic_write_bytes,
    atomic_write_text,
):
    _legacy_export.__module__ = "mutaco.atomic_io"
del _legacy_export
