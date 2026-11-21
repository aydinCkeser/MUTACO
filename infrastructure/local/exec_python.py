"""Execute exact Python arguments serialized by the PowerShell launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys


_ARGUMENT_ENV = "MUTACO_LAUNCH_ARGUMENTS_JSON"


def launch_arguments() -> tuple[str, ...]:
    raw = os.environ.pop(_ARGUMENT_ENV, None)
    if raw is None:
        raise RuntimeError(f"{_ARGUMENT_ENV} is required")
    value = json.loads(raw)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise RuntimeError("serialized Python arguments must be a nonempty string list")
    return tuple(value)


def main() -> None:
    arguments = launch_arguments()
    completed = subprocess.run([sys.executable, *arguments], check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
