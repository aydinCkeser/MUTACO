#!/bin/bash
# Keep all pytest temporary data outside the source checkout.
set -euo pipefail

script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTEST_DEBUG_TEMPROOT="${MUTACO_PYTEST_TEMP_ROOT:-${TMPDIR:-/tmp}/mutaco-pytest}"
# Validate before creating anything; conftest.py also checks --basetemp.
exec "$script_root/run_mutaco.sh" -c '
import os
from pathlib import Path
import sys
import tempfile
from infrastructure.local.runtime_hygiene import validate_external_pytest_temp

root = Path(sys.argv[1]).resolve()
temporary = Path(os.environ["PYTEST_DEBUG_TEMPROOT"]).resolve()
validate_external_pytest_temp(root, basetemp=None, debug_temproot=temporary,
                             system_temp=tempfile.gettempdir())
temporary.mkdir(parents=True, exist_ok=True)
os.execv(sys.executable, [sys.executable, "-m", "pytest", *sys.argv[2:]])
' "$script_root/../.." "$@"
