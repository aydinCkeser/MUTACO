#!/bin/bash
# macOS counterpart of run_mutaco.ps1. Use a separate mutaco-sbi environment.
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${MUTACO_SBI_PYTHON:-$repository_root/.venv/mutaco-sbi/bin/python}"
if [[ ! -x "$python" ]]; then
    printf 'mutaco-sbi interpreter is missing or not executable: %s\n' "$python" >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    printf 'at least one Python argument is required after the launcher boundary\n' >&2
    exit 1
fi

for relative in experiments/analytic_target_matching/src/mutaco_analytic/__init__.py experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json; do
    if [[ ! -f "$repository_root/$relative" ]]; then
        printf 'Required checkout path is unavailable: %s. Use a Git checkout with working symbolic links.\n' "$relative" >&2
        exit 1
    fi
done

export PYTHONPATH="$repository_root/src:$repository_root/experiments/analytic_target_matching/src:$repository_root/simulators/graph-simulator/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1

"$python" - "$repository_root" <<'PY'
import json
from pathlib import Path
import sys

repository = Path(sys.argv[1]).resolve()
if Path(sys.prefix).name.lower() != "mutaco-sbi":
    raise SystemExit(f"refusing interpreter prefix {sys.prefix!r}; expected environment 'mutaco-sbi'")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("the macOS qualification environment requires Python 3.11")
import torch
import sbi
import mutaco

module_path = Path(mutaco.__file__).resolve()
if repository not in module_path.parents:
    raise SystemExit(f"mutaco imported outside this checkout: {module_path}")
if str(torch.get_default_device()) != "cpu":
    raise SystemExit("the initial macOS qualification requires CPU execution")
print("MUTACO_RUNTIME=" + json.dumps({
    "environment": "mutaco-sbi", "python": sys.executable,
    "torch": torch.__version__, "sbi": sbi.__version__,
    "mutaco": str(module_path), "default_device": "cpu",
}, sort_keys=True))
PY

cd -- "$repository_root"
exec "$python" "$@"
