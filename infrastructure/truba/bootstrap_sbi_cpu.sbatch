#!/usr/bin/env bash
# Build/reuse a user-owned Linux CPU environment for Prompt 20J-Q/20J.
#SBATCH --nodes=1
#SBATCH --partition=barbun
#SBATCH --ntasks=1
# Barbun currently enforces a 20-CPU minimum allocation per node.
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --job-name=mutaco-sbi-env
#SBATCH --output=prompt20jq-env-%j.out
#SBATCH --error=prompt20jq-env-%j.err

set -euo pipefail

: "${MUTACO_QUALIFICATION_ROOT:?pin exact clean qualification checkout}"
: "${MUTACO_SBI_ENV_ROOT:?pin user-owned isolated environment path}"
: "${MUTACO_ENV_EVIDENCE:?pin environment evidence output}"

module load comp/python/miniconda3
mkdir -p "${MUTACO_ENV_EVIDENCE}"
cd "${MUTACO_QUALIFICATION_ROOT}"

if [[ ! -x "${MUTACO_SBI_ENV_ROOT}/bin/python" ]]; then
  conda create -y -p "${MUTACO_SBI_ENV_ROOT}" python=3.11 pip
fi

PYTHON="${MUTACO_SBI_ENV_ROOT}/bin/python"
"${PYTHON}" -m pip install --upgrade "pip<26" "setuptools>=77,<82" wheel
"${PYTHON}" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.11,<2.13"
"${PYTHON}" -m pip install -e ".[dev,plot-test]"
"${PYTHON}" -m pip install -e experiments/analytic_target_matching -e simulators/graph-simulator
"${PYTHON}" -m pip install -r infrastructure/environments/mutaco-sbi-requirements.txt
"${PYTHON}" -m pip check
"${PYTHON}" -m pip freeze > "${MUTACO_ENV_EVIDENCE}/pip-freeze.txt"
"${PYTHON}" - <<'PY' > "${MUTACO_ENV_EVIDENCE}/runtime.json"
import importlib.metadata, json, platform, sys, torch
names = ("sbi", "torch", "botorch", "gpytorch", "numpy", "scipy", "nflows", "skorch", "tensorboard", "zuko")
print(json.dumps({
    "python": platform.python_version(),
    "executable": sys.executable,
    "packages": {name: importlib.metadata.version(name) for name in names},
    "cuda_available": torch.cuda.is_available(),
}, indent=2))
PY
