#!/usr/bin/env bash
#SBATCH --job-name=mutaco-p16-env
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/arf/scratch/%u/mutaco/prompt16/logs/bootstrap-%j.out
#SBATCH --error=/arf/scratch/%u/mutaco/prompt16/logs/bootstrap-%j.err

set -euo pipefail
: "${MUTACO_EXPECTED_COMMIT:?set MUTACO_EXPECTED_COMMIT}"

ROOT="/arf/home/${USER}/mutaco/src"
ENV_ROOT="/arf/home/${USER}/mutaco/envs/mutaco-cpu"
OUT="/arf/scratch/${USER}/mutaco/prompt16/${MUTACO_EXPECTED_COMMIT}/environment"
mkdir -p "${OUT}" "/arf/scratch/${USER}/mutaco/prompt16/logs" "$(dirname "${ENV_ROOT}")"

cd "${ROOT}"
CURRENT="$(git rev-parse HEAD)"
test "${CURRENT}" = "${MUTACO_EXPECTED_COMMIT}"
test -z "$(git status --porcelain)"

source /etc/profile.d/modules.sh
module load comp/python/miniconda3

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  conda create -y -p "${ENV_ROOT}" python=3.11 pip
fi

"${ENV_ROOT}/bin/python" -m pip install --upgrade "pip<26" "setuptools>=77,<82" wheel
"${ENV_ROOT}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.11,<2.13"
"${ENV_ROOT}/bin/python" -m pip install -e ".[dev,plot-test]"
"${ENV_ROOT}/bin/python" -m pip install -e experiments/analytic_target_matching -e simulators/graph-simulator

"${ENV_ROOT}/bin/python" -m pip check
"${ENV_ROOT}/bin/python" -m pytest \
  tests/analytic/test_handwritten.py \
  tests/runtime/test_parallelism.py \
  tests/contracts/test_simulator_contract.py -q > "${OUT}/bounded-tests.txt"
"${ENV_ROOT}/bin/python" -m pip freeze > "${OUT}/pip-freeze.txt"
"${ENV_ROOT}/bin/python" - <<'PY' > "${OUT}/runtime.txt"
import json, os, platform, sys, torch
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "torch_intraop_threads": torch.get_num_threads(),
    "torch_interop_threads": torch.get_num_interop_threads(),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
}, indent=2, sort_keys=True))
PY
lscpu > "${OUT}/lscpu.txt"
env | sort | grep '^SLURM_' > "${OUT}/slurm-environment.txt"
sha256sum "${OUT}"/* > "${OUT}/SHA256SUMS"
