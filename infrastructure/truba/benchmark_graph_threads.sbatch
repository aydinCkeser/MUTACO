#!/usr/bin/env bash
#SBATCH --job-name=mutaco-p16-thr
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:20:00
#SBATCH --output=/arf/scratch/%u/mutaco/prompt16/logs/threads-%j.out
#SBATCH --error=/arf/scratch/%u/mutaco/prompt16/logs/threads-%j.err

set -euo pipefail
: "${MUTACO_EXPECTED_COMMIT:?set MUTACO_EXPECTED_COMMIT}"
ROOT="/arf/home/${USER}/mutaco/src"
PYTHON="/arf/home/${USER}/mutaco/envs/mutaco-cpu/bin/python"
OUT="/arf/scratch/${USER}/mutaco/prompt16/${MUTACO_EXPECTED_COMMIT}/threads"
mkdir -p "${OUT}" "/arf/scratch/${USER}/mutaco/prompt16/logs"
cd "${ROOT}"
test "$(git rev-parse HEAD)" = "${MUTACO_EXPECTED_COMMIT}"
test -z "$(git status --porcelain)"

for THREADS in 1 4 8 20 40; do
  OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
  OPENBLAS_NUM_THREADS="${THREADS}" "${PYTHON}" \
    -m infrastructure.truba.benchmark_graph_threads \
    --threads "${THREADS}" --repetitions 3 \
    --output "${OUT}/threads_${THREADS}.json"
done
"${PYTHON}" -m infrastructure.truba.summarize_thread_benchmark \
  --input "${OUT}" --output "${OUT}/summary.json"
find "${OUT}" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "${OUT}/SHA256SUMS"
