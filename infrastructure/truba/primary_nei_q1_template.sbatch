#!/usr/bin/env bash
#SBATCH --nodes=5
#SBATCH --partition=barbun
#SBATCH --exclude=barbun70,barbun[112-119]
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=mutaco-primary-nei-q1

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
srun --ntasks="${MUTACO_PRIMARY_Q1_WORKERS}" \
  --cpus-per-task="${MUTACO_PRIMARY_Q1_SBATCH_CPUS_PER_TASK}" \
  --cpu-bind=cores --kill-on-bad-exit=0 \
  bash infrastructure/truba/run_primary_nei_q1_worker.sh

"${MUTACO_V3_PYTHON}" -u -m infrastructure.truba.collect_primary_benchmark q1 \
  --declaration "${MUTACO_PRIMARY_Q1_DECLARATION}" \
  --workers-root "${MUTACO_PRIMARY_Q1_OUTPUT_ROOT}" \
  --output "${MUTACO_PRIMARY_Q1_OUTPUT_ROOT}/collection" \
  --workers "${MUTACO_PRIMARY_Q1_WORKERS}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}"
