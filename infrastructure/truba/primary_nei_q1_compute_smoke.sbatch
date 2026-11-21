#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=barbun
#SBATCH --exclude=barbun70,barbun[112-119]
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=mutaco-primary-q1-gp-smoke

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

exec "${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.run_primary_q1_compute_smoke \
  --declaration "${MUTACO_PRIMARY_Q1_SMOKE_DECLARATION}" \
  --config "${MUTACO_PRIMARY_Q1_SMOKE_CONFIG}" \
  --output "${MUTACO_PRIMARY_Q1_SMOKE_OUTPUT_ROOT}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${SLURM_CPUS_PER_TASK}" \
  --torch-interop-threads 1 --execute --confirm-approved
