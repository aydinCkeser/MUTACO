#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=barbun
#SBATCH --exclude=barbun70,barbun[112-119]
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=mutaco-primary-q1-targets

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec "${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.freeze_targets_v2 \
  --declaration "${MUTACO_Q1_TARGET_DECLARATION}" \
  --config "${MUTACO_Q1_TARGET_CONFIG}" \
  --output "${MUTACO_Q1_TARGET_OUTPUT_ROOT}" \
  --maximum-target-calls "${MUTACO_Q1_TARGET_MAXIMUM_CALLS}" \
  --torch-intraop-threads "${SLURM_CPUS_PER_TASK}" \
  --torch-interop-threads 1 --execute --confirm-approved
