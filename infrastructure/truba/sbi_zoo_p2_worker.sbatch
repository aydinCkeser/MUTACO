#!/usr/bin/env bash
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --time=01:00:00
#SBATCH --job-name=mutaco-sbi-zoo-p2

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
: "${MUTACO_SBI_ZOO_P2_RUNTIME_THREADS_PER_WORKER:?pin worker runtime threads}"
[[ "${MUTACO_SBI_ZOO_P2_RUNTIME_THREADS_PER_WORKER}" == "20" ]]
export OMP_NUM_THREADS="${MUTACO_SBI_ZOO_P2_RUNTIME_THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${MUTACO_SBI_ZOO_P2_RUNTIME_THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${MUTACO_SBI_ZOO_P2_RUNTIME_THREADS_PER_WORKER}"
exec "${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.sbi_zoo_p2 physical-task \
  --declaration "${MUTACO_SBI_ZOO_P2_DECLARATION}" \
  --physical-task-index "${SLURM_ARRAY_TASK_ID}" --execute
