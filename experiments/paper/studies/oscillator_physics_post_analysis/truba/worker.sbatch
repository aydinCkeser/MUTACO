#!/usr/bin/env bash
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --job-name=mutaco-oscillator-physics-t098

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
[[ "${MUTACO_OSCILLATOR_EIGHT_CALLS_AUTHORIZED:-0}" == "1" ]]
export OMP_NUM_THREADS="${MUTACO_OSCILLATOR_RUNTIME_THREADS}"
export MKL_NUM_THREADS="${MUTACO_OSCILLATOR_RUNTIME_THREADS}"
export OPENBLAS_NUM_THREADS="${MUTACO_OSCILLATOR_RUNTIME_THREADS}"
exec "${MUTACO_V3_PYTHON}" -u -m experiments.paper.oscillator_physics_post_analysis.analysis \
  execute-worker --output-root "${MUTACO_OSCILLATOR_OUTPUT_ROOT}" \
  --authorization AUTHORIZE_EXACTLY_EIGHT_OSCILLATOR_SENSITIVITY_CALLS
