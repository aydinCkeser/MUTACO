#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_PROCID:?Q1 workers require a Slurm rank}"
: "${MUTACO_PRIMARY_Q1_OUTPUT_ROOT:?pin Q1 output root}"
export MUTACO_PRIMARY_Q1_RANK="${SLURM_PROCID}"
export MUTACO_PRIMARY_Q1_OUTPUT="${MUTACO_PRIMARY_Q1_OUTPUT_ROOT}/worker-$(printf '%02d' "${SLURM_PROCID}")"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

exec "${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.run_primary_q1_worker \
  --declaration "${MUTACO_PRIMARY_Q1_DECLARATION}" \
  --targets-root "${MUTACO_PRIMARY_Q1_TARGET_RESPONSES}" \
  --config "${MUTACO_PRIMARY_Q1_CONFIG}" \
  --output "${MUTACO_PRIMARY_Q1_OUTPUT}" \
  --rank "${MUTACO_PRIMARY_Q1_RANK}" \
  --workers "${MUTACO_PRIMARY_Q1_WORKERS}" \
  --maximum-calls "${MUTACO_PRIMARY_Q1_MAXIMUM_CALLS}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${SLURM_CPUS_PER_TASK}" \
  --torch-interop-threads 1 --execute --confirm-approved
