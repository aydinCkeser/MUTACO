#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_PROCID:?assessment workers require a Slurm rank}"
output="${MUTACO_REPAIRED_SBI_OUTPUT_ROOT}/worker-$(printf '%03d' "${SLURM_PROCID}")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

exec "${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.run_repaired_sbi_assessment_worker \
  --tasks "${MUTACO_REPAIRED_SBI_TASKS}" \
  --targets-root "${MUTACO_REPAIRED_SBI_TARGETS}" \
  --config "${MUTACO_REPAIRED_SBI_CONFIG}" \
  --output "${output}" \
  --rank "${SLURM_PROCID}" --workers "${MUTACO_REPAIRED_SBI_WORKERS}" \
  --maximum-calls "${MUTACO_REPAIRED_SBI_MAXIMUM_CALLS_PER_WORKER}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${SLURM_CPUS_PER_TASK}" --torch-interop-threads 1 \
  --execute --confirm-approved
