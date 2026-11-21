#!/usr/bin/env bash
# Unified Paper-v3 execution surface. Submission remains separately approval-gated.
#SBATCH --nodes=8
#SBATCH --partition=barbun
#SBATCH --exclude=barbun[112-119]
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=mutaco-paper-v3

set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean Paper-v3 checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI 0.26.1 Python}"
: "${MUTACO_V3_OUTPUT_ROOT:?pin new append-only worker root}"
: "${MUTACO_EXPECTED_COMMIT:?pin Paper-120 source commit}"
: "${MUTACO_V3_DECLARATION:?pin the approved population declaration}"
: "${MUTACO_V3_METHOD_SPECS:?pin unified method specs}"
: "${MUTACO_V3_WORKERS:?pin the worker count}"
: "${MUTACO_V3_MAXIMUM_CALLS:?pin the exact per-worker call cap}"
: "${MUTACO_V3_SBATCH_CPUS_PER_TASK:?pin CPUs per worker task}"

cd "${MUTACO_PAPER120_ROOT}"

set +e
srun --ntasks="${MUTACO_V3_WORKERS}" --cpus-per-task="${MUTACO_V3_SBATCH_CPUS_PER_TASK}" --cpu-bind=cores --kill-on-bad-exit=0 \
  bash infrastructure/truba/run_paper120_v3_worker.sh
worker_status=$?
set -e

set +e
"${MUTACO_V3_PYTHON}" -u -m infrastructure.truba.collect_paper120_v3 \
  --root "${MUTACO_PAPER120_ROOT}" \
  --declaration "${MUTACO_V3_DECLARATION}" \
  --workers-root "${MUTACO_V3_OUTPUT_ROOT}" \
  --output "${MUTACO_V3_OUTPUT_ROOT}/collection" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --workers "${MUTACO_V3_WORKERS}"
collection_status=$?
set -e

if [[ ${worker_status} -ne 0 || ${collection_status} -ne 0 ]]; then
  exit 1
fi

# Deterministic compact evidence only: no model tensors or posterior samples.
"${MUTACO_V3_PYTHON}" -u -m experiments.paper.workflow.cli ingest export \
  "${MUTACO_V3_OUTPUT_ROOT}" \
  --output-prefix "${MUTACO_V3_OUTPUT_ROOT}/compact-export/paper-v3-${SLURM_JOB_ID:-local}" \
  --confirm-authorized
