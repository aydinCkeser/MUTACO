#!/usr/bin/env bash
# Validate by default. The sole permitted execution is a zero-simulator compute smoke.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI 0.26.1 Python}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"
: "${MUTACO_PRIMARY_Q1_SMOKE_OUTPUT_ROOT:?pin new append-only smoke output}"
: "${MUTACO_PRIMARY_Q1_SMOKE_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_PRIMARY_Q1_SMOKE_DECLARATION:?pin retry declaration}"
: "${MUTACO_PRIMARY_Q1_SMOKE_CONFIG:?pin graph GP configuration}"

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_primary_nei_q1_compute_smoke.sh [--validate-only|--submit]" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_Q1_SMOKE_SBATCH_NODES != 1 || MUTACO_PRIMARY_Q1_SMOKE_SBATCH_NTASKS != 1 || MUTACO_PRIMARY_Q1_SMOKE_SBATCH_NTASKS_PER_NODE != 1 )); then
  echo "Q1 compute smoke requires exactly one node and one task" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_Q1_SMOKE_SBATCH_CPUS_PER_TASK != 20 )); then
  echo "Q1 compute smoke must reserve Barbun's 20 cores" >&2
  exit 64
fi
root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" \
  --log-root "${MUTACO_PRIMARY_Q1_SMOKE_JOB_LOG_DIR}" \
  --output-root "${MUTACO_PRIMARY_Q1_SMOKE_OUTPUT_ROOT}"
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.run_primary_q1_compute_smoke \
  --declaration "${MUTACO_PRIMARY_Q1_SMOKE_DECLARATION}" \
  --config "${MUTACO_PRIMARY_Q1_SMOKE_CONFIG}" \
  --output "${MUTACO_PRIMARY_Q1_SMOKE_OUTPUT_ROOT}/preview-not-created" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${MUTACO_PRIMARY_Q1_SMOKE_SBATCH_CPUS_PER_TASK}" \
  --torch-interop-threads 1
[[ ! -e "${MUTACO_PRIMARY_Q1_SMOKE_OUTPUT_ROOT}" ]]
mkdir -p "${MUTACO_PRIMARY_Q1_SMOKE_JOB_LOG_DIR}"
if [[ "${mode}" == "--validate-only" ]]; then
  exit 0
fi
if [[ "${MUTACO_PRIMARY_Q1_COMPUTE_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
  echo "Q1 compute-smoke submission requires separate authorization" >&2
  exit 64
fi
exec sbatch --parsable \
  --nodes="${MUTACO_PRIMARY_Q1_SMOKE_SBATCH_NODES}" \
  --ntasks="${MUTACO_PRIMARY_Q1_SMOKE_SBATCH_NTASKS}" \
  --ntasks-per-node="${MUTACO_PRIMARY_Q1_SMOKE_SBATCH_NTASKS_PER_NODE}" \
  --cpus-per-task="${MUTACO_PRIMARY_Q1_SMOKE_SBATCH_CPUS_PER_TASK}" \
  --exclude="${MUTACO_PRIMARY_Q1_SMOKE_SBATCH_EXCLUDE}" \
  --chdir="${root}" \
  --output="${MUTACO_PRIMARY_Q1_SMOKE_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_PRIMARY_Q1_SMOKE_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/primary_nei_q1_compute_smoke.sbatch"
