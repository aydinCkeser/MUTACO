#!/usr/bin/env bash
# Validate by default. Submission remains blocked until target construction and approval.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI 0.26.1 Python}"
: "${MUTACO_PRIMARY_Q1_PREPARATION:?pin Q1 preparation}"
: "${MUTACO_PRIMARY_Q1_WORKERS:?pin Q1 workers}"
: "${MUTACO_PRIMARY_Q1_MAXIMUM_CALLS:?pin Q1 worker cap}"
: "${MUTACO_PRIMARY_Q1_OUTPUT_ROOT:?pin new append-only output root}"
: "${MUTACO_PRIMARY_Q1_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_primary_nei_q1.sh [--validate-only|--submit]" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_Q1_SBATCH_NTASKS != MUTACO_PRIMARY_Q1_SBATCH_NODES * MUTACO_PRIMARY_Q1_SBATCH_NTASKS_PER_NODE )); then
  echo "Q1 scheduler allocation must fill every requested node" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_Q1_SBATCH_NTASKS_PER_NODE * MUTACO_PRIMARY_Q1_SBATCH_CPUS_PER_TASK != 20 )); then
  echo "Q1 Barbun allocation must request exactly 20 cores per node" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_Q1_SBATCH_NTASKS != MUTACO_PRIMARY_Q1_WORKERS )); then
  echo "Q1 scheduler tasks must equal scientific workers" >&2
  exit 64
fi
root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_PRIMARY_Q1_JOB_LOG_DIR}" \
  --output-root "${MUTACO_PRIMARY_Q1_OUTPUT_ROOT}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.primary_q1 \
  --preparation "${MUTACO_PRIMARY_Q1_PREPARATION}" \
  --workers "${MUTACO_PRIMARY_Q1_WORKERS}" \
  --maximum-calls "${MUTACO_PRIMARY_Q1_MAXIMUM_CALLS}"
[[ ! -e "${MUTACO_PRIMARY_Q1_OUTPUT_ROOT}" ]]
mkdir -p "${MUTACO_PRIMARY_Q1_JOB_LOG_DIR}"
if [[ -z "${MUTACO_PRIMARY_Q1_TARGET_RESPONSES:-}" || ! -f "${MUTACO_PRIMARY_Q1_TARGET_RESPONSES}" ]]; then
  if [[ ! -d "${MUTACO_PRIMARY_Q1_TARGET_RESPONSES:-}" ]]; then
    echo "Q1 submission requires the separately constructed frozen target responses" >&2
    exit 64
  fi
fi
if [[ -z "${MUTACO_PRIMARY_Q1_DECLARATION:-}" || ! -f "${MUTACO_PRIMARY_Q1_DECLARATION}" ]]; then
  echo "Q1 submission requires the post-construction canonical declaration" >&2
  exit 64
fi
if [[ -z "${MUTACO_PRIMARY_Q1_CONFIG:-}" || ! -f "${MUTACO_PRIMARY_Q1_CONFIG}" ]]; then
  echo "Q1 submission requires the frozen graph configuration" >&2
  exit 64
fi
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.run_primary_q1_worker \
  --declaration "${MUTACO_PRIMARY_Q1_DECLARATION}" \
  --targets-root "${MUTACO_PRIMARY_Q1_TARGET_RESPONSES}" \
  --config "${MUTACO_PRIMARY_Q1_CONFIG}" \
  --output "${MUTACO_PRIMARY_Q1_OUTPUT_ROOT}/preview-not-created" \
  --rank 0 --workers "${MUTACO_PRIMARY_Q1_WORKERS}" \
  --maximum-calls "${MUTACO_PRIMARY_Q1_MAXIMUM_CALLS}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${MUTACO_PRIMARY_Q1_SBATCH_CPUS_PER_TASK}" \
  --torch-interop-threads 1
[[ ! -e "${MUTACO_PRIMARY_Q1_OUTPUT_ROOT}" ]]
if [[ "${mode}" == "--validate-only" ]]; then
  exit 0
fi
if [[ "${MUTACO_PRIMARY_Q1_EXECUTION_AUTHORIZED:-0}" != "1" ]]; then
  echo "Q1 submission requires a separately approved execution authorization" >&2
  exit 64
fi
exec sbatch --parsable \
  --nodes="${MUTACO_PRIMARY_Q1_SBATCH_NODES}" \
  --ntasks="${MUTACO_PRIMARY_Q1_SBATCH_NTASKS}" \
  --ntasks-per-node="${MUTACO_PRIMARY_Q1_SBATCH_NTASKS_PER_NODE}" \
  --cpus-per-task="${MUTACO_PRIMARY_Q1_SBATCH_CPUS_PER_TASK}" \
  --exclude="${MUTACO_PRIMARY_Q1_SBATCH_EXCLUDE}" \
  --chdir="${root}" \
  --output="${MUTACO_PRIMARY_Q1_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_PRIMARY_Q1_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/primary_nei_q1_template.sbatch"
