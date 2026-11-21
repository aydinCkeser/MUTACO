#!/usr/bin/env bash
# Validate by default. Submission requires a separate exact three-call authorization.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI Python}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"
: "${MUTACO_PRIMARY_RESCUE_OUTPUT_ROOT:?pin new append-only rescue output}"
: "${MUTACO_PRIMARY_RESCUE_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_PRIMARY_RESCUE_DECLARATION:?pin rescue declaration}"
: "${MUTACO_PRIMARY_RESCUE_PRIMARY_DECLARATION:?pin original primary declaration}"
: "${MUTACO_PRIMARY_RESCUE_DECISION_FREEZE:?pin immutable failed-attempt freeze}"
: "${MUTACO_PRIMARY_RESCUE_TARGET_RESPONSES:?pin frozen target ledger}"
: "${MUTACO_PRIMARY_RESCUE_CONFIG:?pin graph configuration}"

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_primary_assessment_rescue.sh [--validate-only|--submit]" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_RESCUE_MAXIMUM_CALLS != 3 )); then
  echo "primary assessment rescue requires an exact three-call cap" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_RESCUE_SBATCH_NODES != 1 || MUTACO_PRIMARY_RESCUE_SBATCH_NTASKS != 1 || MUTACO_PRIMARY_RESCUE_SBATCH_NTASKS_PER_NODE != 1 )); then
  echo "primary assessment rescue requires exactly one node and one task" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_RESCUE_SBATCH_CPUS_PER_TASK != 20 )); then
  echo "primary assessment rescue must reserve Barbun's 20 cores" >&2
  exit 64
fi
root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_PRIMARY_RESCUE_JOB_LOG_DIR}" \
  --output-root "${MUTACO_PRIMARY_RESCUE_OUTPUT_ROOT}"
[[ ! -e "${MUTACO_PRIMARY_RESCUE_OUTPUT_ROOT}" ]]
mkdir -p "${MUTACO_PRIMARY_RESCUE_JOB_LOG_DIR}"
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.run_primary_assessment_rescue \
  --rescue-declaration "${MUTACO_PRIMARY_RESCUE_DECLARATION}" \
  --primary-declaration "${MUTACO_PRIMARY_RESCUE_PRIMARY_DECLARATION}" \
  --decision-freeze "${MUTACO_PRIMARY_RESCUE_DECISION_FREEZE}" \
  --targets-root "${MUTACO_PRIMARY_RESCUE_TARGET_RESPONSES}" \
  --config "${MUTACO_PRIMARY_RESCUE_CONFIG}" \
  --output "${MUTACO_PRIMARY_RESCUE_OUTPUT_ROOT}/preview-not-created" \
  --maximum-calls 3 --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${MUTACO_PRIMARY_RESCUE_SBATCH_CPUS_PER_TASK}" \
  --torch-interop-threads 1
[[ ! -e "${MUTACO_PRIMARY_RESCUE_OUTPUT_ROOT}" ]]
if [[ "${mode}" == "--validate-only" ]]; then
  exit 0
fi
if [[ "${MUTACO_PRIMARY_RESCUE_EXECUTION_AUTHORIZED:-0}" != "1" ]]; then
  echo "primary assessment rescue submission requires separate authorization" >&2
  exit 64
fi
exec sbatch --parsable \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${MUTACO_PRIMARY_RESCUE_SBATCH_CPUS_PER_TASK}" \
  --exclude="${MUTACO_PRIMARY_RESCUE_SBATCH_EXCLUDE}" \
  --chdir="${root}" \
  --output="${MUTACO_PRIMARY_RESCUE_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_PRIMARY_RESCUE_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/primary_assessment_rescue.sbatch"
