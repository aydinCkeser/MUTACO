#!/usr/bin/env bash
# Validation is default; --submit requires a separate explicit five-call authorization.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI Python}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"
: "${MUTACO_Q1_TARGET_DECLARATION:?pin Q1 theta declaration}"
: "${MUTACO_Q1_TARGET_CONFIG:?pin target simulator configuration}"
: "${MUTACO_Q1_TARGET_OUTPUT_ROOT:?pin new append-only target output}"
: "${MUTACO_Q1_TARGET_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_Q1_TARGET_MAXIMUM_CALLS:?pin exact target call cap}"

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_primary_nei_q1_target_construction.sh [--validate-only|--submit]" >&2
  exit 64
fi
[[ "${MUTACO_Q1_TARGET_MAXIMUM_CALLS}" == "5" ]]
root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_Q1_TARGET_JOB_LOG_DIR}" \
  --output-root "${MUTACO_Q1_TARGET_OUTPUT_ROOT}"
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.freeze_targets_v2 \
  --declaration "${MUTACO_Q1_TARGET_DECLARATION}" \
  --config "${MUTACO_Q1_TARGET_CONFIG}" \
  --output "${MUTACO_Q1_TARGET_OUTPUT_ROOT}" \
  --maximum-target-calls 5
[[ ! -e "${MUTACO_Q1_TARGET_OUTPUT_ROOT}" ]]
mkdir -p "${MUTACO_Q1_TARGET_JOB_LOG_DIR}"
if [[ "${mode}" == "--validate-only" ]]; then
  exit 0
fi
if [[ "${MUTACO_Q1_TARGET_CONSTRUCTION_AUTHORIZED:-0}" != "1" ]]; then
  echo "Q1 target construction requires separate explicit five-call authorization" >&2
  exit 64
fi
exec sbatch --parsable --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${MUTACO_Q1_TARGET_SBATCH_CPUS_PER_TASK}" \
  --exclude="barbun70,barbun[112-119]" --chdir="${root}" \
  --output="${MUTACO_Q1_TARGET_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_Q1_TARGET_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/primary_nei_q1_target_construction.sbatch"
