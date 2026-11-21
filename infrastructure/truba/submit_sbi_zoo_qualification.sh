#!/usr/bin/env bash
# Validate by default. Physical qualification submission is a later explicit gate.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI Python}"
: "${MUTACO_SBI_ZOO_DECLARATION:?pin external immutable declaration}"
: "${MUTACO_SBI_ZOO_TARGET_RECORD:?pin accepted off-cohort target record}"
: "${MUTACO_SBI_ZOO_PAPER120_MANIFEST:?pin Paper-120 exclusion manifest}"
: "${MUTACO_SBI_ZOO_CONFIG:?pin graph configuration}"
: "${MUTACO_SBI_ZOO_OUTPUT_ROOT:?pin append-only output root}"
: "${MUTACO_SBI_ZOO_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_sbi_zoo_qualification.sh [--validate-only|--submit]" >&2
  exit 64
fi
(( MUTACO_SBI_ZOO_MAXIMUM_CALLS == 140 ))
(( MUTACO_SBI_ZOO_SBATCH_NODES == 1 ))
(( MUTACO_SBI_ZOO_SBATCH_NTASKS == 1 ))
(( MUTACO_SBI_ZOO_SBATCH_NTASKS_PER_NODE == 1 ))
(( MUTACO_SBI_ZOO_SBATCH_CPUS_PER_TASK == 20 ))

root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"

"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_SBI_ZOO_JOB_LOG_DIR}" \
  --output-root "${MUTACO_SBI_ZOO_OUTPUT_ROOT}"
[[ ! -e "${MUTACO_SBI_ZOO_OUTPUT_ROOT}" ]]
[[ -f "${MUTACO_SBI_ZOO_TARGET_RECORD}" ]]
[[ -f "${MUTACO_SBI_ZOO_TARGET_RECORD%.json}.sha256" ]]
[[ -f "${MUTACO_SBI_ZOO_PAPER120_MANIFEST}" ]]
[[ -f "${MUTACO_SBI_ZOO_CONFIG}" ]]

if [[ "${mode}" == "--validate-only" ]]; then
  "${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_qualification preflight \
    --target-record "${MUTACO_SBI_ZOO_TARGET_RECORD}" \
    --paper120-manifest "${MUTACO_SBI_ZOO_PAPER120_MANIFEST}" \
    --config "${MUTACO_SBI_ZOO_CONFIG}" \
    --output-root "${MUTACO_SBI_ZOO_OUTPUT_ROOT}" \
    --execution-commit "${MUTACO_EXPECTED_COMMIT}" \
    --execution-tree "$(git rev-parse HEAD^{tree})"
  [[ ! -e "${MUTACO_SBI_ZOO_OUTPUT_ROOT}" ]]
  exit 0
fi

[[ "${MUTACO_SBI_ZOO_EXECUTION_AUTHORIZED:-0}" == "1" ]]
[[ -f "${MUTACO_SBI_ZOO_DECLARATION}" ]]
[[ -f "${MUTACO_SBI_ZOO_DECLARATION%.json}.sha256" ]]
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_qualification preview \
  --declaration "${MUTACO_SBI_ZOO_DECLARATION}"
mkdir -p "${MUTACO_SBI_ZOO_JOB_LOG_DIR}"
exec sbatch --parsable --no-requeue --exclusive \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=20 \
  --exclude="${MUTACO_SBI_ZOO_SBATCH_EXCLUDE}" --chdir="${root}" \
  --output="${MUTACO_SBI_ZOO_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_SBI_ZOO_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/sbi_zoo_qualification.sbatch"
