#!/usr/bin/env bash
# Validate by default. Paper-120 P1 submission requires a later explicit gate.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI Python}"
: "${MUTACO_SBI_ZOO_P1_DECLARATION:?pin external immutable declaration}"
: "${MUTACO_SBI_ZOO_P1_COHORT_ROOT:?pin Paper-120 cohort}"
: "${MUTACO_SBI_ZOO_P1_CONFIG:?pin graph configuration}"
: "${MUTACO_SBI_ZOO_P1_OUTPUT_ROOT:?pin append-only output root}"
: "${MUTACO_SBI_ZOO_P1_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_sbi_zoo_p1.sh [--validate-only|--submit]" >&2
  exit 64
fi
(( MUTACO_SBI_ZOO_P1_TARGETS == 120 ))
(( MUTACO_SBI_ZOO_P1_TOTAL_CALLS == 16800 ))
[[ "${MUTACO_SBI_ZOO_P1_ARRAY}" == "0-59" ]]
(( MUTACO_SBI_ZOO_P1_CONCURRENCY == 20 ))
(( MUTACO_SBI_ZOO_P1_PHYSICAL_TASKS == 60 ))
(( MUTACO_SBI_ZOO_P1_LOGICAL_TARGETS_PER_TASK == 2 ))
(( MUTACO_SBI_ZOO_P1_PHYSICAL_TASKS * MUTACO_SBI_ZOO_P1_LOGICAL_TARGETS_PER_TASK == MUTACO_SBI_ZOO_P1_TARGETS ))
(( MUTACO_SBI_ZOO_P1_RUNTIME_THREADS_PER_WORKER == 20 ))
(( MUTACO_SBI_ZOO_P1_CPUS_PER_TASK == 20 ))
[[ "${MUTACO_SBI_ZOO_P1_MEMORY}" == "64G" ]]
[[ "${MUTACO_SBI_ZOO_P1_WALLTIME}" == "00:30:00" ]]
(( MUTACO_SBI_ZOO_P1_COLLECTOR_CPUS_PER_TASK == 20 ))
[[ "${MUTACO_SBI_ZOO_P1_COLLECTOR_MEMORY}" == "64G" ]]
[[ "${MUTACO_SBI_ZOO_P1_COLLECTOR_WALLTIME}" == "00:15:00" ]]

root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"

"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_SBI_ZOO_P1_JOB_LOG_DIR}" \
  --output-root "${MUTACO_SBI_ZOO_P1_OUTPUT_ROOT}"
[[ ! -e "${MUTACO_SBI_ZOO_P1_OUTPUT_ROOT}" ]]
[[ -d "${MUTACO_SBI_ZOO_P1_COHORT_ROOT}" ]]
[[ -f "${MUTACO_SBI_ZOO_P1_CONFIG}" ]]

if [[ "${mode}" == "--validate-only" ]]; then
  "${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p1 preflight \
    --cohort-root "${MUTACO_SBI_ZOO_P1_COHORT_ROOT}" \
    --config "${MUTACO_SBI_ZOO_P1_CONFIG}" \
    --output-root "${MUTACO_SBI_ZOO_P1_OUTPUT_ROOT}" \
    --execution-commit "${MUTACO_EXPECTED_COMMIT}" \
    --execution-tree "$(git rev-parse HEAD^{tree})"
  [[ ! -e "${MUTACO_SBI_ZOO_P1_OUTPUT_ROOT}" ]]
  exit 0
fi

[[ "${MUTACO_SBI_ZOO_P1_EXECUTION_AUTHORIZED:-0}" == "1" ]]
[[ -f "${MUTACO_SBI_ZOO_P1_DECLARATION}" ]]
[[ -f "${MUTACO_SBI_ZOO_P1_DECLARATION%.json}.sha256" ]]
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p1 preview \
  --declaration "${MUTACO_SBI_ZOO_P1_DECLARATION}"
mkdir -p "${MUTACO_SBI_ZOO_P1_JOB_LOG_DIR}"

array_job=""
collector_job=""
submission_committed=0
rollback_partial_submission() {
  if (( submission_committed == 0 )); then
    [[ -z "${collector_job}" ]] || scancel "${collector_job%%;*}" >/dev/null 2>&1 || true
    [[ -z "${array_job}" ]] || scancel "${array_job%%;*}" >/dev/null 2>&1 || true
  fi
}
trap rollback_partial_submission EXIT

array_job="$(sbatch --parsable --begin="${MUTACO_SBI_ZOO_P1_BEGIN_DELAY}" \
  --array="${MUTACO_SBI_ZOO_P1_ARRAY}%${MUTACO_SBI_ZOO_P1_CONCURRENCY}" \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${MUTACO_SBI_ZOO_P1_CPUS_PER_TASK}" \
  --mem="${MUTACO_SBI_ZOO_P1_MEMORY}" --time="${MUTACO_SBI_ZOO_P1_WALLTIME}" \
  --no-requeue --chdir="${root}" \
  --output="${MUTACO_SBI_ZOO_P1_JOB_LOG_DIR}/%x-%A_%a.out" \
  --error="${MUTACO_SBI_ZOO_P1_JOB_LOG_DIR}/%x-%A_%a.err" \
  "${root}/infrastructure/truba/sbi_zoo_p1_worker.sbatch")"
array_job_id="${array_job%%;*}"
[[ "${array_job_id}" =~ ^[0-9]+$ ]]
collector_job="$(sbatch --parsable --dependency="afterany:${array_job_id}" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="${MUTACO_SBI_ZOO_P1_COLLECTOR_CPUS_PER_TASK}" \
  --mem="${MUTACO_SBI_ZOO_P1_COLLECTOR_MEMORY}" \
  --time="${MUTACO_SBI_ZOO_P1_COLLECTOR_WALLTIME}" \
  --no-requeue --chdir="${root}" \
  --output="${MUTACO_SBI_ZOO_P1_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_SBI_ZOO_P1_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/sbi_zoo_p1_collector.sbatch")"
collector_job_id="${collector_job%%;*}"
[[ "${collector_job_id}" =~ ^[0-9]+$ ]]
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p1 scheduler-record \
  --declaration "${MUTACO_SBI_ZOO_P1_DECLARATION}" \
  --array-job-id "${array_job_id}" --collector-job-id "${collector_job_id}"
submission_committed=1
trap - EXIT
printf '{"array_job_id":"%s","collector_job_id":"%s","automatic_retry":false,"requeue":false}\n' \
  "${array_job_id}" "${collector_job_id}"
