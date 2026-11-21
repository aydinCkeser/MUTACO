#!/usr/bin/env bash
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin SBI Python}"
: "${MUTACO_SBI_ZOO_P3_DECLARATION:?pin P3 declaration}"
: "${MUTACO_SBI_ZOO_P3_P2_DECLARATION:?pin P2 declaration}"
: "${MUTACO_SBI_ZOO_P3_P2_COLLECTION:?pin P2 PASS collection}"
: "${MUTACO_SBI_ZOO_P3_OUTPUT_ROOT:?pin P3 output}"
: "${MUTACO_SBI_ZOO_P3_JOB_LOG_DIR:?pin P3 logs}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact commit}"

mode="${1:---validate-only}"
[[ "${mode}" == "--validate-only" || "${mode}" == "--submit" ]] || exit 64
[[ "${MUTACO_SBI_ZOO_P3_ARRAY}" == "0-59" ]]
(( MUTACO_SBI_ZOO_P3_CONCURRENCY == 20 ))
(( MUTACO_SBI_ZOO_P3_RUNTIME_THREADS_PER_WORKER == 20 ))
(( MUTACO_SBI_ZOO_P3_CPUS_PER_TASK == 20 ))
[[ "${MUTACO_SBI_ZOO_P3_MEMORY}" == "64G" ]]

root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_SBI_ZOO_P3_JOB_LOG_DIR}" \
  --output-root "${MUTACO_SBI_ZOO_P3_OUTPUT_ROOT}"
[[ -f "${MUTACO_SBI_ZOO_P3_P2_DECLARATION}" ]]
[[ -f "${MUTACO_SBI_ZOO_P3_P2_COLLECTION}" ]]

if [[ "${mode}" == "--validate-only" ]]; then
  [[ ! -e "${MUTACO_SBI_ZOO_P3_OUTPUT_ROOT}" ]]
  "${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p3 preflight \
    --p2-declaration "${MUTACO_SBI_ZOO_P3_P2_DECLARATION}" \
    --p2-collection "${MUTACO_SBI_ZOO_P3_P2_COLLECTION}" \
    --output-root "${MUTACO_SBI_ZOO_P3_OUTPUT_ROOT}" \
    --execution-commit "${MUTACO_EXPECTED_COMMIT}" --execution-tree "$(git rev-parse HEAD^{tree})"
  exit 0
fi

[[ "${MUTACO_SBI_ZOO_P3_EXECUTION_AUTHORIZED:-0}" == "1" ]]
[[ -f "${MUTACO_SBI_ZOO_P3_DECLARATION}" ]]
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p3 preview --declaration "${MUTACO_SBI_ZOO_P3_DECLARATION}"
[[ ! -e "${MUTACO_SBI_ZOO_P3_OUTPUT_ROOT}" ]]
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p3 initialize --declaration "${MUTACO_SBI_ZOO_P3_DECLARATION}"
mkdir -p "${MUTACO_SBI_ZOO_P3_JOB_LOG_DIR}"

array_job="$(sbatch --parsable --array="${MUTACO_SBI_ZOO_P3_ARRAY}%${MUTACO_SBI_ZOO_P3_CONCURRENCY}" \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task="${MUTACO_SBI_ZOO_P3_CPUS_PER_TASK}" \
  --mem="${MUTACO_SBI_ZOO_P3_MEMORY}" --time="${MUTACO_SBI_ZOO_P3_WALLTIME}" --no-requeue \
  --chdir="${root}" --output="${MUTACO_SBI_ZOO_P3_JOB_LOG_DIR}/%x-%A_%a.out" \
  --error="${MUTACO_SBI_ZOO_P3_JOB_LOG_DIR}/%x-%A_%a.err" \
  "${root}/infrastructure/truba/sbi_zoo_p3_worker.sbatch")"
array_job_id="${array_job%%;*}"
[[ "${array_job_id}" =~ ^[0-9]+$ ]]
collector_job="$(sbatch --parsable --dependency="afterany:${array_job_id}" \
  --nodes=1 --ntasks=1 --cpus-per-task="${MUTACO_SBI_ZOO_P3_COLLECTOR_CPUS_PER_TASK}" \
  --mem="${MUTACO_SBI_ZOO_P3_COLLECTOR_MEMORY}" --time="${MUTACO_SBI_ZOO_P3_COLLECTOR_WALLTIME}" \
  --no-requeue --chdir="${root}" --output="${MUTACO_SBI_ZOO_P3_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_SBI_ZOO_P3_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/sbi_zoo_p3_collector.sbatch")"
collector_job_id="${collector_job%%;*}"
[[ "${collector_job_id}" =~ ^[0-9]+$ ]]
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p3 scheduler-record \
  --declaration "${MUTACO_SBI_ZOO_P3_DECLARATION}" --array-job-id "${array_job_id}" --collector-job-id "${collector_job_id}"
printf '{"array_job_id":"%s","collector_job_id":"%s","automatic_retry":false}\n' "${array_job_id}" "${collector_job_id}"
