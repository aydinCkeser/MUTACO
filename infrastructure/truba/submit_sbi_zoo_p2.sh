#!/usr/bin/env bash
# Validate by default. P2 submission requires explicit authorization.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI Python}"
: "${MUTACO_SBI_ZOO_P2_DECLARATION:?pin immutable P2 declaration}"
: "${MUTACO_SBI_ZOO_P2_P1_DECLARATION:?pin immutable P1 declaration}"
: "${MUTACO_SBI_ZOO_P2_P1_COLLECTION:?pin immutable P1 PASS collection}"
: "${MUTACO_SBI_ZOO_P2_OUTPUT_ROOT:?pin append-only P2 output root}"
: "${MUTACO_SBI_ZOO_P2_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"

mode="${1:---validate-only}"
[[ "${mode}" == "--validate-only" || "${mode}" == "--submit" ]] || exit 64
[[ "${MUTACO_SBI_ZOO_P2_ARRAY}" == "0-59" ]]
(( MUTACO_SBI_ZOO_P2_CONCURRENCY == 10 ))
(( MUTACO_SBI_ZOO_P2_RUNTIME_THREADS_PER_WORKER == 20 ))
(( MUTACO_SBI_ZOO_P2_CPUS_PER_TASK == 20 ))
[[ "${MUTACO_SBI_ZOO_P2_MEMORY}" == "64G" ]]
[[ "${MUTACO_SBI_ZOO_P2_WALLTIME}" == "01:00:00" ]]

root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"

"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_SBI_ZOO_P2_JOB_LOG_DIR}" \
  --output-root "${MUTACO_SBI_ZOO_P2_OUTPUT_ROOT}"
[[ ! -e "${MUTACO_SBI_ZOO_P2_OUTPUT_ROOT}" ]]
[[ -f "${MUTACO_SBI_ZOO_P2_P1_DECLARATION}" ]]
[[ -f "${MUTACO_SBI_ZOO_P2_P1_COLLECTION}" ]]

if [[ "${mode}" == "--validate-only" ]]; then
  "${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p2 preflight \
    --p1-declaration "${MUTACO_SBI_ZOO_P2_P1_DECLARATION}" \
    --p1-collection "${MUTACO_SBI_ZOO_P2_P1_COLLECTION}" \
    --output-root "${MUTACO_SBI_ZOO_P2_OUTPUT_ROOT}" \
    --execution-commit "${MUTACO_EXPECTED_COMMIT}" \
    --execution-tree "$(git rev-parse HEAD^{tree})"
  [[ ! -e "${MUTACO_SBI_ZOO_P2_OUTPUT_ROOT}" ]]
  exit 0
fi

[[ "${MUTACO_SBI_ZOO_P2_EXECUTION_AUTHORIZED:-0}" == "1" ]]
[[ -f "${MUTACO_SBI_ZOO_P2_DECLARATION}" ]]
[[ -f "${MUTACO_SBI_ZOO_P2_DECLARATION%.json}.sha256" ]]
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p2 preview \
  --declaration "${MUTACO_SBI_ZOO_P2_DECLARATION}"
mkdir -p "${MUTACO_SBI_ZOO_P2_JOB_LOG_DIR}"

array_job="$(sbatch --parsable \
  --array="${MUTACO_SBI_ZOO_P2_ARRAY}%${MUTACO_SBI_ZOO_P2_CONCURRENCY}" \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${MUTACO_SBI_ZOO_P2_CPUS_PER_TASK}" \
  --mem="${MUTACO_SBI_ZOO_P2_MEMORY}" --time="${MUTACO_SBI_ZOO_P2_WALLTIME}" \
  --no-requeue --chdir="${root}" \
  --output="${MUTACO_SBI_ZOO_P2_JOB_LOG_DIR}/%x-%A_%a.out" \
  --error="${MUTACO_SBI_ZOO_P2_JOB_LOG_DIR}/%x-%A_%a.err" \
  "${root}/infrastructure/truba/sbi_zoo_p2_worker.sbatch")"
array_job_id="${array_job%%;*}"
[[ "${array_job_id}" =~ ^[0-9]+$ ]]

collector_job="$(sbatch --parsable --dependency="afterany:${array_job_id}" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="${MUTACO_SBI_ZOO_P2_COLLECTOR_CPUS_PER_TASK}" \
  --mem="${MUTACO_SBI_ZOO_P2_COLLECTOR_MEMORY}" \
  --time="${MUTACO_SBI_ZOO_P2_COLLECTOR_WALLTIME}" \
  --no-requeue --chdir="${root}" \
  --output="${MUTACO_SBI_ZOO_P2_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_SBI_ZOO_P2_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/sbi_zoo_p2_collector.sbatch")"
collector_job_id="${collector_job%%;*}"
[[ "${collector_job_id}" =~ ^[0-9]+$ ]]

"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.sbi_zoo_p2 scheduler-record \
  --declaration "${MUTACO_SBI_ZOO_P2_DECLARATION}" \
  --array-job-id "${array_job_id}" --collector-job-id "${collector_job_id}"
printf '{"array_job_id":"%s","collector_job_id":"%s","automatic_retry":false,"requeue":false}\n' \
  "${array_job_id}" "${collector_job_id}"
