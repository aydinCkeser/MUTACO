#!/usr/bin/env bash
# Validate by default. Scientific submission requires a later explicit gate.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI Python}"
: "${MUTACO_SCALAR_PRIMARY_DECLARATION:?pin population declaration}"
: "${MUTACO_SCALAR_PRIMARY_PLAN:?pin population plan}"
: "${MUTACO_SCALAR_PRIMARY_TARGET_LEDGER:?pin frozen Paper-120 ledger}"
: "${MUTACO_SCALAR_PRIMARY_CONFIG:?pin graph configuration}"
: "${MUTACO_SCALAR_PRIMARY_QLOGNEI_SETTINGS:?pin scalar qLogNEI settings}"
: "${MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT:?pin append-only output root}"
: "${MUTACO_SCALAR_PRIMARY_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_scalar_primary_population.sh [--validate-only|--submit]" >&2
  exit 64
fi
if (( MUTACO_SCALAR_PRIMARY_WORKERS != 120 )); then
  echo "scalar primary population requires exactly 120 workers" >&2
  exit 64
fi
if [[ "${MUTACO_SCALAR_PRIMARY_ARRAY_TASKS:-}" != "0-59" ]]; then
  echo "scalar primary v3 physical array task indices must be 0-59" >&2
  exit 64
fi
if (( MUTACO_SCALAR_PRIMARY_PHYSICAL_TASKS != 60 || MUTACO_SCALAR_PRIMARY_LOGICAL_WORKERS_PER_TASK != 2 )); then
  echo "scalar primary v3 must map 60 physical tasks to two logical workers each" >&2
  exit 64
fi
if (( MUTACO_SCALAR_PRIMARY_ARRAY_CONCURRENCY != 60 )); then
  echo "scalar primary physical array concurrency is not frozen" >&2
  exit 64
fi
if (( MUTACO_SCALAR_PRIMARY_CPUS_PER_TASK != 20 )); then
  echo "each scalar primary array task must reserve one 20-core Barbun node" >&2
  exit 64
fi
if [[ "${MUTACO_SCALAR_PRIMARY_BEGIN_DELAY:-}" != "now+10minutes" ]]; then
  echo "scalar primary delayed-start transaction is not frozen" >&2
  exit 64
fi
if (( MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER != 123 && MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER != 303 )); then
  echo "scalar primary worker cap must be exactly 123 or 303" >&2
  exit 64
fi
if (( MUTACO_SCALAR_PRIMARY_GLOBAL_MAXIMUM_CALLS != MUTACO_SCALAR_PRIMARY_WORKERS * MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER )); then
  echo "scalar primary global and per-worker call caps disagree" >&2
  exit 64
fi

root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_SCALAR_PRIMARY_JOB_LOG_DIR}" \
  --output-root "${MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT}"
[[ ! -e "${MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT}" ]]
mkdir -p "${MUTACO_SCALAR_PRIMARY_JOB_LOG_DIR}"
for path in \
  "${MUTACO_SCALAR_PRIMARY_DECLARATION}" \
  "${MUTACO_SCALAR_PRIMARY_PLAN}" \
  "${MUTACO_SCALAR_PRIMARY_CONFIG}" \
  "${MUTACO_SCALAR_PRIMARY_QLOGNEI_SETTINGS}"; do
  [[ -f "${path}" ]]
done
[[ -d "${MUTACO_SCALAR_PRIMARY_TARGET_LEDGER}" ]]
"${MUTACO_V3_PYTHON}" \
  -m experiments.graph_inversion.run_primary_integration_smoke_worker \
  --declaration "${MUTACO_SCALAR_PRIMARY_DECLARATION}" \
  --plan "${MUTACO_SCALAR_PRIMARY_PLAN}" \
  --targets-root "${MUTACO_SCALAR_PRIMARY_TARGET_LEDGER}" \
  --config "${MUTACO_SCALAR_PRIMARY_CONFIG}" \
  --qlognei-settings "${MUTACO_SCALAR_PRIMARY_QLOGNEI_SETTINGS}" \
  --output "${MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT}/preview-not-created" \
  --rank 0 --workers "${MUTACO_SCALAR_PRIMARY_WORKERS}" --validate-all-shards \
  --maximum-calls "${MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${MUTACO_SCALAR_PRIMARY_CPUS_PER_TASK}" \
  --torch-interop-threads 1
[[ ! -e "${MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT}" ]]
if [[ "${mode}" == "--validate-only" ]]; then
  exit 0
fi
if [[ "${MUTACO_SCALAR_PRIMARY_EXECUTION_AUTHORIZED:-0}" != "1" ]]; then
  echo "scalar primary population submission requires separate authorization" >&2
  exit 64
fi

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

array_job="$(sbatch --parsable --begin="${MUTACO_SCALAR_PRIMARY_BEGIN_DELAY}" \
  --array="${MUTACO_SCALAR_PRIMARY_ARRAY_TASKS}%${MUTACO_SCALAR_PRIMARY_ARRAY_CONCURRENCY}" \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${MUTACO_SCALAR_PRIMARY_CPUS_PER_TASK}" --exclusive \
  --no-requeue \
  --exclude="${MUTACO_SCALAR_PRIMARY_EXCLUDE}" \
  --chdir="${root}" \
  --output="${MUTACO_SCALAR_PRIMARY_JOB_LOG_DIR}/%x-%A_%a.out" \
  --error="${MUTACO_SCALAR_PRIMARY_JOB_LOG_DIR}/%x-%A_%a.err" \
  "${root}/infrastructure/truba/scalar_primary_population_worker.sbatch")"
array_job_id="${array_job%%;*}"
[[ "${array_job_id}" =~ ^[0-9]+$ ]]
collector_job="$(sbatch --parsable --dependency="afterany:${array_job_id}" --no-requeue \
  --chdir="${root}" \
  --output="${MUTACO_SCALAR_PRIMARY_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_SCALAR_PRIMARY_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/scalar_primary_population_collector.sbatch")"
collector_job_id="${collector_job%%;*}"
[[ "${collector_job_id}" =~ ^[0-9]+$ ]]
submission_committed=1
trap - EXIT
printf '{"array_job_id":"%s","collector_job_id":"%s","transactional_delayed_start":true,"array_begin_delay":"%s","automatic_retry":false}\n' \
  "${array_job_id}" "${collector_job_id}" "${MUTACO_SCALAR_PRIMARY_BEGIN_DELAY}"
