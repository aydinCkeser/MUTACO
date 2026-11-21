#!/usr/bin/env bash
# Validate by default. Submission is a separate scientific authorization gate.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI Python}"
: "${MUTACO_PRIMARY_SMOKE_DECLARATION:?pin smoke declaration}"
: "${MUTACO_PRIMARY_SMOKE_PLAN:?pin smoke plan}"
: "${MUTACO_PRIMARY_SMOKE_TARGET_RESPONSES:?pin frozen target ledger}"
: "${MUTACO_PRIMARY_SMOKE_CONFIG:?pin graph configuration}"
: "${MUTACO_PRIMARY_SMOKE_OUTPUT_ROOT:?pin append-only output root}"
: "${MUTACO_PRIMARY_SMOKE_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_primary_integration_smoke.sh [--validate-only|--submit]" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_SMOKE_WORKERS != 1 || (MUTACO_PRIMARY_SMOKE_MAXIMUM_CALLS != 123 && MUTACO_PRIMARY_SMOKE_MAXIMUM_CALLS != 303) )); then
  echo "primary smoke requires one worker and an exact 123- or 303-call cap" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_SMOKE_SBATCH_NTASKS != MUTACO_PRIMARY_SMOKE_SBATCH_NODES * MUTACO_PRIMARY_SMOKE_SBATCH_NTASKS_PER_NODE )); then
  echo "primary smoke allocation must fill its node" >&2
  exit 64
fi
if (( MUTACO_PRIMARY_SMOKE_SBATCH_NTASKS_PER_NODE * MUTACO_PRIMARY_SMOKE_SBATCH_CPUS_PER_TASK != 20 )); then
  echo "primary smoke Barbun allocation must request 20 cores" >&2
  exit 64
fi
root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_PRIMARY_SMOKE_JOB_LOG_DIR}" \
  --output-root "${MUTACO_PRIMARY_SMOKE_OUTPUT_ROOT}"
[[ ! -e "${MUTACO_PRIMARY_SMOKE_OUTPUT_ROOT}" ]]
mkdir -p "${MUTACO_PRIMARY_SMOKE_JOB_LOG_DIR}"
for path in "${MUTACO_PRIMARY_SMOKE_DECLARATION}" "${MUTACO_PRIMARY_SMOKE_PLAN}" "${MUTACO_PRIMARY_SMOKE_CONFIG}"; do
  [[ -f "${path}" ]]
done
if [[ -n "${MUTACO_PRIMARY_SMOKE_QLOGNEI_SETTINGS:-}" ]]; then
  [[ -f "${MUTACO_PRIMARY_SMOKE_QLOGNEI_SETTINGS}" ]]
fi
[[ -d "${MUTACO_PRIMARY_SMOKE_TARGET_RESPONSES}" ]]
preview_args=(
  -m experiments.graph_inversion.run_primary_integration_smoke_worker \
  --declaration "${MUTACO_PRIMARY_SMOKE_DECLARATION}" \
  --plan "${MUTACO_PRIMARY_SMOKE_PLAN}" \
  --targets-root "${MUTACO_PRIMARY_SMOKE_TARGET_RESPONSES}" \
  --config "${MUTACO_PRIMARY_SMOKE_CONFIG}" \
  --output "${MUTACO_PRIMARY_SMOKE_OUTPUT_ROOT}/preview-not-created" \
  --maximum-calls "${MUTACO_PRIMARY_SMOKE_MAXIMUM_CALLS}" --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${MUTACO_PRIMARY_SMOKE_SBATCH_CPUS_PER_TASK}" \
  --torch-interop-threads 1
)
if [[ -n "${MUTACO_PRIMARY_SMOKE_QLOGNEI_SETTINGS:-}" ]]; then
  preview_args+=(--qlognei-settings "${MUTACO_PRIMARY_SMOKE_QLOGNEI_SETTINGS}")
fi
"${MUTACO_V3_PYTHON}" "${preview_args[@]}"
[[ ! -e "${MUTACO_PRIMARY_SMOKE_OUTPUT_ROOT}" ]]
if [[ "${mode}" == "--validate-only" ]]; then
  exit 0
fi
if [[ "${MUTACO_PRIMARY_SMOKE_EXECUTION_AUTHORIZED:-0}" != "1" ]]; then
  echo "primary smoke submission requires separate authorization" >&2
  exit 64
fi
exec sbatch --parsable \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${MUTACO_PRIMARY_SMOKE_SBATCH_CPUS_PER_TASK}" \
  --exclude="${MUTACO_PRIMARY_SMOKE_SBATCH_EXCLUDE}" \
  --chdir="${root}" \
  --output="${MUTACO_PRIMARY_SMOKE_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_PRIMARY_SMOKE_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/primary_integration_smoke_template.sbatch"
