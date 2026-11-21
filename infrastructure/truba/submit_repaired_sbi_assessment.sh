#!/usr/bin/env bash
# Validate by default. Scientific submission requires a later explicit gate.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI Python}"
: "${MUTACO_REPAIRED_SBI_PACKAGE:?pin preparation package}"
: "${MUTACO_REPAIRED_SBI_TASKS:?pin assessment tasks}"
: "${MUTACO_REPAIRED_SBI_TARGETS:?pin frozen targets}"
: "${MUTACO_REPAIRED_SBI_CONFIG:?pin graph configuration}"
: "${MUTACO_REPAIRED_SBI_OUTPUT_ROOT:?pin append-only output root}"
: "${MUTACO_REPAIRED_SBI_JOB_LOG_DIR:?pin disjoint scheduler logs}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact source commit}"

mode="${1:---validate-only}"
[[ "${mode}" == "--validate-only" || "${mode}" == "--submit" ]]
(( MUTACO_REPAIRED_SBI_WORKERS == 120 ))
(( MUTACO_REPAIRED_SBI_MAXIMUM_CALLS_PER_WORKER == 6 ))
(( MUTACO_REPAIRED_SBI_SBATCH_NODES == 30 ))
(( MUTACO_REPAIRED_SBI_SBATCH_NTASKS == 120 ))
(( MUTACO_REPAIRED_SBI_SBATCH_NTASKS_PER_NODE == 4 ))
(( MUTACO_REPAIRED_SBI_SBATCH_CPUS_PER_TASK == 5 ))
(( MUTACO_REPAIRED_SBI_SBATCH_NTASKS == MUTACO_REPAIRED_SBI_SBATCH_NODES * MUTACO_REPAIRED_SBI_SBATCH_NTASKS_PER_NODE ))
(( MUTACO_REPAIRED_SBI_SBATCH_NTASKS_PER_NODE * MUTACO_REPAIRED_SBI_SBATCH_CPUS_PER_TASK == 20 ))

root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_REPAIRED_SBI_JOB_LOG_DIR}" \
  --output-root "${MUTACO_REPAIRED_SBI_OUTPUT_ROOT}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.repaired_sbi_assessment \
  --package "${MUTACO_REPAIRED_SBI_PACKAGE}" \
  --workers "${MUTACO_REPAIRED_SBI_WORKERS}" \
  --maximum-calls "${MUTACO_REPAIRED_SBI_MAXIMUM_CALLS_PER_WORKER}"
[[ ! -e "${MUTACO_REPAIRED_SBI_OUTPUT_ROOT}" ]]
for path in "${MUTACO_REPAIRED_SBI_TASKS}" "${MUTACO_REPAIRED_SBI_CONFIG}"; do [[ -f "${path}" ]]; done
[[ -d "${MUTACO_REPAIRED_SBI_TARGETS}" ]]
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.run_repaired_sbi_assessment_worker \
  --tasks "${MUTACO_REPAIRED_SBI_TASKS}" --targets-root "${MUTACO_REPAIRED_SBI_TARGETS}" \
  --config "${MUTACO_REPAIRED_SBI_CONFIG}" \
  --output "${MUTACO_REPAIRED_SBI_OUTPUT_ROOT}/preview-not-created" \
  --rank 0 --workers "${MUTACO_REPAIRED_SBI_WORKERS}" --validate-all-shards \
  --maximum-calls "${MUTACO_REPAIRED_SBI_MAXIMUM_CALLS_PER_WORKER}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${MUTACO_REPAIRED_SBI_SBATCH_CPUS_PER_TASK}" --torch-interop-threads 1
[[ ! -e "${MUTACO_REPAIRED_SBI_OUTPUT_ROOT}" ]]
if [[ "${mode}" == "--validate-only" ]]; then exit 0; fi
[[ "${MUTACO_REPAIRED_SBI_EXECUTION_AUTHORIZED:-0}" == "1" ]]
exec sbatch --parsable --no-requeue \
  --nodes="${MUTACO_REPAIRED_SBI_SBATCH_NODES}" --ntasks="${MUTACO_REPAIRED_SBI_SBATCH_NTASKS}" \
  --ntasks-per-node="${MUTACO_REPAIRED_SBI_SBATCH_NTASKS_PER_NODE}" \
  --cpus-per-task="${MUTACO_REPAIRED_SBI_SBATCH_CPUS_PER_TASK}" \
  --exclude="${MUTACO_REPAIRED_SBI_SBATCH_EXCLUDE}" --chdir="${root}" \
  --output="${MUTACO_REPAIRED_SBI_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_REPAIRED_SBI_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/infrastructure/truba/repaired_sbi_assessment_template.sbatch"
