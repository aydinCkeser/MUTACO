#!/usr/bin/env bash
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean checkout}"
: "${MUTACO_V3_PYTHON:?pin mutaco-sbi Python}"
: "${MUTACO_OSCILLATOR_OUTPUT_ROOT:?pin append-only output root}"
: "${MUTACO_OSCILLATOR_JOB_LOG_DIR:?pin job log root}"
: "${MUTACO_EXPECTED_COMMIT:?pin exact commit}"

mode="${1:---validate-only}"
[[ "${mode}" == "--validate-only" || "${mode}" == "--submit" ]] || exit 64
[[ "${MUTACO_OSCILLATOR_CPUS_PER_TASK}" == "20" ]]
[[ "${MUTACO_OSCILLATOR_RUNTIME_THREADS}" == "20" ]]
[[ "${MUTACO_OSCILLATOR_MEMORY}" == "64G" ]]
[[ "${MUTACO_OSCILLATOR_WALLTIME}" == "00:30:00" ]]

root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
[[ -f "${root}/${MUTACO_OSCILLATOR_DECLARATION}" ]]
[[ ! -e "${MUTACO_OSCILLATOR_OUTPUT_ROOT}" ]]
export PYTHONPATH="${root}/src:${root}/experiments/analytic_target_matching/src:${root}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${root}"
"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" --log-root "${MUTACO_OSCILLATOR_JOB_LOG_DIR}" \
  --output-root "${MUTACO_OSCILLATOR_OUTPUT_ROOT}"
"${MUTACO_V3_PYTHON}" -m experiments.paper.oscillator_physics_post_analysis.analysis \
  preflight --output-root "${MUTACO_OSCILLATOR_OUTPUT_ROOT}"

if [[ "${mode}" == "--validate-only" ]]; then
  exit 0
fi

[[ "${MUTACO_OSCILLATOR_EIGHT_CALLS_AUTHORIZED:-0}" == "1" ]]
mkdir -p "${MUTACO_OSCILLATOR_JOB_LOG_DIR}"
job="$(sbatch --parsable --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${MUTACO_OSCILLATOR_CPUS_PER_TASK}" \
  --mem="${MUTACO_OSCILLATOR_MEMORY}" --time="${MUTACO_OSCILLATOR_WALLTIME}" \
  --no-requeue --chdir="${root}" \
  --output="${MUTACO_OSCILLATOR_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_OSCILLATOR_JOB_LOG_DIR}/%x-%j.err" \
  "${root}/experiments/paper/oscillator_physics_post_analysis/truba/worker.sbatch")"
job_id="${job%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]]
printf '{"job_id":"%s","automatic_retry":false,"maximum_scientific_calls":8}\n' "${job_id}"
