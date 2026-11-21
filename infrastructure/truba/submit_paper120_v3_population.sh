#!/usr/bin/env bash
# Validate by default; --submit remains an explicit later authorization boundary.
set -euo pipefail

: "${MUTACO_PAPER120_ROOT:?pin exact clean Paper-v3 checkout}"
: "${MUTACO_V3_PYTHON:?pin isolated SBI 0.26.1 Python}"
: "${MUTACO_V3_OUTPUT_ROOT:?pin new append-only worker root}"
: "${MUTACO_V3_JOB_LOG_DIR:?pin scheduler logs outside source and output}"
: "${MUTACO_EXPECTED_COMMIT:?pin Paper-v3 source commit}"
: "${MUTACO_V3_DECLARATION:?pin the approved population declaration}"
: "${MUTACO_V3_METHOD_SPECS:?pin unified method specs}"
: "${MUTACO_V3_WORKERS:?pin the worker count}"
: "${MUTACO_V3_MAXIMUM_CALLS:?pin the exact per-worker call cap}"

sbatch_nodes="${MUTACO_V3_SBATCH_NODES:-8}"
sbatch_ntasks="${MUTACO_V3_SBATCH_NTASKS:-40}"
sbatch_ntasks_per_node="${MUTACO_V3_SBATCH_NTASKS_PER_NODE:-5}"
sbatch_cpus_per_task="${MUTACO_V3_SBATCH_CPUS_PER_TASK:-8}"
sbatch_exclude="${MUTACO_V3_SBATCH_EXCLUDE:-barbun[112-119]}"

if (( sbatch_ntasks != sbatch_nodes * sbatch_ntasks_per_node )); then
  echo "scheduler allocation must fill every requested node" >&2
  exit 64
fi
if (( sbatch_ntasks_per_node * sbatch_cpus_per_task != 20 )); then
  echo "Barbun scheduler allocation must request exactly 20 CPU cores per node" >&2
  exit 64
fi
if (( sbatch_ntasks < MUTACO_V3_WORKERS )); then
  echo "scheduler task allocation cannot be smaller than the scientific worker count" >&2
  exit 64
fi

mode="${1:---validate-only}"
if [[ "${mode}" != "--validate-only" && "${mode}" != "--submit" ]]; then
  echo "usage: submit_paper120_v3_population.sh [--validate-only|--submit]" >&2
  exit 64
fi

root="$(git -C "${MUTACO_PAPER120_ROOT}" rev-parse --show-toplevel)"
[[ "${root}" == "$(realpath "${MUTACO_PAPER120_ROOT}")" ]]
[[ "$(git -C "${root}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
cd "${root}"

target_args=()
if [[ -n "${MUTACO_V3_TARGET_LEDGER:-}" && -z "${MUTACO_V3_SUPPLIED_TARGETS:-}" ]]; then
  target_args+=(--target-ledger "${MUTACO_V3_TARGET_LEDGER}")
elif [[ -n "${MUTACO_V3_SUPPLIED_TARGETS:-}" && -z "${MUTACO_V3_TARGET_LEDGER:-}" ]]; then
  target_args+=(--supplied-targets "${MUTACO_V3_SUPPLIED_TARGETS}")
else
  echo "pin exactly one of MUTACO_V3_TARGET_LEDGER or MUTACO_V3_SUPPLIED_TARGETS" >&2
  exit 64
fi

"${MUTACO_V3_PYTHON}" -m infrastructure.truba.scheduler_layout \
  --source-root "${root}" \
  --log-root "${MUTACO_V3_JOB_LOG_DIR}" \
  --output-root "${MUTACO_V3_OUTPUT_ROOT}"
"${MUTACO_V3_PYTHON}" -m experiments.graph_inversion.run_paper_v3_worker \
  --declaration "${MUTACO_V3_DECLARATION}" \
  --method-specs "${MUTACO_V3_METHOD_SPECS}" \
  "${target_args[@]}" \
  --output "${MUTACO_V3_OUTPUT_ROOT}/preview-not-created" \
  --rank 0 --workers "${MUTACO_V3_WORKERS}" \
  --maximum-calls "${MUTACO_V3_MAXIMUM_CALLS}" \
  --torch-intraop-threads "${sbatch_cpus_per_task}" --torch-interop-threads 1

[[ ! -e "${MUTACO_V3_OUTPUT_ROOT}" ]]
mkdir -p "${MUTACO_V3_JOB_LOG_DIR}"
[[ -z "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]
if [[ "${mode}" == "--validate-only" ]]; then
  exit 0
fi

exec sbatch --parsable \
  --nodes="${sbatch_nodes}" \
  --ntasks="${sbatch_ntasks}" \
  --ntasks-per-node="${sbatch_ntasks_per_node}" \
  --cpus-per-task="${sbatch_cpus_per_task}" \
  --exclude="${sbatch_exclude}" \
  --chdir="${root}" \
  --output="${MUTACO_V3_JOB_LOG_DIR}/%x-%j.out" \
  --error="${MUTACO_V3_JOB_LOG_DIR}/%x-%j.err" \
  --export="ALL,MUTACO_PAPER120_ROOT=${root},MUTACO_V3_PYTHON=${MUTACO_V3_PYTHON},MUTACO_V3_OUTPUT_ROOT=${MUTACO_V3_OUTPUT_ROOT},MUTACO_EXPECTED_COMMIT=${MUTACO_EXPECTED_COMMIT}" \
  "${root}/infrastructure/truba/paper_v3_population_template.sbatch"
