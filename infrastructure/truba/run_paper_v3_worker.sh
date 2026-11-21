#!/usr/bin/env bash
# Strict capped Paper-v3 worker. Submission always requires explicit approval.
set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

: "${MUTACO_EXPECTED_COMMIT:?future approved execution must pin MUTACO_EXPECTED_COMMIT}"
: "${MUTACO_V3_DECLARATION:?approved execution must pin a declaration}"
: "${MUTACO_V3_METHOD_SPECS:?approved execution must pin unified method specs}"
: "${MUTACO_V3_OUTPUT:?approved execution must isolate worker output}"
: "${MUTACO_V3_MAXIMUM_CALLS:?approved execution must pin an exact call cap}"
: "${MUTACO_V3_RANK:?approved execution must pin the worker rank}"
: "${MUTACO_V3_WORKERS:?approved execution must pin the worker count}"
: "${MUTACO_V3_PYTHON:?approved execution must pin the isolated Python}"
: "${SLURM_CPUS_PER_TASK:?approved execution must pin CPUs per worker task}"

ROOT="$(git rev-parse --show-toplevel)"
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${MUTACO_EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${ROOT}" status --porcelain)" ]]
export PYTHONPATH="${ROOT}:${ROOT}/src:${ROOT}/experiments/analytic_target_matching/src:${ROOT}/simulators/graph-simulator/src${PYTHONPATH:+:${PYTHONPATH}}"

success_flag=()
if [[ "${MUTACO_V3_REQUIRE_ALL_SUCCESSFUL:-1}" == "1" ]]; then
  success_flag+=(--require-all-successful)
elif [[ "${MUTACO_V3_REQUIRE_ALL_SUCCESSFUL}" != "0" ]]; then
  echo "MUTACO_V3_REQUIRE_ALL_SUCCESSFUL must be 0 or 1" >&2
  exit 64
fi

target_args=()
if [[ -n "${MUTACO_V3_TARGET_LEDGER:-}" && -z "${MUTACO_V3_SUPPLIED_TARGETS:-}" ]]; then
  target_args+=(--target-ledger "${MUTACO_V3_TARGET_LEDGER}")
elif [[ -n "${MUTACO_V3_SUPPLIED_TARGETS:-}" && -z "${MUTACO_V3_TARGET_LEDGER:-}" ]]; then
  target_args+=(--supplied-targets "${MUTACO_V3_SUPPLIED_TARGETS}")
else
  echo "pin exactly one of MUTACO_V3_TARGET_LEDGER or MUTACO_V3_SUPPLIED_TARGETS" >&2
  exit 64
fi

exec "${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.run_paper_v3_worker \
  --declaration "${MUTACO_V3_DECLARATION}" \
  --method-specs "${MUTACO_V3_METHOD_SPECS}" \
  "${target_args[@]}" \
  --output "${MUTACO_V3_OUTPUT}" \
  --rank "${MUTACO_V3_RANK}" \
  --workers "${MUTACO_V3_WORKERS}" \
  --maximum-calls "${MUTACO_V3_MAXIMUM_CALLS}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${SLURM_CPUS_PER_TASK}" \
  --torch-interop-threads 1 \
  "${success_flag[@]}" \
  --execute --confirm-approved
