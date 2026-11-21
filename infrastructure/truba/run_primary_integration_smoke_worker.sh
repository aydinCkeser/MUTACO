#!/usr/bin/env bash
set -euo pipefail

args=(
  -u -m experiments.graph_inversion.run_primary_integration_smoke_worker \
  --declaration "${MUTACO_PRIMARY_SMOKE_DECLARATION}" \
  --plan "${MUTACO_PRIMARY_SMOKE_PLAN}" \
  --targets-root "${MUTACO_PRIMARY_SMOKE_TARGET_RESPONSES}" \
  --config "${MUTACO_PRIMARY_SMOKE_CONFIG}" \
  --output "${MUTACO_PRIMARY_SMOKE_OUTPUT_ROOT}/worker-00" \
  --maximum-calls "${MUTACO_PRIMARY_SMOKE_MAXIMUM_CALLS}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${MUTACO_PRIMARY_SMOKE_SBATCH_CPUS_PER_TASK}" \
  --torch-interop-threads 1 \
  --execute --confirm-approved
)
if [[ -n "${MUTACO_PRIMARY_SMOKE_QLOGNEI_SETTINGS:-}" ]]; then
  args+=(--qlognei-settings "${MUTACO_PRIMARY_SMOKE_QLOGNEI_SETTINGS}")
fi
exec "${MUTACO_V3_PYTHON}" "${args[@]}"
