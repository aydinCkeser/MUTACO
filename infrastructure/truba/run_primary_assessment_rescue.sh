#!/usr/bin/env bash
set -euo pipefail

exec "${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.run_primary_assessment_rescue \
  --rescue-declaration "${MUTACO_PRIMARY_RESCUE_DECLARATION}" \
  --primary-declaration "${MUTACO_PRIMARY_RESCUE_PRIMARY_DECLARATION}" \
  --decision-freeze "${MUTACO_PRIMARY_RESCUE_DECISION_FREEZE}" \
  --targets-root "${MUTACO_PRIMARY_RESCUE_TARGET_RESPONSES}" \
  --config "${MUTACO_PRIMARY_RESCUE_CONFIG}" \
  --output "${MUTACO_PRIMARY_RESCUE_OUTPUT_ROOT}/worker-00" \
  --maximum-calls "${MUTACO_PRIMARY_RESCUE_MAXIMUM_CALLS}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --torch-intraop-threads "${MUTACO_PRIMARY_RESCUE_SBATCH_CPUS_PER_TASK}" \
  --torch-interop-threads 1 --execute --confirm-approved
