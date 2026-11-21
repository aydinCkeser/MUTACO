#!/usr/bin/env bash
# Zero-call collector runs after every array task reaches a terminal state.
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --job-name=scalar-primary-collect

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
exec "${MUTACO_V3_PYTHON}" -u -m infrastructure.truba.collect_primary_benchmark primary \
  --declaration "${MUTACO_SCALAR_PRIMARY_DECLARATION}" \
  --workers-root "${MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT}" \
  --output "${MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT}/collection" \
  --workers "${MUTACO_SCALAR_PRIMARY_WORKERS}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}"
