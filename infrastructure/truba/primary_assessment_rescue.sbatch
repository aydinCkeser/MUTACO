#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=barbun
#SBATCH --exclude=barbun70,barbun[112-119]
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=mutaco-primary-rescue

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
srun --ntasks=1 --cpus-per-task="${MUTACO_PRIMARY_RESCUE_SBATCH_CPUS_PER_TASK}" \
  --cpu-bind=cores --kill-on-bad-exit=0 \
  bash infrastructure/truba/run_primary_assessment_rescue.sh

"${MUTACO_V3_PYTHON}" -u -m infrastructure.truba.collect_primary_benchmark assessment-rescue \
  --declaration "${MUTACO_PRIMARY_RESCUE_DECLARATION}" \
  --workers-root "${MUTACO_PRIMARY_RESCUE_OUTPUT_ROOT}" \
  --output "${MUTACO_PRIMARY_RESCUE_OUTPUT_ROOT}/collection" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}"
