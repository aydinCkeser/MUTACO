#!/usr/bin/env bash
#SBATCH --partition=barbun
#SBATCH --nodes=30
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --time=01:00:00
#SBATCH --job-name=repaired-sbi-assess

set -uo pipefail
cd "${MUTACO_PAPER120_ROOT}"
worker_status=0
srun --ntasks="${MUTACO_REPAIRED_SBI_WORKERS}" \
  --cpus-per-task="${MUTACO_REPAIRED_SBI_SBATCH_CPUS_PER_TASK}" \
  --cpu-bind=cores --kill-on-bad-exit=0 \
  bash infrastructure/truba/run_repaired_sbi_assessment_worker.sh || worker_status=$?
collector_status=0
"${MUTACO_V3_PYTHON}" -u -m infrastructure.truba.collect_repaired_sbi_assessment \
  --tasks "${MUTACO_REPAIRED_SBI_TASKS}" \
  --workers-root "${MUTACO_REPAIRED_SBI_OUTPUT_ROOT}" \
  --output "${MUTACO_REPAIRED_SBI_OUTPUT_ROOT}/collection" \
  --workers "${MUTACO_REPAIRED_SBI_WORKERS}" \
  --expected-commit "${MUTACO_EXPECTED_COMMIT}" \
  --package-identity "${MUTACO_REPAIRED_SBI_PACKAGE_IDENTITY}" || collector_status=$?
if (( worker_status != 0 )); then exit "${worker_status}"; fi
exit "${collector_status}"
