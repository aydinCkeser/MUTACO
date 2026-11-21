#!/usr/bin/env bash
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=24:00:00
#SBATCH --job-name=mutaco-sbi-zoo-qual

set -uo pipefail
cd "${MUTACO_PAPER120_ROOT}"

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

module=("${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.sbi_zoo_qualification)
common=(--declaration "${MUTACO_SBI_ZOO_DECLARATION}")

status=0
"${module[@]}" scheduler-record "${common[@]}" \
  --job-id "${SLURM_JOB_ID}" --node-list "${SLURM_JOB_NODELIST}" || status=$?

if (( status == 0 )); then
  "${module[@]}" dataset "${common[@]}" --budget 40 \
    --target-record "${MUTACO_SBI_ZOO_TARGET_RECORD}" \
    --execute --confirm-approved || status=$?
fi
if (( status == 0 )); then
  "${module[@]}" dataset "${common[@]}" --budget 100 \
    --target-record "${MUTACO_SBI_ZOO_TARGET_RECORD}" \
    --execute --confirm-approved || status=$?
fi
if (( status == 0 )); then
  for budget in 40 100; do
    for method in sbi_nle_single_round sbi_nre_single_round; do
      "${module[@]}" method "${common[@]}" --budget "${budget}" --method "${method}" \
        --target-record "${MUTACO_SBI_ZOO_TARGET_RECORD}" \
        --execute --confirm-approved || { status=$?; break 2; }
    done
  done
fi

collection_status=0
"${module[@]}" collect "${common[@]}" --write || collection_status=$?
if (( status != 0 )); then exit "${status}"; fi
exit "${collection_status}"
