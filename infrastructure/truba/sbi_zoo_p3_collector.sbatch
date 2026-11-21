#!/usr/bin/env bash
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --time=00:15:00
#SBATCH --job-name=mutaco-sbi-zoo-p3-collect

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
exec "${MUTACO_V3_PYTHON}" -u -m experiments.graph_inversion.sbi_zoo_p3 collect \
  --declaration "${MUTACO_SBI_ZOO_P3_DECLARATION}" --write
