#!/usr/bin/env bash
# Scalar-primary population array worker. All scheduler values are pinned by sbatch CLI.
#SBATCH --partition=barbun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --job-name=scalar-primary

set -euo pipefail
cd "${MUTACO_PAPER120_ROOT}"
exec bash infrastructure/truba/run_scalar_primary_population_worker.sh
