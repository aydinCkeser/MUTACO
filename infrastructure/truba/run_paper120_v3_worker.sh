#!/usr/bin/env bash
# Bind one Slurm task to one deterministic unified Paper-v3 shard.
set -euo pipefail

: "${SLURM_PROCID:?Paper-v3 workers require a Slurm process rank}"
: "${MUTACO_V3_OUTPUT_ROOT:?pin the append-only worker-output root}"
: "${MUTACO_V3_WORKERS:?pin the worker count}"

export MUTACO_V3_RANK="${SLURM_PROCID}"
export MUTACO_V3_OUTPUT="${MUTACO_V3_OUTPUT_ROOT}/worker-$(printf '%02d' "${SLURM_PROCID}")"

exec bash infrastructure/truba/run_paper_v3_worker.sh
