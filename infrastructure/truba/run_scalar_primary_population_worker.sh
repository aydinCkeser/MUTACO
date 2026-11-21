#!/usr/bin/env bash
set -euo pipefail

exec "${MUTACO_V3_PYTHON}" \
  -u -m infrastructure.truba.scalar_primary_population_task
