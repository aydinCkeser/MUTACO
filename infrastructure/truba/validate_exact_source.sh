#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: validate_exact_source.sh SOURCE OUTPUT LOGS PYTHON COMMIT PROFILE" >&2
  exit 64
fi

root="$1"
output_root="$2"
log_root="$3"
runtime_python="$4"
expected_commit="$5"
profile="$6"

safe_path() {
  [[ "$1" =~ ^/[A-Za-z0-9._/-]+$ ]] && [[ "$1" != *"/../"* ]] && [[ "$1" != */.. ]]
}
safe_path "$root"
safe_path "$output_root"
safe_path "$log_root"
safe_path "$runtime_python"
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$profile" =~ ^[A-Za-z0-9._/-]+$ ]]
[[ "$profile" != *"../"* && "$profile" != ../* && "$profile" != */.. ]]

[[ "$(git -C "$root" rev-parse HEAD)" == "$expected_commit" ]]
[[ -z "$(git -C "$root" status --porcelain --untracked-files=all)" ]]
[[ ! -e "$output_root" ]]
[[ -x "$runtime_python" ]]
[[ -f "$root/$profile" ]]

export MUTACO_PAPER120_ROOT="$root"
export MUTACO_V3_PYTHON="$runtime_python"
export MUTACO_V3_OUTPUT_ROOT="$output_root"
export MUTACO_V3_JOB_LOG_DIR="$log_root"
export MUTACO_EXPECTED_COMMIT="$expected_commit"
source "$root/$profile"
cd "$root"
case "${MUTACO_TRUBA_VALIDATION_SURFACE:-paper_v3}" in
  readability_regression)
    "$runtime_python" -m infrastructure.truba.readability_regression validate \
      --source-root "$root" --output-root "$output_root"
    ;;
  b100_refactor_reference)
    "$runtime_python" -m infrastructure.truba.b100_regression validate \
      --source-root "$root" --output-root "$output_root"
    ;;
  refactor_reference)
    # Preparation runs only stdlib metadata checks on the login node.
    "$runtime_python" infrastructure/truba/refactor_reference.py validate \
      --baseline-commit "$MUTACO_REFACTOR_BASELINE" --source-root "$root" \
      --output-root "$output_root"
    ;;
  paper_v3)
    bash infrastructure/truba/submit_paper120_v3_population.sh --validate-only
    ;;
  primary_nei_q1)
    bash infrastructure/truba/submit_primary_nei_q1.sh --validate-only
    ;;
  primary_nei_q1_target_construction)
    bash infrastructure/truba/submit_primary_nei_q1_target_construction.sh --validate-only
    ;;
  primary_nei_q1_compute_smoke)
    bash infrastructure/truba/submit_primary_nei_q1_compute_smoke.sh --validate-only
    ;;
  primary_integration_smoke)
    bash infrastructure/truba/submit_primary_integration_smoke.sh --validate-only
    ;;
  scalar_primary_population)
    bash infrastructure/truba/submit_scalar_primary_population.sh --validate-only
    ;;
  primary_assessment_rescue)
    bash infrastructure/truba/submit_primary_assessment_rescue.sh --validate-only
    ;;
  repaired_sbi_assessment)
    bash infrastructure/truba/submit_repaired_sbi_assessment.sh --validate-only
    ;;
  sbi_zoo_qualification)
    bash infrastructure/truba/submit_sbi_zoo_qualification.sh --validate-only
    ;;
  sbi_zoo_p1)
    bash infrastructure/truba/submit_sbi_zoo_p1.sh --validate-only
    ;;
  sbi_zoo_p2)
    bash infrastructure/truba/submit_sbi_zoo_p2.sh --validate-only
    ;;
  sbi_zoo_p3)
    bash infrastructure/truba/submit_sbi_zoo_p3.sh --validate-only
    ;;
  oscillator_physics_target098)
    bash experiments/paper/oscillator_physics_post_analysis/truba/submit.sh --validate-only
    ;;
  oscillator_physics_multitarget)
    bash experiments/paper/oscillator_physics_multi_target_sensitivity/truba/submit.sh --validate-only
    ;;
  *)
    echo "unsupported tracked TRUBA validation surface" >&2
    exit 64
    ;;
esac

[[ ! -e "$output_root" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
