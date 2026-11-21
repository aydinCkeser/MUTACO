#!/usr/bin/env bash
# One bounded archived B100 reference, followed conditionally by its refactor.
set -euo pipefail
[[ "$#" -eq 6 ]]
source_root="$1"; output="$2"; runtime="$3"; baseline="$4"; accepted="$5"; b40_packet="$6"
[[ -n "$SLURM_JOB_ID" && "$SLURM_CPUS_PER_TASK" -eq 20 ]]
[[ "$(hostname)" != *ui* && ! -e "$output" && ! -e "$baseline" && ! -e "$accepted" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
mkdir -- "$output"
stage=setup
scientific_exit=2
test_exit=0
finalize() {
  status="$?"
  trap - EXIT
  "$runtime" -m infrastructure.truba.b100_regression finish --output-root "$output" --stage "$stage" \
    --execution-exit "$status" --scientific-exit "$scientific_exit"
  exit "$status"
}
trap finalize EXIT
# The original science and harness use their revision's native paths. Keep both
# checkouts outside the candidate repository and retain them with the run.
harness="${baseline}.qualification-harness"
[[ ! -e "$harness" && ! -L "$harness" ]]
case "$(realpath -m -- "$baseline")/" in "$(realpath -- "$source_root")/"*) exit 64 ;; esac
case "$(realpath -m -- "$accepted")/" in "$(realpath -- "$source_root")/"*) exit 64 ;; esac
git -C "$source_root" worktree add --detach "$harness" 9f3ce1893112bb37c23e3aad1508564f55d2c81b
git -C "$source_root" worktree add --detach "$baseline" aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3
git -C "$source_root" worktree add --detach "$accepted" 242fac08c1c3dc0041651a731c20ed228969ed6a
"$runtime" -m pip freeze --all > "$output/environment.freeze.txt"
worker="$harness/infrastructure/truba/refactor_reference.py"
stage=baseline_tests
"$runtime" "$worker" tests --source-root "$harness" --baseline-root "$baseline" \
  --output-root "$output/baseline-tests" > "$output/baseline-tests.log" 2>&1 || test_exit=$?
"$runtime" -m infrastructure.truba.b100_regression tests --first "$output/baseline-tests" --output "$output/baseline-tests-gate.json"
stage=baseline_historical_replay
"$runtime" "$worker" historical --source-root "$harness" --baseline-root "$baseline" \
  --budget 100 --target 0 --output "$output/baseline-history.json.gz" > "$output/baseline-history.log" 2>&1
"$runtime" -m infrastructure.truba.b100_regression history --first "$output/baseline-history.json.gz" --output "$output/baseline-history-gate.json"
stage=baseline_checkpoints
"$runtime" "$worker" capture --source-root "$harness" --baseline-root "$baseline" \
  --budget 100 --target 0 --output "$output/baseline-capture.json.gz" > "$output/baseline-capture.log" 2>&1
"$runtime" -m infrastructure.truba.b100_regression environment --first "$output/baseline-capture.json.gz" --second "$b40_packet" \
  --output "$output/qualified-environment-gate.json"
stage=refactor_tests
"$runtime" "$worker" tests --source-root "$harness" --baseline-root "$accepted" \
  --output-root "$output/refactor-tests" > "$output/refactor-tests.log" 2>&1 || test_exit=$?
"$runtime" -m infrastructure.truba.b100_regression tests --first "$output/refactor-tests" --second "$output/baseline-tests" \
  --output "$output/refactor-tests-gate.json"
stage=refactor_historical_replay
"$runtime" "$worker" historical --source-root "$harness" --baseline-root "$accepted" \
  --budget 100 --target 0 --output "$output/refactor-history.json.gz" > "$output/refactor-history.log" 2>&1
"$runtime" -m infrastructure.truba.b100_regression history --first "$output/refactor-history.json.gz" --output "$output/refactor-history-gate.json"
"$runtime" -m infrastructure.truba.b100_regression compare-history --first "$output/baseline-history.json.gz" --second "$output/refactor-history.json.gz" \
  --output "$output/full-replay-comparison.json"
stage=refactor_checkpoints
"$runtime" "$worker" capture --source-root "$harness" --baseline-root "$accepted" \
  --budget 100 --target 0 --output "$output/refactor-capture.json.gz" > "$output/refactor-capture.log" 2>&1
"$runtime" "$worker" compare --first "$output/baseline-capture.json.gz" --second "$output/refactor-capture.json.gz" \
  --output "$output/checkpoint-comparison.json"
[[ -z "$(git -C "$baseline" status --porcelain --untracked-files=all)" ]]
[[ -z "$(git -C "$accepted" status --porcelain --untracked-files=all)" ]]
scientific_exit=0
stage=completed
# Preserve known baseline NEI pytest failure in Slurm status, independently of acceptance.
exit "$test_exit"
