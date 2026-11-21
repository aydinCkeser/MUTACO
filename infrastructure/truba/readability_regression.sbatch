#!/usr/bin/env bash
# One candidate-only allocation: verify saved baselines, B40 gate, then B100.
set -euo pipefail
[[ "$#" -eq 4 || "$#" -eq 5 ]]
source_root="$1"; output="$2"; runtime="$3"; references="$4"
input_options=()
if [[ "$#" -eq 5 ]]; then input_options=(--input-manifest "$5"); fi
[[ -n "$SLURM_JOB_ID" && "$SLURM_CPUS_PER_TASK" -eq 20 ]]
[[ "$(hostname)" != *ui* && ! -e "$output" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
mkdir -- "$output"
cd "$source_root"
worker="$source_root/infrastructure/truba/refactor_reference.py"
stage=private_inputs
scientific_exit=2
test_exit=0
finalize() {
  status="$?"
  trap - EXIT
  "$runtime" -m infrastructure.truba.readability_regression finish --output-root "$output" --stage "$stage" \
    --execution-exit "$status" --scientific-exit "$scientific_exit"
  exit "$status"
}
trap finalize EXIT
"$runtime" -m infrastructure.truba.readability_regression inputs --source-root "$source_root" --references "$references" "${input_options[@]}"
"$runtime" -m pip freeze --all > "$output/environment.freeze.txt"
stage=candidate_tests
"$runtime" "$worker" tests --source-root "$source_root" --baseline-root "$source_root" \
  --output-root "$output/candidate-tests" > "$output/candidate-tests.log" 2>&1 || test_exit=$?
"$runtime" -m infrastructure.truba.readability_regression tests --output-root "$output/candidate-tests" \
  --references "$references" "${input_options[@]}" --output "$output/tests-gate.json"
capture_events=0
if [[ "$#" -eq 5 ]]; then
  capture_events="$("$runtime" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("capture_fit_events",False)))' "$5")"
fi
for budget in 40 100; do
  target=0
  [[ "$budget" -ne 40 ]] || target=16
  event_options=()
  if [[ "$capture_events" -eq 1 ]]; then event_options=(--fit-events "$output/b$budget-fit-events.json"); fi
  stage="b$budget-historical-replay"
  "$runtime" "$worker" historical --source-root "$source_root" --baseline-root "$source_root" \
    --budget "$budget" --target "$target" --output "$output/b$budget-history.json.gz" "${event_options[@]}" > "$output/b$budget-history.log" 2>&1
  "$runtime" -m infrastructure.truba.readability_regression history --budget "$budget" \
    --first "$references/b$budget-history.json.gz" --second "$output/b$budget-history.json.gz" \
    --output "$output/b$budget-history-comparison.json"
  if [[ "$capture_events" -eq 1 ]]; then
    "$runtime" -m infrastructure.truba.readability_regression fit-events \
      --first "$references/b$budget-history.json.gz" --second "$output/b$budget-fit-events.json" \
      --output "$output/b$budget-fit-event-comparison.json"
  fi
  stage="b$budget-checkpoints"
  "$runtime" "$worker" capture --source-root "$source_root" --baseline-root "$source_root" \
    --budget "$budget" --target "$target" --output "$output/b$budget-capture.json.gz" > "$output/b$budget-capture.log" 2>&1
  "$runtime" "$worker" compare --first "$references/b$budget-capture.json.gz" \
    --second "$output/b$budget-capture.json.gz" --output "$output/b$budget-checkpoint-comparison.json"
done
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
scientific_exit=0
stage=completed
# Keep the unchanged known NEI failure visible separately from scientific acceptance.
exit "$test_exit"
