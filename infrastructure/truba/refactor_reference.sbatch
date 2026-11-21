#!/usr/bin/env bash
# Resource arguments and user-owned log paths are supplied explicitly to sbatch.
set -euo pipefail
[[ "$#" -eq 4 ]]
source_root="$1"
output_root="$2"
runtime="$3"
baseline_root="$4"
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-0}" -eq 20 ]]
[[ "$(hostname)" != *ui* ]]
[[ ! -e "$output_root" && ! -e "$baseline_root" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
# The original science and harness use their revision's native paths. Keep both
# checkouts outside the candidate repository and retain them with the run.
harness="${baseline_root}.qualification-harness"
[[ ! -e "$harness" && ! -L "$harness" ]]
case "$(realpath -m -- "$baseline_root")/" in "$(realpath -- "$source_root")/"*) exit 64 ;; esac
git -C "$source_root" worktree add --detach "$harness" 9f3ce1893112bb37c23e3aad1508564f55d2c81b
git -C "$source_root" worktree add --detach "$baseline_root" aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3
mkdir -- "$output_root"
"$runtime" -m pip freeze --all > "$output_root/environment.freeze.txt"
worker="$harness/infrastructure/truba/refactor_reference.py"
test_status=0
"$runtime" "$worker" tests --source-root "$harness" --baseline-root "$baseline_root" \
  --output-root "$output_root/tests" > "$output_root/pytest.log" 2>&1 || test_status=$?
math_status=0
for repeat in 1 2; do
  "$runtime" "$worker" capture --source-root "$harness" --baseline-root "$baseline_root" \
    --budget 40 --target 16 --output "$output_root/b40-reference-$repeat.json.gz" \
    > "$output_root/b40-reference-$repeat.log" 2>&1 || { math_status=$?; break; }
done
if [[ "$math_status" -eq 0 ]]; then
  "$runtime" "$worker" compare --first "$output_root/b40-reference-1.json.gz" \
    --second "$output_root/b40-reference-2.json.gz" --output "$output_root/b40-repeat-comparison.json" || math_status=$?
fi
historical_status=0
"$runtime" "$worker" historical --source-root "$harness" --baseline-root "$baseline_root" \
  --budget 40 --target 16 --output "$output_root/b40-historical-replay.json.gz" \
  > "$output_root/b40-historical-replay.log" 2>&1 || historical_status=$?
if [[ "$math_status" -eq 0 && "$SECONDS" -lt 500 ]]; then
  for repeat in 1 2; do
    "$runtime" "$worker" capture --source-root "$harness" --baseline-root "$baseline_root" \
      --budget 100 --target 0 --output "$output_root/b100-reference-$repeat.json.gz" \
      > "$output_root/b100-reference-$repeat.log" 2>&1 || { math_status=$?; break; }
  done
  if [[ "$math_status" -eq 0 ]]; then
    "$runtime" "$worker" compare --first "$output_root/b100-reference-1.json.gz" \
      --second "$output_root/b100-reference-2.json.gz" --output "$output_root/b100-repeat-comparison.json" || math_status=$?
  fi
fi
"$runtime" - "$output_root" "$test_status" "$math_status" "$historical_status" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
status={'pytest_exit':int(sys.argv[2]),'mathematical_reference_exit':int(sys.argv[3]),
        'historical_replay_execution_exit':int(sys.argv[4]),'physical_simulator_calls_authorized':0}
with (root/'job-status.json').open('x') as stream:json.dump(status,stream,indent=2)
with (root/'SHA256SUMS').open('x') as stream:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.name!='SHA256SUMS' and 'temp' not in path.relative_to(root).parts:
   stream.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(root))+'\n')
print(json.dumps(status))
PY
[[ -z "$(git -C "$baseline_root" status --porcelain --untracked-files=all)" ]]
[[ "$test_status" -eq 0 && "$math_status" -eq 0 && "$historical_status" -eq 0 ]]
