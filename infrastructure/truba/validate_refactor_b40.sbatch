#!/usr/bin/env bash
# One authorized candidate-vs-saved-reference job; no new reference generation.
set -euo pipefail
[[ "$#" -eq 4 ]]
candidate="$1"
output="$2"
runtime="$3"
capture_root="$4"
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-0}" -eq 20 ]]
[[ "$(hostname)" != *ui* && ! -e "$output" ]]
[[ -z "$(git -C "$candidate" status --porcelain --untracked-files=all)" ]]
[[ -z "$(git -C "$capture_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$capture_root" rev-parse HEAD)" == 27f654a8e590d3aadb89a2791c91f1e1491a1aad ]]
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
mkdir -- "$output"
references="${MUTACO_PRIVATE_REFERENCE_DIR:?provide verified original raw B40 references outside the checkout}"
[[ -d "$references" && "$references" == /* && "$references" != "$candidate/"* ]]
worker="$capture_root/infrastructure/truba/refactor_reference.py"
comparator="$candidate/infrastructure/truba/refactor_reference.py"
"$runtime" - "$references" "$worker" "$comparator" <<'PY'
import hashlib, json, pathlib, sys
root, worker, comparator = map(pathlib.Path, sys.argv[1:])
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
manifest = json.loads((root / 'manifest.json').read_text())
assert hashlib.sha256(worker.read_bytes()).hexdigest() == manifest['capture_harness_worker_sha256']
assert hashlib.sha256(comparator.read_bytes()).hexdigest() == manifest['comparator_source_sha256']
PY
test_exit=0
"$runtime" "$worker" tests --source-root "$capture_root" --baseline-root "$candidate" \
  --output-root "$output/tests" > "$output/pytest.log" 2>&1 || test_exit=$?
test_gate=0
"$runtime" - "$references" "$output" "$test_exit" <<'PY' || test_gate=$?
import json, pathlib, sys, xml.etree.ElementTree as ET
references, output = map(pathlib.Path, sys.argv[1:3])
def outcomes(path):
    return {(c.attrib['classname'], c.attrib['name']):
            next((tag for tag in ('failure', 'error', 'skipped') if c.find(tag) is not None), 'passed')
            for c in ET.parse(path).iter('testcase')}
before = outcomes(references / 'baseline-tests.xml')
after = outcomes(output / 'tests/junit.xml')
changed = [list(k) for k, value in before.items() if after.get(k) != value]
added = {k: value for k, value in after.items() if k not in before}
status = json.loads((output / 'tests/status.json').read_text())
passed = (not changed and len(added) == 6
          and all(k[0] == 'tests.orchestration.test_runner'
                  and k[1].startswith('test_response_preparation_failure_preserves_committed_state[')
                  and value == 'passed' for k, value in added.items())
          and int(sys.argv[3]) == status['pytest_exit'] == 1
          and status['physical_backend_attempts'] == 0)
report = {'baseline_test_outcomes_preserved': not changed, 'changed_baseline_cases': changed,
          'added_cases': [{'case': list(k), 'outcome': v} for k, v in sorted(added.items())],
          'new_failures': [list(k) for k, v in added.items() if v != 'passed'],
          'physical_backend_attempts': status['physical_backend_attempts'],
          'test_gate_passed': passed, 'pytest_exit': int(sys.argv[3])}
(output / 'test-comparison.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
raise SystemExit(0 if passed else 2)
PY
math_exit=2
historical_gate=2
if [[ "$test_gate" -eq 0 ]]; then
  math_exit=0
  "$runtime" "$worker" capture --source-root "$capture_root" --baseline-root "$candidate" \
    --budget 40 --target 16 --output "$output/candidate-b40.json.gz" \
    > "$output/candidate-b40.log" 2>&1 || math_exit=$?
  if [[ "$math_exit" -eq 0 ]]; then
    "$runtime" "$comparator" compare --first "$references/baseline-b40.json.gz" \
      --second "$output/candidate-b40.json.gz" --output "$output/b40-comparison.json" || math_exit=$?
  fi
  if [[ "$math_exit" -eq 0 ]]; then
    "$runtime" "$worker" historical --source-root "$capture_root" --baseline-root "$candidate" \
      --budget 40 --target 16 --output "$output/candidate-historical-b40.json.gz" \
      > "$output/candidate-historical-b40.log" 2>&1 && historical_gate=0 || historical_gate=$?
    if [[ "$historical_gate" -eq 0 ]]; then
      "$runtime" - "$output/candidate-historical-b40.json.gz" <<'PY' || historical_gate=$?
import gzip, json, sys
with gzip.open(sys.argv[1], 'rt') as stream:
    result = json.load(stream)
assert result['status'] == 'passed', result.get('first_divergence', result)
assert result['archived_observations_consumed'] == 40
assert result['ambient_rng_preserved'] and result['physical_simulator_calls'] == 0
assert result['logical_replay_accounting'] == {
    'committed_fit_events': 25, 'committed_observations': 40, 'committed_proposal_events': 24,
    'committed_recommendation_events': 1, 'initial_observations': 16, 'planned_observations': 40,
    'sequential_observations': 24, 'training_data_versions': 25}
PY
    fi
  fi
fi
"$runtime" - "$output" "$test_exit" "$test_gate" "$math_exit" "$historical_gate" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
test, test_gate, math, historical = map(int, sys.argv[2:])
result = {'pytest_exit': test, 'test_gate_exit': test_gate, 'mathematical_comparison_exit': math,
          'historical_gate_exit': historical,
          'scientific_refactor_gate_passed': test_gate == math == historical == 0,
          'physical_simulator_calls_authorized': 0}
(root / 'job-status.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
with (root / 'SHA256SUMS').open('x') as stream:
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name != 'SHA256SUMS' and 'temp' not in p.relative_to(root).parts:
            stream.write(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(root)) + '\n')
print(json.dumps(result))
PY
[[ -z "$(git -C "$candidate" status --porcelain --untracked-files=all)" ]]
[[ "$test_gate" -eq 0 && "$math_exit" -eq 0 && "$historical_gate" -eq 0 ]]
# Preserve the known baseline pytest failure in Slurm status; use job-status.json
# for the independent scientific gate. Never disable or hide the failing test.
exit "$test_exit"
