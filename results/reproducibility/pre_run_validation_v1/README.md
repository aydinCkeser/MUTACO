# Pre-run option readability regression

The second controlled extraction moves only the existing option validation from
run_sequential into _validate_pre_run_options. The runner now states that setup
stage in one call before initial design. Validation statements, named seed
coordinates and their order are identical. Inlining the helper recovers the
entire previous runner AST. Capability capture, seed-family choice, exception
handling, events, numerical operations and evaluation logic remain in place.

Characterization ran before extraction: 29 new cases passed. They cover real
valid options, float32/64, ordinary/diagnostic seed families, beta checks, first
failure when several settings are invalid, and the three existing exception
classes. Failures retain the capability snapshot and identities, their exact
status/message, RUN_STARTED/RUN_FINISHED and zero committed/call counters.
The planned budget remains recorded. No fallback or validation rule changed.

The local relevant suite had 129 passes/3 skips before extraction and
132 passes/3 skips afterward, including three extra import-boundary tests.
All shared cases retained their outcomes. The complete selected scientific
suite had 204 passes, 8 skips and the unchanged NEI Cholesky failure, with zero
physical-backend attempts. Metadata/lifecycle checks had 41 passes.

One Slurm CPU job, 6854173, used 20 CPUs, a 20-GiB request and a 20-minute cap.
It finished in 571 seconds. Final batch MaxRSS was 13462608 KiB (12.839 GiB).
Slurm FAILED/exit1 reflects the known NEI failure, separately from the passing
scientific gate. TRUBA tests: 204 passed, 8 skipped, the same one failure;
29 added cases passed and no accepted case changed outcome.

B40 target 16 (/records/48) passed all 40 stored observations, with 25 fits,
24 proposals and one recommendation. Its replay mathematical hash remains
687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782;
its checkpoint hash remains
12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c.
Only after both B40 gates passed did B100 target 0 (/records/0) execute.
B100 passed all 100 observations, 85 fit/update events, 84 proposals and one
recommendation. Its replay hash remains
c97b62ec714905ff97037561aad8ac3aef638abf9e189a926d6306844d3854c2;
its checkpoint hash remains
06338d126708d62fdb3967db3d1603eb1529270ec4c6704322b621cd6b4fa150.

Both captures match the qualified environment and saved original-source
references exactly. The unchanged comparator normalizes only the previously
declared order of complete library metadata records. Transform/standardized
data, fit/posterior/acquisition states, candidates/contexts, seeds/RNG, duplicate
recovery, terminal physical-MAE recommendation and accounting agree. No
candidate substitution, numerical tolerance, baseline regeneration, physical
simulator attempt, additional job or GitHub push occurred.

Starting clean commit: c295641b74abf6791b74553e7b3a180d487629e5.
Executed clean snapshot: c960fef4d4654253bfbfb94a0003f07d9ac86f1e.
Original scientific source: aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3.
The snapshot is retained privately as a self-contained Git bundle; its runner,
characterization tests and executed numerical harness match this milestone's
files byte for byte. Final metadata/documentation describes the completed run.

The active development branch is ege/repro-refactor, renamed from the clean
temporary branch without resetting or merging histories. The original d7b5b12
checkout remains on an archival private branch with no upstream association;
its previously verified bundle and raw backups are unchanged. The temporary
shareable branch name is removed. Remote development remains at 242fac0.

Files in this change:

- src/mutaco/orchestration/runner.py
- tests/orchestration/test_runner.py
- infrastructure/truba/readability_regression.py
- infrastructure/truba/readability_regression.sbatch
- infrastructure/truba/readability_regression_v1.env
- infrastructure/truba/readability_reference_inputs.json
- infrastructure/truba/validate_exact_source.sh
- tests/infrastructure/test_readability_regression.py
- scientific/refactor_validation/README.md
- project/state.toml and project/experiment_registry.toml
- This directory's README, manifest, derived comparisons, environment summary,
  raw hash index and SHA256SUMS.

The manifest binds original packet hashes to unchanged mathematical hashes and
derived-file hashes. These portable reports are derived summaries, not copies
of raw acceptance packets. Raw logs, original environment records, inputs,
source bundle and derivation script remain in private evidence storage.
Original historical intermediate GP snapshots remain unavailable: captured GP
values are reconstructed. The final-data cold checkpoint remains conditional;
full strict replay separately protects the historical sequential decisions.

Verify portable bytes from this directory:

    shasum -a 256 -c SHA256SUMS

For independent numerical verification, restore the private snapshot/inputs,
check readability_reference_inputs.json and worker SHA256SUMS, then rerun the
unchanged refactor_reference.py compare on each saved baseline/candidate packet.
The tracked readability_regression history gate recomputes exact replay hashes.
Sanitized environment summaries must never replace those private raw inputs.

The next proposed bounded change is initial-observation collection as one named
stage. Characterize partial failures/retries and committed-state accounting
first, then require both saved trajectory gates. It is not implemented here.
Inherited absolute paths in existing remote project registries remain a separate
sharing concern; this change introduces none. No broader release readiness is
claimed.
