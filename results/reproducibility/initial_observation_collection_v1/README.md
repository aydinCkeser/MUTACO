# Initial-observation readability regression

The third controlled extraction moves the existing ordered initial-observation
loop into _collect_initial_observations. Initial-design generation and its ready
event stay in run_sequential; _evaluate retains retries, context construction,
seed derivation, capability checks, attempt budgets and attempt events. The
helper commits each successful observation before incrementing the same two
logical counters. It returns the existing failure status/message, or None;
the runner's unchanged finish closure retains all partially committed state.
No observation data structure or simulator interface was redesigned.

All 26 new characterization cases passed before production extraction. They
cover float32/64, all three evaluation-seed policies, response/observation-ID
order, retry contexts, partial completion, competing limits, capability drift,
callback failure, exact statuses/messages/events, RNG and accounting. The fake
adapter never enters a physical backend. Successful collection is stopped at
the frozen-preparation boundary so these checks require no GP fit or physical simulator.

Before and after extraction, the full local suite including import boundaries
recorded 233 passes, 8 skips and the same known NEI Cholesky fixture failure.
All 242 individual outcomes match. Metadata/lifecycle tests recorded 42 passes.
Whole-runner AST comparison passes after inlining the helper and restoring the
original finish call in its failure return. Every moved evaluation, append and
counter operation retains its order; the tuple only transports failure values.

The single Slurm job 6854320 used 20 CPUs, 20 GiB and a 20-minute limit. It
finished in 380 seconds; reported batch MaxRSS was 13836304 KiB
(13.195 GiB). Slurm FAILED/exit1 retains the known NEI
failure separately from the passed scientific gate. TRUBA tests: 230 passed,
8 skipped, the same failure. All 26 added cases passed; no accepted case changed.

B40 target 16 (/records/48) passed all 40 archived observations, 25 fits,
24 proposals and one recommendation. Only after its strict replay and checkpoint
gates passed did B100 target 0 (/records/0) run: all 100 observations, 85 fits,
84 proposals and one recommendation passed. Full replay and mathematical
checkpoint hashes match the saved original-source references exactly:

| Case | Full replay hash | Checkpoint hash |
| --- | --- | --- |
| B40 | 687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782 | 12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c |
| B100 | c97b62ec714905ff97037561aad8ac3aef638abf9e189a926d6306844d3854c2 | 06338d126708d62fdb3967db3d1603eb1529270ec4c6704322b621cd6b4fa150 |

Frozen transforms, standardized data, GP fitting/posteriors, acquisition
branches/candidates, effective contexts, seeds/RNG, duplicate recovery, terminal
physical-MAE recommendation and accounting agree. The qualified environment
matches. Only predeclared timer/provenance exclusions and unordered enumeration
of complete backend-library metadata apply. No tolerance, candidate replacement,
reference update, baseline rerun or extra cluster job was used.

The runner's attempt counters include checked fake/archived-response evaluations.
They do not mean real physical simulator execution: every physical entry-point
tripwire recorded zero attempts. Failed logical evaluations retain prior
committed records but are never appended themselves. Capability validation
still precedes total-budget rejection; retry-limit rejection still occurs at
the failed attempt before considering a later attempt's total budget. Runtime,
metadata and tensor failures are not retried, and their existing missing
attempt-finished event is preserved. Callback failure remains isolated.

Starting/published milestone: 5e8bd33c270443a6b6ba617404f23667a08886a1.
Executed clean candidate snapshot: e4867390d85be7be5ec796aef429830295b66256.
Original scientific baseline: aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3.
The candidate source, scientific tests and executed capture/gate files match
the privately preserved snapshot byte for byte. Final metadata documents the
completed run. The permanent branch remains ege/repro-refactor. Exactly one
initial milestone push updated existing draft PR #1 to 5e8bd33; this third
refactor has not been pushed. The original private d7b5b12 history and previous
raw evidence remain preserved and checksum-verified.

The workflow reuses readability_regression and its batch script. Its optional
input manifest selects this extraction's saved accepted test outcomes and 26
new cases; prior defaults remain unchanged. The four original numerical input
hashes and refactor_reference.py numerical engine remain unchanged. No GP prior,
fit/recovery policy, acquisition optimizer, terminal policy, precision, scientific
configuration or manuscript was modified.

Changed files:

- src/mutaco/orchestration/runner.py
- tests/orchestration/test_runner.py
- tests/infrastructure/test_readability_regression.py
- infrastructure/truba/readability_regression.py
- infrastructure/truba/readability_regression.sbatch
- infrastructure/truba/initial_collection_reference_inputs.json
- scientific/refactor_validation/README.md
- project/state.toml and project/experiment_registry.toml
- This directory's README, manifest, comparisons, environment summary, raw hash
  index and SHA256SUMS.

These files are derived summaries, not raw acceptance packets. The manifest
records both original raw-file SHA-256 and derived-file hashes. Raw logs,
original environment records, numerical captures, source bundle and processing
code remain privately preserved outside Git. Historical intermediate GP states
were not archived; all newly captured GP values are reconstructed, and the
conditional final-data cold-fit checkpoint is distinct from the historical
terminal warm state. Coverage is for these two trajectories and the tested
failure paths, without a broader scientific-performance claim.

Verify portable bytes from this directory:

    shasum -a 256 -c SHA256SUMS

Restore the private snapshot/inputs for independent raw verification. Check
initial_collection_reference_inputs.json and worker SHA256SUMS, then use the
unchanged refactor_reference compare command on each baseline/candidate capture.
The readability_regression history gate recomputes exact full-replay hashes.
Sanitized environment summaries cannot replace the original private packets.

Next proposal only: extract training-data assembly immediately before GP fitting,
preserving tensor-stack order, observation/transform IDs and failure boundaries.
Characterize it first and require both saved trajectory gates. No fitter,
recovery, acquisition or terminal change is proposed or implemented here.
Inherited operational paths already present in remote project registries remain
a separate public-release concern; this milestone introduces none.
