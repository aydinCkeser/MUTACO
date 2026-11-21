# Sequential-observation commitment regression

The fifth extraction moves only post-evaluation standardization and commitment
to _standardize_and_commit_sequential_observation. The actual order remains:
proposal/duplicate handling and ID creation, _evaluate (attempt accounting and
events), evaluation-failure return, frozen transformation, record replacement,
append, committed-observations counter, sequential-observations counter, then
fit_version. No event, callback or accounting boundary moves into a new stage.
The runner now names the scientific operation between evaluation and GP update.

The helper receives the successful raw observation, frozen transform, intent,
original observation list and logical counters. It returns None on commitment
or the original exception class/message for the runner's existing
RESPONSE_TRANSFORM_FAILED finish. Only the transform call is caught. The exact
standardized.detach().clone().contiguous() expression, replace, append and
counter operations stay outside that catch. Their exceptions still propagate.
No new data structure, numerical operation, validation or random draw is added.

A committed record replaces rather than mutates the raw record. Theta, physical
response, context and attempts keep their original references; standardized
values receive the existing independent clone. The raw record stays
unstandardized. Physical-attempt success is retained even if logical commitment
fails. Failed evaluation wins before any transform call. The next GP update
runs only after successful commitment.

All 21 characterization cases passed before and after extraction: multiple
updates, both dtypes, all evaluation-seed policies, order/identity/storage,
retries, partial evaluation failure and precedence, callbacks and duplicate
recovery. Real GP/proposal/transform/recovery functions use fake adapters with
physical-backend tripwires. Two explicit resource-fault fixtures inject an
allocation failure at the existing transform or commit clone. Transform failure
returns the same status/message and partial state; commit-clone failure remains
uncaught, with no committed row/counters and no RUN_FINISHED. A controlled
duplicate proposal uses a real raw pool and production identity/recovery code;
it does not substitute candidates or observations in historical replay.

The full local suite including import boundaries had exactly identical 273
outcomes before and after: 264 passed, 8 skipped and the known NEI Cholesky
fixture failure. Metadata/lifecycle checks passed all 42 cases. Whole-runner
AST comparison passes after inlining the raw-record alias and original error
return. The fourth extraction's reachable overflow-to-GP-failure test also
passes unchanged; no mathematical error policy was tightened.

One Slurm CPU job, 6854744, requested 20 CPUs, 20 GiB and a 20-minute limit.
It ran 404 seconds, with batch MaxRSS 14168904 KiB
(13.513 GiB). TRUBA tests: 261 passed, 8 skipped and the
same NEI failure. All 21 additions passed; no accepted test changed outcome.
Slurm FAILED/exit1 retains that known failure separately from the passing
scientific gate. No automatic retry or second job was submitted.

B40 target 16 (/records/48) passed all 40 archived observations, 25 fit/update
events, 24 proposals and one recommendation. Its strict replay and exact
checkpoint gates passed before B100 ran. B100 target 0 (/records/0) passed all
100 observations, 85 fit/update events, 84 proposals and one recommendation.
Observation order, physical/standardized data, frozen transforms, every training
identity (25/85), GP/posterior states, acquisition decisions/candidates, contexts,
seeds/RNG, duplicate recovery, terminal recommendation and accounting agree.

| Case | Exact full replay hash | Exact checkpoint hash |
| --- | --- | --- |
| B40 | 687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782 | 12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c |
| B100 | c97b62ec714905ff97037561aad8ac3aef638abf9e189a926d6306844d3854c2 | 06338d126708d62fdb3967db3d1603eb1529270ec4c6704322b621cd6b4fa150 |

The qualified numerical environment matches. Only previously declared timer/
provenance exclusions and unordered enumeration of complete library metadata
apply. No tolerance, baseline value, candidate or numerical policy was changed.
The original baseline was not rerun. Logical archived-response evaluations are
counted; every real physical-simulator tripwire recorded zero attempts.

Starting accepted commit: 47c81dc71bd9608b952448d5beb878017b96bb28.
Executed clean snapshot: 15a6ff67dd73d4b47b2a6cb40fdf2a8361aa00e0.
Original scientific baseline: aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3.
The production source/tests and unchanged capture/workflow files match the
private executed snapshot byte for byte. Final metadata records the completed
run. Development remains on ege/repro-refactor; no new development branch,
GitHub push or PR update was made. A separate clean snapshot on the existing
branch name served the exact-source lifecycle. Original private history and
previous reference backups remain intact.

Changed files are runner.py, its focused tests, the refactor-validation
contract, sequential_commit_reference_inputs.json, project/state.toml,
project/experiment_registry.toml and this directory's six evidence files.
The GP/acquisition/recommendation implementations, scientific configurations,
existing launchers, numerical engine, batch script and manuscript are unchanged.

These are portable derivatives, not raw acceptance packets. The manifest
separately binds raw-file SHA-256, scientific hashes and derived-file SHA-256.
Raw logs, complete captures, original environment records, processing code and
source bundle remain privately checksum-verified outside Git. Historical GP
intermediates were not archived: reference GP states are reconstructed, and
the conditional final-data cold fit is distinct from the historical terminal
warm state. Coverage remains these two trajectories and characterized paths.
The known NEI fixture and unavailable CUDA/SBI coverage remain limitations.

Verify portable bytes from this directory:

    shasum -a 256 -c SHA256SUMS

Independent raw verification restores the private snapshot and original inputs,
checks sequential_commit_reference_inputs.json and worker SHA256SUMS, then uses
the unchanged refactor_reference compare and readability_regression history
gates. Sanitized environment summaries cannot replace raw acceptance inputs.

Next proposal only: separate GP-fit outcome bookkeeping and progress reporting
from the actual fit, preserving recovery-attempt counts, failure-state updates
and event order. Characterize those paths and require both saved gates first.
No next extraction, fitting-policy change or scientific investigation is begun.
