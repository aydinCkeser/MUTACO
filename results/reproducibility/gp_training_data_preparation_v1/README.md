# GP training-data preparation regression

A single cohesive block inside fit_version is shared by the initial fit and
all sequential updates. The fourth extraction moves only that block into
_prepare_gp_training_data(observations, intent, transform). It gathers existing
standardized responses, retains the missing-response check, stacks theta and
responses with their original contiguous calls, and constructs the existing
SurrogateTrainingData with parameter spec, ordered observation IDs and frozen
transform ID. The fit path now explicitly reads prepare data, configure fit,
then fit/recover. No GP implementation or policy is changed.

The two returned tensor stacks own storage independently of observation rows
and earlier/later data versions. No clone, detach, cast, device transfer,
normalization, standardization or in-place mutation was added. Task expansion
and covariance/model inputs remain in the unchanged surrogate implementation.
The existing fit-start event and try/except boundary surround the helper call;
preparation errors retain the original status/message and committed state,
without a fit-finished event or committed fit counter for that attempt.

Ten new characterization cases passed before extraction and afterward. Real
standardization, data validation, GP fitting and model building run with fake
simulator adapters and physical-backend tripwires. They cover 16 initial rows,
17/18-row updates, exact row/response/ID order, frozen transforms, model task
indices and point-major expansion, dtype/device/shape, input-storage ownership,
seed/RNG isolation and accounting. Legitimate duplicate caller parameter rows
are retained with distinct IDs. Partial initial or sequential evaluations never
enter the next fit. No corrupted internal observation is manufactured to test
the defensive missing-standardization guard.

A reachable existing update failure is preserved: finite physical responses can
overflow division by a small frozen scale. The record is committed, then the
unchanged data validator rejects nonfinite standardized values. The runner
returns GP_FIT_FAILED with the same message, prior fit, counters and events.
This synthetic extreme-input characterization is not a new campaign failure.

The complete local scientific/import suite before and after recorded identical
252 individual outcomes: 243 passed, 8 skipped and the known NEI Cholesky fixture
failure. Metadata/lifecycle checks passed all 42 cases. Inlining the helper
recovers the entire original runner AST. No new local failure occurred.

One scheduled CPU job, 6854501, requested 20 CPUs, 20 GiB and a 20-minute limit.
It ran for 413 seconds. Final batch MaxRSS: 12002420 KiB
(11.446 GiB). TRUBA tests: 240 passed, 8 skipped and the
same NEI failure. All ten additions passed; no accepted case changed outcome.
Slurm FAILED/exit1 reflects that known fixture failure; the scientific gate
passed independently.

B40 target 16 (/records/48) passed all 40 archived observations, 25 fit/update
events, 24 proposals and one recommendation. Its exact replay/checkpoint gates
passed before B100 started. B100 target 0 (/records/0) passed all 100 observations,
85 fit/update events, 84 proposals and one recommendation. First/final training
tensor checkpoints and every training-data identity (25/85) match saved original
references. These identities encode ordered theta, standardized responses,
dtype, observation IDs, parameter bounds/names and frozen-transform identity.

| Case | Exact full replay hash | Exact checkpoint hash |
| --- | --- | --- |
| B40 | 687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782 | 12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c |
| B100 | c97b62ec714905ff97037561aad8ac3aef638abf9e189a926d6306844d3854c2 | 06338d126708d62fdb3967db3d1603eb1529270ec4c6704322b621cd6b4fa150 |

Frozen data, GP/posterior checkpoints, seeds/RNG, acquisition branches,
candidates, effective contexts, duplicate recovery, terminal recommendation and
accounting match exactly. Environment comparison retains the established
unordered-library enumeration rule and every complete library field. No
numerical tolerance, candidate substitution or baseline regeneration was used.
Logical archived-response evaluations are counted; all real physical simulator
entry-point tripwires recorded zero attempts. There was one job and no retry.

Starting accepted commit: 415c9eab620fe8e6e1a9b85b26990d17b347043b.
Executed clean snapshot: 8f74cd59b465639e12e795a61df1ab0ddababd9e.
Scientific baseline: aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3.
The final runner, scientific tests and unchanged capture/workflow files match
the privately preserved executed snapshot byte for byte. Final documentation
records the completed run. Development remains on ege/repro-refactor; no new
branch, GitHub push or PR update was performed in the development repository.
An isolated clean snapshot on the same branch name was used by the lifecycle.

Changed files are the runner, its focused tests, the refactor-validation
contract, gp_training_data_reference_inputs.json, project/state.toml,
project/experiment_registry.toml, and this directory's six evidence files.
The numerical engine, job script, GP/acquisition/recommendation implementations,
scientific configurations, manuscript and earlier evidence are unchanged.

These reports are derived, not byte-identical raw acceptance packets. The
manifest binds raw-file hashes, scientific hashes and derived-file hashes.
Raw logs, original environment metadata, complete captures and source bundle
remain privately checksum-verified outside Git. Original historical GP states
remain unavailable; the GP references are reconstructed, and the conditional
final cold-data fit is distinct from the historical terminal warm state.
The existing NEI fixture and unavailable CUDA/SBI coverage remain limitations.

Verify portable bytes from this directory with:

    shasum -a 256 -c SHA256SUMS

Independent raw comparison restores the private snapshot/inputs, verifies
gp_training_data_reference_inputs.json and worker SHA256SUMS, and runs the
unchanged refactor_reference compare and readability_regression history gates.
Sanitized environment summaries never replace the private acceptance inputs.

Next proposal only: extract sequential-observation standardization and commit
as one small stage, preserving transform-failure and partial-state behavior,
then require both reference gates. It is not implemented. GP priors, fitting,
recovery, acquisition and terminal-policy changes remain outside this work.
