# Same-environment scientific refactor validation

The supervising researcher's 26 October 2026 direction replaces cross-platform
historical equality as the refactor acceptance criterion. macOS bitwise
agreement with old trajectories is not required. The completed macOS diagnosis
remains historical engineering documentation; it is not reopened here.

The original scientific source is
`aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3`. Both original and future refactored
code must run in the same qualified TRUBA environment, with identical archived
inputs, scientific options, seeds, dtype/device, thread settings and budgets.
No GP prior, acquisition formula, optimization policy, terminal policy or
published scientific setting is changed by this validation strategy.

## Scientific-core documentation revision

The [Milestone 3C record](../../../benchmarks/reproducibility/scientific_core_documentation_v1/manifest.json)
binds the previous and newly documented bytes of eight core modules. Only
docstrings/comments changed: executable AST, compiled operations, annotations
and imported interfaces agree with `ecf4b35`. Line locations and `__doc__` values
change intentionally; old source hashes are not hashes of the documented files.
The previous 523-pass/8-skip TRUBA and exact B40/B100 results still qualify source
`4f56343`, with evidence recorded at `ecf4b35`. They are not numerical validation
of this documentation revision. Keep original references unchanged; the new
source requires separately authorized qualified testing/replay before acceptance.

## Shareable evidence and private verification inputs

Shareable reports and exact mathematical payloads are in
benchmarks/reproducibility/shareable_milestone_v1/. The original branch, raw
logs, packets, manifests and source history remain in a verified private backup
outside the checkout. Original and derived hashes are recorded separately;
derived summaries are not byte-identical raw acceptance packets.

The raw comparator and exact environment check are unchanged. Its fingerprint
includes library paths, so a redacted backend summary cannot replace it.
Restore the private backup independently and use its original packets for
exact-environment acceptance. The stdlib-only shareable_evidence verifier
checks portable integrity and, with explicit private inputs, also reruns every
original checksum/comparison without numerical calculations or simulations.

The candidate batch script requires MUTACO_PRIVATE_REFERENCE_DIR to select
the original verified B40 reference directory outside the checkout; it fails
when absent and never substitutes a sanitized report. Recover the recorded
historical capture-harness checkout from the private bundle. Unpublished
commit IDs in reports are provenance, not ancestors of the shareable branch.

## Three separate evidence layers

1. **Historical closed-loop replay.** The existing strict archive oracle
   verifies each candidate and effective context before returning its response.
   If the current TRUBA environment differs from the old execution, replay
   stops. Historical disagreement is reported separately; it is not erased or
   converted into a passing historical replay.
2. **Production first-adaptive reference.** The original runner executes the
   first 16 checked archived observations, freezes its transform, fits the GP
   and proposes candidate 17. It stops before returning observation 17. The
   future implementation receives exactly the same inputs and must reproduce
   the mathematical state and logical/oracle accounting in this environment.
3. **Conditional mathematical checkpoints.** A final-data checkpoint uses all
   stored B40 (or B100) parameter/response pairs as a fixed mathematical input.
   The initial-16 transform remains frozen. The existing production recovery
   fitter is called with its declared final training-version seed and an
   explicitly empty warm-start history. The existing production physical-MAE
   recommender uses the actual terminal context/seeds. This is a newly
   reconstructed cold-fit reference, **not** the historical final GP, a
   reconstructed warm trajectory, or evidence that the runner selected those
   archived points in the current environment. No response is assigned to a
   different proposed candidate.

The B40 duplicate-recovery fixture reconstructs the stored recovery operands:
the original duplicate is an indexed stored evaluated point; the raw pool is
regenerated and its identity checked; rankings and identity metadata are read
from the archive. The production recovery function must return the recorded
unevaluated raw candidate. No historical GP parameters are fabricated. Existing
runner tests additionally protect budget, failure, RNG and duplicate-control
paths with simulator-independent test adapters.

## Gate frozen before refactoring

The harness is `infrastructure/truba/refactor_reference.py`. A baseline-to-
baseline repeated capture must first establish local numerical repeatability.
The comparison then requires:

- Matching numerical-environment fingerprints: Python and numerical packages,
  CPU/backend builds, runtime threadpools and explicit Torch threads (20/1).
- Exact mathematical output hashes and tensor values, including frozen
  transforms, standardized data, initialization/fit parameters, posterior and
  acquisition quantities, candidates, recovery decisions and recommendation.
- Exact seeds, observation identities, option identities, budgets and accounting.
- Preserved ambient Python/NumPy/Torch RNG state, valid posterior contracts and
  zero real physical-backend calls.

Wall times, scheduler job IDs, hostnames, source paths and source commit hashes
are provenance, not mathematical equality targets. A refactored commit is
expected to have different source hashes. Comparison recomputes the artifact's
scientific and environment hashes and rejects stale/tampered hashes.
Runtime-library enumeration is an unordered collection: compare the complete
library records after sorting canonical JSON, retaining every path, version,
thread count and duplicate record. The comparator verifies the original
ordered-v1 signatures before applying this v2 comparison to old packets.
It does not normalize tensor order, floating values or scientific policies.

**No floating tolerance is introduced for this first operation-preserving
refactor.** If baseline repeats differ or a later mathematical rearrangement
needs a numerical tolerance, stop and define a justified, quantity-specific
rule with researcher review before implementing that refactor. Never adjust a
tolerance after observing a failed refactor to force acceptance.

Passing these checkpoint gates does not establish complete adaptive trajectory
equivalence after historical replay has stopped. Changes to the unseen
sequential feedback path require an additional applicable regression gate.
Scientific performance claims and new simulator campaigns remain separate.

## Bounded qualification and execution

The supplied endpoint is an ARF login node. Only metadata discovery, source
preparation and scheduler operations run there. Official TRUBA guidance sends
numerical work through Slurm:
[connection](https://docs.truba.gov.tr/1-kaynaklar/arf/arf_baglanti.html),
[workflow](https://docs.truba.gov.tr/2-temel_bilgiler/ybh-akis.html),
[queues](https://wiki.truba.gov.tr/1-kaynaklar/arf/arf_kuyruk_bilgisi.html).
Live `scontrol` policy and `sbatch --test-only` take precedence when online
pages disagree about allocation geometry.

Use the tracked `infrastructure.truba.lifecycle prepare-validate` command,
profile `infrastructure/truba/refactor_reference_v1.env`, an explicit
transport-only `--ssh-config` if needed, and fresh user-owned remote/output
roots. The lifecycle performs only stdlib validation on the login node. It
does not submit a job. No password, token or private-key contents belong in
profiles, source, environment dumps, logs or receipts.

The qualification is capped at one CPU node, one task, 20 CPUs and 8 GiB memory,
without GPUs. The initial job used 3 minutes 7 seconds. A single diagnostic
repair attempt is allowed at most 11 minutes, keeping combined allocated time
below the original 15-minute budget; no further jobs are part of this task.
Use an authorized
account/partition after live scheduler validation. The job creates a separate
detached baseline worktree, runs selected baseline tests, captures B40 twice,
checks equality, and separately attempts strict B40 historical replay. If time
permits, it captures B100 twice. It submits no child jobs or array/campaign.

The first attempt retained a serialization failure in the new checkpoint
writer (`GPFitDiagnostic` is serialized through production `FitRecord`), and
a strict historical-metadata comparison treated Python tuples differently
from their JSON arrays. The repairs affect diagnostic serialization only.
Canonical JSON container comparison preserves all numeric values; exact
candidate/context checks and numerical regression hashes are unchanged.

The existing user-owned numerical environment is read-only. Its core packages
match the reference pins; Python 3.11.17 differs from historical 3.11.15 and is
recorded as a patch-version difference. SBI is absent: this qualification is
for MUTACO, not SBI training or all paper workflows. The harness fails on a
missing/mismatched core numerical package; it records optional SBI absence.
No environment or shared installation is modified.

Private originals retain JUnit/status, package freeze, raw packets and scheduler
records. Shareable derivatives retain numerical findings and original SHA-256
bindings while omitting operational locations. New GP values remain reconstructed
references, not original historical states.

## Completed B40 qualification

The retained result is
[`benchmarks/reproducibility/shareable_milestone_v1/manifest.json`](../../../benchmarks/reproducibility/shareable_milestone_v1/manifest.json).
Jobs 6852386 and 6852464 each used 187 seconds (374 seconds combined) on
Intel Xeon Gold 6148 CPU compute nodes. Both jobs have Slurm state
`FAILED`; the original test suite had **169 passes, 8 skips, and 1 failure**
in each. The known NEI Cholesky success fixture is exactly singular and is
unchanged. No test or production factorization policy was fixed or disabled.

The repaired diagnostic produced two B40 packets with exactly identical
scientific hashes and zero physical-backend attempts. Its on-node comparator
incorrectly rejected the environment because NumPy/SciPy library records were
enumerated in different orders. Their full contents were identical as a
collection. The v2 comparator verifies both original packets and passes their
same-environment/exact-science gate locally without recalculating any GP state.
The original failed comparison and all first-attempt writer/replay failures
remain in the preserved job exports. No tolerance was changed.

Strict B40 target-16 replay separately **passed all 40 observations**, with
25 committed fits, 24 proposals, one terminal recommendation and preserved
ambient RNG. It checks exact candidates and contexts before returning each
stored response, including the archived duplicate recovery. Historical GP
parameter snapshots are still unavailable; new fits are reconstructed values.

B100 was skipped after the false environment-order rejection, and the two-job
allowance is closed. There is no B100 qualification claim. Further allocation
and broader gates require a new bounded execution decision. The existing SBI
absence and baseline NEI failure remain explicit scope limitations.

## First proposed refactor

After the applicable TRUBA gates pass, extract the frozen-response preparation
block in `src/mutaco/orchestration/runner.py::run_sequential` into one named
helper. Keep the stack order, mean/std reduction expressions, floor/ddof,
clones, device/dtype, identities and exception/event boundaries unchanged.
Leave the surrounding simulator and GP/acquisition loop intact. The main path
then reads directly as initial design → observations → frozen response state →
fit → propose → evaluate → final recommendation.

Its acceptance gate is exact first-adaptive and fixed-data checkpoint equality
against the retained baseline artifacts on the same qualified environment,
plus exact full B40 historical replay and the existing transform, RNG and
runner success/failure/accounting tests. Preserve the documented baseline
failure set; any additional failure blocks acceptance. This B40 gate is ready
for the proposed helper extraction; broader feedback-path work needs its own
coverage, including B100 where applicable.
This proposal changes no GP priors, terminal dispatch or global-best selection.
It is not implemented by the qualification task.

## B100 extension of the same exact gate

The next case is B100 target 0, archive record /records/0: 16 initial and
84 adaptive observations, independent of B40. The original scientific source
remains aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3; the historical campaign source
is 649235bcb7e8773f5f4b0425ad111314d7d3b1f9. The accepted refactor source is
the published clean milestone 242fac08c1c3dc0041651a731c20ed228969ed6a.

The single CPU allocation requested 20 CPUs, 8 GiB and a 20-minute wall limit.
The measured peak is reported below rather than inferred from that request. All calculations
run under Slurm in the existing qualified environment, with no installation
changes. The original baseline is tested and replayed first. A first candidate
or effective-context mismatch halts replay before the stored response is
returned; any failed reference gate prevents execution of the refactored source.

A successful reference must consume 100 archived observations and record
85 committed fit/update events, 84 proposals and one terminal recommendation.
The accepted source must preserve all scientific diagnostics and checkpoint
hashes exactly under the same environment. First-fit and conditional final-data
cold-fit packets retain their existing labels; no original historical GP
snapshots are inferred. Existing timer/execution exclusions and unordered
runtime-library metadata are the only declared representation normalizations.
No numerical tolerance, seed, policy or evaluation budget is changed.

The known NEI fixture stays in both test suites. Its pytest/Slurm failure is
separate from the exact scientific gate. Raw evidence stays private; derived
B100 artifacts retain numerical values and original/new SHA-256 mappings.
No B40 reference or previously accepted evidence is overwritten.

After B100 passes, a proposed next extraction is the option-validation part
of the pre-run try block. Inputs would be the existing scientific intent and
already constructed seed deriver; outputs would be normalized GP/acquisition
settings. Keep capability capture, seed-deriver construction and the outer
failure handler in the runner. Preserve strict metadata conversion order,
beta-policy response-dimension checks, GP/proposal config validation, the
training-version 0 / raw-round 0 seed coordinates and recommendation-policy enum
validation. Preserve configuration-vs-simulator-definition status selection,
failure explanations, zero-attempt accounting and RUN_STARTED/RUN_FINISHED
boundaries. No fit, proposal or simulator evaluation occurs in this validation
step. This is a proposal only; no second extraction is implemented here.

### Completed B100 outcome

Job 6853779 completed both scientific comparisons in 598 seconds. Original and
accepted implementations each passed all 100 archived observations, with 85
fit/update events, 84 proposals and one terminal recommendation. Checkpoint
hashes and full retained mathematical diagnostics matched exactly; RNG and
effective-context checks passed with zero physical simulator attempts.

The baseline suite recorded 169 passes, 8 skips and the known NEI fixture
failure; the accepted source recorded 175 passes, 8 skips and the same failure.
Slurm state FAILED/exit1 therefore remains separate from scientific acceptance.
All new GP values are reconstructed, not original historical checkpoint states.

The recorded 8 GiB memory request was below Slurm's reported 13927916 KiB
MaxRSS (about 13.3 GiB). This result does not certify 8 GiB as a sufficient bound.
Review accounting/enforcement and a request of at least 16 GiB before another
B100 allocation. No additional allocation is authorized by this result.

Portable exact mathematical payloads, original/new hashes, environment summary
and the resource discrepancy are retained in
benchmarks/reproducibility/b100_regression_v1/. Raw captures, original manifests
and logs are privately preserved; the earlier B40 artifacts remain unchanged.
This gate supports the next separately approved option-validation extraction,
not a broader fitting/acquisition/terminal rewrite. No second extraction has
been implemented.

## Second controlled extraction: pre-run options

The option-validation statements move together, without reordering, to
`runner._validate_pre_run_options(intent, seed_deriver)`. They materialize
strict GP settings, materialize strict acquisition settings, validate optional
beta metadata and response dimension, construct the training-version-zero GP
configuration, construct the sequential-round-zero proposal configuration,
and validate the recommendation enum. The two strict settings documents are
returned for the existing fit/proposal closures. Configuration construction
still derives the same named seeds and performs the same validations, including
the proposal constructor's own beta validation. No random draws are added.

Capability capture/matching and seed-family selection remain in the runner,
before the helper. The existing catch remains outside it: metadata-contract,
value and type errors retain the first exception's class/message and the
existing snapshot-dependent status selection. Validation failures retain the
captured capability snapshot and run identities, emit only RUN_STARTED and
RUN_FINISHED, and commit no design, observations, transform, fit, proposal or
recommendation. All physical/logical counters remain zero except the planned
budget; total elapsed time remains measured. Characterization tests execute
these original success/failure paths before extraction, including float32/64,
diagnostic seed families, multiple-invalid-option precedence and all three
caught exception classes.

The candidate-only workflow is `readability_regression.sbatch`, prepared through
the tracked lifecycle and `readability_regression_v1.env`. It checks immutable
private input hashes from `readability_reference_inputs.json`, preserves the
accepted test outcomes, and runs B40 before B100. Both strict historical replay
and the unchanged numerical checkpoint comparator must pass for B40 before
B100 starts. Saved original-source references are reused without rerunning the
baseline. One 20-CPU, 20-GiB, 20-minute CPU allocation is the authorized bound;
live partition policy and scheduler dry-run must accept it before submission.
The physical-backend tripwire remains active. Known NEI pytest failure and
scientific acceptance retain separate outcomes. No tolerance is introduced.

## Third controlled extraction: initial observations

The accepted initial loop is the boundary for _collect_initial_observations.
Initial-design generation and INITIAL_DESIGN_READY remain in the runner. The
helper receives the design, capability snapshot, seed deriver, runner config,
original observation list/counters, start time and callback state. For each
initial index it calls the unchanged _evaluate with the contiguous design row,
matching observation ID, INITIAL phase and matching logical/phase indices.
A successful record is appended before committed_observations and
initial_observations are incremented. On failure, collection stops immediately
and returns the existing status/explanation to the runner's finish closure.
Earlier records and all physical-attempt accounting remain committed. A tuple
or None transports only that failure; no observation structure is replaced.

Retries, context construction, seed policy, capability/budget precedence and
attempt events remain entirely inside _evaluate. Retry attempts reuse the
scientific realization with a different attempt index. Callback errors remain
isolated. Failed runtime/metadata/tensor attempts retain the existing event
asymmetry, and failed logical observations are not appended. No failed or
partially completed collection reaches frozen-response preparation. Successful
collection preserves response/ID order for the existing frozen transform.

Twenty-six pre-extraction characterization cases cover both dtypes, all three
evaluation-seed policies, retries, partial failure, competing limits, capability
drift and progress failure. They use a deterministic fake response adapter with
the physical-backend tripwire active. Logical checked evaluations of fake or
archived responses are counted by the runner; they are not physical simulator
calls. The existing candidate-only B40-then-B100 workflow is reused with an
explicit initial_collection_reference_inputs.json specification. Original
mathematical reference hashes and numerical comparators remain unchanged.

## Fourth controlled extraction: GP training data

The preparation block shared by every fit_version call moves verbatim into
_prepare_gp_training_data(observations, intent, transform). It reads committed
records in order, checks that standardized responses exist, stacks theta and
standardized responses with the existing contiguous calls, and constructs the
same SurrogateTrainingData with parameter spec, observation IDs and frozen
transform ID. Returned stacks own storage independently of observation rows
and other training versions. No conversion, clone, detach, mutation or random
draw is added. Standardization is already complete before this boundary.

The first fit receives 16 records in the regression cases; updates include each
newly committed standardized record. Partially evaluated/unstandardized rows do
not enter this path under the ordinary lifecycle. Legitimate duplicate caller
parameter rows remain included with distinct indexed observation IDs. The
missing-standardization check is retained as defensive behavior, without tests
that corrupt internal observations to make it reachable. A finite response can
overflow frozen standardization on an update: the existing data constructor
then rejects nonfinite training values after that observation is committed.
This real path preserves GP_FIT_FAILED, its message and partial state unchanged.

GP_FIT_STARTED remains before the existing try; preparation exceptions remain
inside its existing handler, before fit configuration/seeds or fitting execute.
No fit-finished event/counter is committed for a preparation exception. Task
expansion, normalization, likelihood/covariance inputs, model construction,
fitting/recovery, warm state and posterior policies remain in their existing
functions. Characterization wraps those real functions without replacing them.

The existing candidate-only workflow uses gp_training_data_reference_inputs.json
for the ten new cases and saved accepted test outcomes. Numerical reference
hashes and engines are unchanged. First/final tensor checkpoints and every
25/85-fit training-data identity remain exact comparison targets in the saved
B40/B100 packets. No original reference is regenerated.

## Fifth controlled extraction: sequential observation commitment

Only the post-evaluation standardization/commit block moves into
_standardize_and_commit_sequential_observation. Proposal handling, duplicate
recovery, observation-ID derivation and _evaluate remain in the runner. That
evaluation completes attempt accounting and attempt events before the helper
is called. A failed evaluation still returns before any standardization.

The helper applies the existing frozen transform, then performs the existing
replace(record, response_standardized=standardized.detach().clone().contiguous()),
appends the replacement record and increments committed_observations followed
by sequential_observations. The raw evaluated record remains unstandardized;
its physical tensor, theta, context and attempt records are shared by replace,
while the standardized tensor receives the original independent clone.
The next fit_version call remains after successful commitment. No helper event,
callback, RNG draw, precision conversion or additional validation is introduced.

Only the transform call remains within its original exception handler. Its
exception class/message is returned to the existing finish closure as
RESPONSE_TRANSFORM_FAILED. Clone/replace/append/counter operations stay outside
that handler, so resource failures there still propagate without RUN_FINISHED.
Physical attempt success is retained even when logical commitment fails.

Twenty-one characterization cases execute before extraction and afterward:
multiple updates, all evaluation-seed policies, ordering/ownership, retries,
evaluation-error precedence, callbacks and real production duplicate recovery.
Allocation failures are explicitly injected at the existing transform or commit
clone operations; no invalid internal observation is fabricated. A controlled
duplicate-proposal fixture uses the real raw pool, identity and recovery code
with a fake simulator. It does not alter historical replay or return an archived
response for a substituted candidate. The existing overflow-to-GP-failure path
also remains protected by the fourth extraction's tests.

The unchanged candidate-only workflow uses sequential_commit_reference_inputs.json
for accepted test outcomes and the 21 additions. B40 gates B100; original raw
reference inputs, mathematical hashes and exact numerical comparisons remain
unchanged. No original baseline fit is rerun.

## Sixth controlled extraction: record returned GP fit diagnostics

The cohesive boundary is the unconditional record/account/report block after
numerical fitting returns: append FitRecord, count executed recovery attempts,
accumulate returned fit time and validation-posterior calls, then emit
GP_FIT_FINISHED. _record_gp_fit_result receives the original diagnostic by
reference, version, record list, physical counters and callback state. It does
not decide whether a fit is usable and does not access the model or warm cache.

Fitting/preparation exceptions remain in the preceding try and produce no
returned record or finish event. The existing failed-status/missing-model
check remains after reporting. Physical success, warm-state retention/trimming,
logical training-version increments and return values stay in fit_version.
Record/report resource failures remain outside the numerical-fitting catch;
callback exceptions remain isolated. The event still precedes the current
fit's success/logical commits. All operation and floating-addition order stays
unchanged, as verified by whole-runner AST inlining.

Eighteen pre-extraction cases cover real successful graph recovery and bounded
warm history, start/finish callback failure, typed first/update failures, raised
fit calls, mixed executed/skipped recovery records and partial state. Explicit
controlled result doubles exercise defensive status/model inconsistency; they
are not fabricated historical GP states. A record-allocation fault verifies
that bookkeeping failure still precedes later failed-fit status selection.

Original reference packets contain fit records but no raw event stream. The
optional observation-only progress sidecar records candidate events during the
same strict replay. Expected GP metadata/order is explicitly reconstructed from
saved original FitRecords and the baseline emitter; both emitter AST hashes
are checked against aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3. Comparisons preserve
IDs, statuses/messages, versions and event boundaries, excluding only existing
elapsed_seconds fields. These are reconstructed expectations, not original
historical event logs or timestamps. Mathematical payloads/comparators are
unchanged, and no extra fit or baseline run is added. The sidecar is opt-in via
fit_result_reference_inputs.json; earlier workflow defaults remain supported.

## Software-test expectations after the NEI fixture correction

The current default test expectation is `corrected-nei-fixture-v2` in
`infrastructure/truba/corrected_test_reference_inputs_v2.json`. It binds the
NEI test file byte identity to fixture revision
`bd3859906a969d362e7d0496e87dcac404804285`. It requires pytest exit zero,
zero physical-backend attempts, a passing precision-promotion test and both
passing float32/float64 singular-input cases. No failing test is exempted.

Against a pre-correction reference, exactly the declared NEI failure-to-pass
transition and the two named singular cases are allowed. Other recorded
outcomes, including skips, must remain identical. Additional characterization
cases remain explicitly bounded by the selected profile's count and prefixes;
they must all pass. A corrected full-suite reference can instead be used with
`expected_added=0` to require identical test identities and outcomes.

`historical-nei-failure-v1` preserves the old verdict and output format. Older
explicit input manifests without a test-expectation version select that legacy
mode; its original NEI failure remains a historical fact. Direct historical
callers must explicitly request `HISTORICAL_TEST_EXPECTATION` and their original
addition count/prefixes. `b100_regression.test_gate` is also a historical gate
for its fixed original/first-refactor source revisions; it is not the current
corrected-development expectation. Historical reports and manifests are not
rewritten to describe a passing suite.

Both modes use the same metadata-only gate locally and on TRUBA. A test-only
Slurm run compares the complete final collection against the qualified local
collection, and checks the core-only projection against the original saved
reference with the v2 profile. Projection selects actually executed cases; it
is not a second test execution. New gate tests explain any collection increase.
No B40/B100 replay, mathematical comparator, tensor tolerance, scientific seed,
fitting/acquisition policy or physical-simulator rule changes with this version.
The closure evidence is in `benchmarks/reproducibility/test_quality_closure_v2/`.

## Raw screening and branch-selection extraction

The seventh extraction moves only the existing scoring try-body in
`acquisition/proposal.py` into `_screen_and_rank_raw_candidates`. Its inputs
are the already evaluated posterior (mean [H,M], covariance [H,M,M]), the
standardized target [M], and the existing ProposalConfig. It returns the eight
existing branch/reason, score [H], index [K], direction, finite-count and ranking
values, retaining tensor references and autograd ownership. A plain return tuple
is the only new container. The surrounding try/except, posterior evaluation,
empty-score failure, restart dispatch, final selection and duplicate recovery
remain explicit and unchanged.

The geometry still computes squared residuals by square/sum and the exploitation
by vector norm in their original functions/order. The strict margin comparison,
safe logarithm, invalid infinity scores and stable finite-index sort are
unchanged. Exploration retains both registered uncertainty policies and the
existing diagnostic normalized-miss/radius variants, including their original
norm-then-square expression and minimization direction. No expression is
rearranged to match an algebraic equivalent or the manuscript. Ranking still
performs its existing bounded detached scalar extraction before optimization.
No posterior or simulator call, random draw, copy, detach or dtype/device change
is introduced by the helper. Its exceptions propagate into the same outer
scoring-error result, preserving first-failure precedence and accounting.

Thirty-six CPU float32/float64 cases characterize real objectives on controlled
posteriors: both branches, ties, exact/adjacent thresholds, the feasibility
boundary, zero-radius and overflow/nonfinite scores, diagnostic policy variants,
input versions, score/optimizer view ownership, gradients, seeds/RNG, and typed
posterior/scoring failures. Downstream failed-restart doubles expose unchanged
starts and raw fallback without implementing an optimizer. Existing tests retain
real-refiner coverage. Before/after captures compare exact bytes, not a newly
relaxed tolerance. CUDA coverage remains the existing optional coverage.

The v2 software expectation uses the complete successful pre-extraction test
inventory, including these cases, with zero added/missing/changed outcomes.
`raw_screening_reference_inputs_v2.json` binds that inventory and the original
B40/B100 packets separately. The private test driver extends only the existing
worker's test-path list to include the full import/metadata suite; all numerical
replay and checkpoint work reuses the existing production/reference engine.
The one 20-CPU, 20-GiB, 20-minute allocation gates B40 before B100, retains the
strict candidate/context oracle and simulator tripwires, and does not rerun the
original baseline. Full raw evidence stays private; portable evidence belongs in
`benchmarks/reproducibility/raw_screening_v1/` after successful validation.

## Refined-candidate selection extraction

The eighth extraction moves only the ordered selection loop into
`_select_nonworsening_refined_outcome`. Inputs are the existing list of restart
outcomes, bounds [2,P], raw-best scalar score, and existing minimization flag.
The result is the original winning outcome object, or None. It does not copy,
detach, reorder or mutate optimizer outputs, initialize a model, consume RNG,
or call a posterior or simulator. The main proposal sequence now reads raw
screening -> unchanged restart execution -> select a non-worsening refinement
-> unchanged candidate/fallback handling -> duplicate diagnostic and identity.

The exact short-circuit order is FAILED status, missing candidate, missing score,
nonfinite score, nonfinite candidate, then bounds violations. Bounds are
inclusive. Non-worsening uses <= in minimization and >= in maximization;
replacement uses strict < or >. An equal score retains the earlier outcome in
raw-rank/restart order. TERMINATED outcomes remain eligible when otherwise valid.
The same raw-best score is used throughout. No accepted numerical policy is
corrected or inferred from the manuscript. Fallback classification, explanation,
candidate aliasing, duplicate handling and identities remain outside the helper.
The existing refinement timer still stops before selection. Exceptions still
propagate without a new handler. Inlining the helper restores the original AST.

Before extraction, 114 CPU float32/float64 cases observe executed winner and
comparison transitions in the existing Python loop. Observation uses temporary
line tracing, with labels found from the current AST, and restores the prior
trace function. The observer does not compute selection decisions. Scripted
optimizer outputs exercise real screening, three existing branch/direction
combinations, equality/improvement/worsening, exact ties, one-ULP score changes,
finite/bounds guards, missing results, FAILED and TERMINATED statuses, raw
fallback, unchanged duplicate diagnostics, input versions, tensor ownership,
RNG and exception precedence. Malformed-result cases characterize defensive
guards, not a claim that current optimizers normally emit them. Existing tests
and full archived replay retain real-optimizer coverage.

`refined_selection_reference_inputs_v2.json` binds the successful complete
pre-extraction test inventory and the unchanged original scientific packets.
The v2 gate requires zero failures and no missing/extra/changed outcomes, with
all existing skips preserved. Private wrappers change only test-path selection
and optional characterization-output locations. The tracked numerical harness,
exact comparators, simulator tripwires and baseline packets are unchanged.
One bounded 20-CPU, 20-GiB, 20-minute allocation must pass tests and B40 before
B100. Original baselines are not rerun. Raw evidence remains private; portable
derived evidence belongs in benchmarks/reproducibility/refined_selection_v1/
only after successful validation. No cross-platform equality is required.

## GP optimizer-result diagnostic summary extraction

The ninth extraction is confined to `_summarize_fit_optimizer_result`: the
existing post-handler optimizer report conversion, fixed-conditioning report,
objective improvement and covariance-record attachment. Inputs are the existing
observation list, optimize flag, initial objective, GPFitConfig and two covariance
records. It returns the same fresh dictionary or None, preserving nested aliases.
The block remains after the RNG fork and exception classification. Its uncaught
conversion/allocation errors keep that boundary. No model, parameter, gradient,
optimizer, posterior, warning context or recovery operation moves into it.

The actual fitting sequence remains identity/seed selection -> fork_rng/model
setup -> compatible warm state -> registered-prior ExactMarginalLogLikelihood
initial objective -> original recording optimizer and callback snapshots ->
parameter/covariance validation -> seeded midpoint posterior validation ->
optimizer-result summary -> attempt/fit diagnostic construction. Nested snapshot
capture stays in place: it owns model clones and restoration inside the original
warning/exception/RNG scopes. Recovery remains warm/cold/four-coordinate-restarts/
fixed-conditioning under the existing guard and selection policies. Priors,
positive-index covariance, task floor, jitter, precision and the 12-iteration
reference limit are unchanged. This is not prior-free MML or a convergence change.

Thirteen cases start from the actual 16 checked archived B40 target-16 inputs and
production configuration, intercepted before fitting. They use real setup,
initial objective, priors/constraints, gradient and posterior operations. One
uses the real library optimizer; controlled optimizer reports cover termination,
nonfinite values, callback/optimizer failures, invalid parameters/posteriors,
warm and fixed conditioning, successful four-restart recovery and exhaustion.
Controlled reports are diagnostic fixtures, not historical optimized GP states.
Capture numerical bits and identities before/after; require exact agreement,
input/warm-state ownership and ambient RNG preservation. Existing full-suite
float64/interface cases remain; these archived inputs use production float32.

A pre-existing nested-scope issue is deliberately characterized, not corrected:
`recording_optimizer` assigns a local last_valid_covariance on its failure path;
its returned outer fit diagnostic still reports None despite computing restored
covariance metrics. The GP file was byte-identical to aa7f4fe before this work;
Python symbol-table inspection establishes the shadowing. This diagnostic
availability issue needs a separate reviewed change, not a readability fix.

The new v2 input profile binds the complete successful pre-extraction inventory
and unchanged scientific packets. No missing/extra/changed outcomes or failures
are accepted. One bounded CPU allocation gates the final suite and B40 before
B100, with the original exact comparators and physical-simulator tripwires.
Reconstructed GP checkpoints remain distinct from unavailable historical states.
Private evidence reuses checksum-bound sealed baseline inputs and source bundles
rather than duplicating them; new source bundles may require the preserved base
bundle. Portable derived results belong in gp_optimizer_diagnostics_v1/ only
after both exact scientific gates pass. No push is authorized for this task.
