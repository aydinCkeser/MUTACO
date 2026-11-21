# Final corrected NEI software-test gate

The final, single TRUBA full-suite run passed **360 tests, with 8 unchanged
skips and zero failures**. Pytest exit: 0. Slurm job 6859583: COMPLETED,
exit 0:0, 96 seconds. Requested: one node/task,
2 CPUs, 4 GiB, at most 5 minutes. Reported batch MaxRSS:
383772 KiB (0.366 GiB).
No additional job or automatic resubmission occurred.

The previous qualified collection contained 346 cases (338 passing outcomes
and 8 skips, established by the prior full attempt plus corrected NEI follow-up).
The current collection contains all those identities in their original order
and adds exactly 22 passing test-expectation cases. No old case changed outcome,
no case disappeared and no new skip was introduced. Thus 360, rather than 338,
is the observed pass count. The final local combined run also passed 360 with
8 skips. Its scientific/import subset passed 285 with 8 skips; infrastructure/
metadata/lifecycle subset passed 75. Focused NEI/expectation verification passed
38. All eight NEI cases passed in the final cluster run.

Current expectation: `corrected-nei-fixture-v2`, the default CLI/profile and
function expectation. It pins the NEI fixture file to the bd38599 byte identity,
requires promotion and both singular-input cases to pass, requires pytest zero
and zero physical-backend attempts, and rejects unexpected failures, missing
cases, extra unregistered cases and changed skips. Added characterization cases
remain explicitly count/prefix bounded. There is no expected-failure exemption.

Historical expectation: `historical-nei-failure-v1`, selected explicitly for old
reports or by an explicit older unversioned input manifest. Its original verdict
and output format were reproduced exactly from preserved raw results. The same
old failing candidate was rejected by v2. Old reports are not rewritten. The
fixed original/first-refactor b100_regression gate remains historical.

The final full-suite gate compares every executed outcome with the locally
qualified corrected collection. A metadata-only core projection also passes the
real current CLI against the original checksum-bound test reference: 282 passes,
8 skips, the one declared NEI failure-to-pass correction, two named singular
cases and the 18 previously accepted fit-bookkeeping cases. This projection is
not a second test run or a historical replay. The local reference XML is a
derived identity/outcome inventory; the original raw JUnit is separately retained.

The source prepared through the tracked zero-call lifecycle was
408c2eee44e9b250a2606768804c8a616ba1848f, descended from starting local commit
bd3859906a969d362e7d0496e87dcac404804285. The executed four engineering/test/contract
files match the final files byte for byte. Only final evidence and project-state
metadata were added afterward. Development remains on ege/repro-refactor.
No new development branch, GitHub push or PR update occurred; the remote remains
6a6dfe85073475c14a2155d0527b98a218419a70.

Production algorithms, policies, priors, seeds, precision, the corrected NEI
fixture, B40/B100 mathematical equality functions and all previous references
remain unchanged. No B40/B100 replay or campaign was run. Physical simulator
attempts/calls: zero. The eight existing optional skips remain outside coverage.
Raw logs, environment records, both full JUnit and derived inventories, original
inputs, worker scripts and source bundle are preserved privately with hashes.
These portable summaries are derived evidence, not byte-identical raw packets.

Next recommendation only: inspect acquisition/proposal.py lines 1101-1144,
from MatchedBarrierObjective evaluation through `_ranking(indices, values,
margin)`. Its inputs are the existing raw posterior, standardized target and
ProposalConfig; outputs are branch/reason, score tensor, ordered finite indices,
margin, direction, finite counts and ranking diagnostics. Keep the raw posterior
call, its separate failure handler, the surrounding scoring exception boundary,
empty-pool failure, restarts, candidate selection and duplicate recovery explicit.
Characterize both branches and diagnostic policy variants, thresholds, ties,
nonfinite/empty scores, dtype/device, gradient ownership, RNG and posterior-call
counts before extraction. Preserve every formula, direction and tie rule; then
require exact B40 and B100 validation in the qualified scientific environment.
No acquisition or other production refactor was implemented in this checkpoint.

Verify these portable files with `shasum -a 256 -c SHA256SUMS`. Full raw recovery
uses the private bundle and original inputs, verifies their hashes, then repeats
the metadata-only v1/v2 and collection checks. Numerical references are unchanged.
