# GP optimizer-result diagnostics: ninth controlled refactor

Accepted: the complete B40 target-16 (40 observations) and B100 target-0
(100 observations) strict replays and saved scientific checkpoints match exactly
in the qualified TRUBA CPU environment. Pytest, scientific checks and Slurm all
pass independently. Physical simulator attempts/calls: zero. No GitHub push or
PR update occurred; the remote remains 7422a0c.

The safe boundary is the original 29-line post-handler optimizer-result summary,
now `_summarize_fit_optimizer_result`. Six existing inputs produce the original
fresh report dictionary or None: last observation conversion, fixed-conditioning
report, initial/final objective improvement and covariance-record attachment.
Dictionary/reference ownership and uncaught error semantics are preserved.
Inlining restores the entire original GP module AST. Fitting still visibly
separates model/seed setup, optimization, validation and diagnostic construction.

Nested recording/callback capture was deliberately left in place. It snapshots
model state, handles optimizer/caller failures, reconstructs diagnostics under
warnings and RNG contexts, and passes callback/gradient state across those
boundaries. Extracting it would require substantially more state plumbing.
No numerical model/closure, parameter registration/mutation, random draw,
warning/gradient context, restoration, posterior or recovery operation moved.
Priors, ExactMarginalLogLikelihood objective (including registered priors),
positive-index/task-floor constraints, jitter, covariance precision and the
accepted 12-iteration limit are unchanged. This is not prior-free MML.

Thirteen cases use the actual 16 archived B40 initial observations and production
preparation/configuration, stopping strict replay before fitting or proposing.
Real setup, prior/constraint registration, initial objectives, gradient flow,
posterior values and ownership are exercised. One case runs the real 12-iteration
library optimizer. Controlled reports cover success, iteration exhaustion,
nonfinite results, failures before/after callbacks, caller exceptions, invalid
parameters/posteriors, warm/fixed conditioning, all four seeded recovery attempts
and recovery exhaustion. Scripted reports are not historical GP states or
convergence measurements; their optimizer-call counter counts the scripted stub.

All 13 cases passed before and after extraction, with byte-identical serialized
numerical/identity states. Training/warm state, parameter/gradient ownership,
model/fit IDs and ambient RNG were preserved. Full local suites before/after
and the final cluster suite each passed 523, skipped the same 8 and failed 0.
Final post-evidence metadata/lifecycle checks also passed all 75 cases with zero
physical attempts. All 531 case identities/outcomes agree, including the previous 518 cases and
13 new passing cases. The unchanged v2 software gate requires zero failures;
old historical NEI failure evidence remains untouched and is not an exemption.
Existing float64/interface tests remain; these archived cases use float32.

A pre-existing diagnostic issue was localized and preserved: on callback-path
optimizer failure, `recording_optimizer` assigns a local last_valid_covariance.
The outer returned report remains None despite separately computing restored
covariance diagnostics. Python symbol-table inspection and byte identity to
original aa7f4fe establish that this predates the refactor. Tests explicitly
protect its current output. No diagnostic repair or scientific policy correction
was made; a separate researcher-reviewed decision is needed for any fix.

One bounded job (6860709) used 20 CPUs, 20 GiB and a 20-minute limit.
Slurm completed, exit 0:0, in 444 seconds; reported batch MaxRSS was
13021380 KiB (12.418 GiB). The final suite gated B40, which gated B100.
There was no original-baseline rerun or automatic resubmission.

B40 retained 25 fits/updates, 24 proposals and one recommendation; B100 retained
85, 84 and one. Training/frozen-transform identities, GP initialization and
available parameter/objective/covariance/posterior records, acquisition decisions,
seeds/RNG, contexts, duplicate recovery, terminal results and accounting match.
Fit-event order/boundaries match 50 and 170 expected events. Those expectations
and GP states are reconstructed from the saved original implementation, not
invented historical intermediate states. Conditional final-data cold checkpoints
remain distinct from historical terminal warm GPs. Existing wall-time/provenance
exclusions and library-enumeration ordering are the only normalizations.

Executed snapshot: 05d9a9595c1fe21e5856e534ddafc2fda294d307, based on 68435659d05d12280c396045c42a15eefaf7c84e.
The four executed source/test/profile/contract files match the final files;
only evidence/state metadata was added afterward. The tracked zero-call lifecycle
prepared the source. Private wrappers change only test selection/capture output;
the numerical driver and exact comparators are unchanged. Previous raw baseline
packets and the full base source bundle are reused through checksum bindings.
New private source deltas require that preserved base commit/bundle. New logs,
captures and deltas are checksum-sealed privately, avoiding repeated raw baseline
copies. These portable files are derived, not byte-identical raw packets.
Verify them with `shasum -a 256 -c SHA256SUMS`. Exact environment acceptance still
requires the original private path-bearing packets.

Next priority: documentation and submission-readiness review, including an
updated GP phase map and reproducibility instructions, rather than more small
core extractions. The coupled callback/recovery scopes and the shadowed diagnostic
need an explicit separate review before any further GP change. No such work,
prior ablation, terminal-policy fix, manuscript edit or release was performed.
