# Refined-candidate selection: eighth controlled refactor

Accepted: B40 target 16 (/records/48) replayed all 40 observations; B100 target 0
(/records/0) replayed all 100. Complete scientific trajectory and checkpoint
hashes equal the preserved unrefactored references in the same qualified TRUBA
CPU environment. No candidate/context substitution, tolerance adjustment,
policy change or new physical simulator attempt/call occurred.

The reviewed milestone was pushed once, by normal fast-forward, from 6a6dfe8
to 7422a0c. The four outgoing commits and their intermediate contents passed
sharing review; 190 portable checksum checks across history and 725 private
backup-file checks passed. Draft PR #1 remained open, at the milestone commit.
The original private d7b5b12 branch was not merged. The eighth extraction remains
local and was not pushed. No new development branch or PR was created.

`_select_nonworsening_refined_outcome` contains the original 36-line loop. It
takes existing restart outcomes, bounds, the raw-best score and direction,
returning the original winning outcome or None. It rejects FAILED, missing,
nonfinite and out-of-bounds results in the same short-circuit order. Inclusive
raw-score equality is eligible; only strict improvement replaces a winner.
Ties retain earlier raw-rank/restart order. A valid TERMINATED result remains
eligible. Bounds, dtype/device, tensor references and mutation versions are
preserved. No optimizer, posterior call, random draw, mathematical expression,
clone/detach or reduction changed. Inlining restores the original module AST.

The proposal path now exposes screening, unchanged refinement execution,
selection, fallback and duplicate/identity handling as readable stages. The
existing refinement timer still ends before selection. All fallback, error,
identity, duplicate and accounting statements remain in their original context.
No scientific defect was corrected or inferred from the manuscript.

Before extraction, 114 float32/float64 CPU cases passed on accepted production
source 7422a0c. All passed afterward; all 114 serialized states are byte-identical.
They observe executed winner/comparison transitions with a restored temporary
trace, rather than implementing a second selector. They cover improved/equal/
worse results, multiple winners, exact ties, one-ULP score thresholds, finite
and inclusive-bound checks, missing fields, FAILED/TERMINATED results, both
major branches plus registered diagnostic minimization, fallback classification,
duplicate diagnostics, identities, tensor ownership/versions, RNG and errors.
Malformed outputs are defensive-result fixtures, not claims about normal
optimizer outputs. Existing tests and strict replays exercise real optimizers.

Local full suites before/after and the final TRUBA suite each passed 510,
skipped 8 and failed 0. All 518 cases retain the same order/outcome, including
the previous 404 cases plus 114 new passing cases. No skip changed. Final post-evidence metadata/lifecycle checks also passed all
75 cases with zero physical attempts. The unchanged
v2 gate uses the full qualified pre-extraction inventory and rejects changed,
missing or extra outcomes. Corrected NEI promotion/singular coverage still
passes; no historical failure exemption is used for this revision. Temporary
wrapper import/extraction-script syntax issues were corrected before qualified
execution; neither changed scientific source or test expectations.

One Slurm job (6860507) requested 20 CPUs, 20 GiB and 20 minutes.
It completed with exit 0:0 in 339 seconds; batch MaxRSS was 11502096 KiB
(10.969 GiB). Pytest and scientific exits were independently zero.
B40 gated B100 within this one allocation. There was no automatic resubmission
and no unnecessary rerun of the original baseline.

B40 retains 25 fits/updates, 24 proposals and one terminal recommendation;
B100 retains 85, 84 and one. Frozen transforms, training data and identities,
available GP/posterior checkpoints, screening and restart diagnostics, selected
candidate identities/coordinates, contexts, seeds/RNG, recovery, terminal
results and accounting match exactly. Fit-event boundaries/order match all
50 and 170 reconstructed expected events. Original historical GP intermediates
and event times are not invented. Final-data checkpoints retain their explicit
conditional cold-fit meaning; they are not historical warm terminal GP states.
Predeclared wall-time/provenance exclusions and library enumeration order are
the only normalizations. This is not a cross-platform equality claim.

The exact executed snapshot was 8e774d238da7dacff5c7edce1b8b7aa038f74bbc, descended from 7422a0c.
The tracked zero-call lifecycle prepared it from a disposable clone of the same
existing development branch. Its four code/test/profile/contract files match
the final files byte for byte; only derived evidence and state were added later.
Private wrappers extend test-path/capture selection only; the scientific driver
loop and all exact comparators are unchanged. Full raw logs/captures, original
inputs and source bundles are retained privately with SHA-256 verification.
These portable summaries are derived, not byte-identical raw packets. Exact
environment checking still requires the original private path-bearing packets.
Verify portable files with `shasum -a 256 -c SHA256SUMS`; raw hashes are separately
indexed, and the final private root checksum file binds the accepted bundle.

Next recommendation only: review the stage boundaries of `fit_exact_multitask_gp`
(model setup -> seeded optimizer -> posterior validation), especially its nested
optimizer diagnostic capture, before choosing any extraction. Preserve all
callback, failure/recovery, warm-state, seed, prior and numerical policies.
The proposal flow now has meaningful scientific stages; another small reporting
helper would add less value. No next refactor was implemented. Coverage remains
CPU, with eight established optional skips.
