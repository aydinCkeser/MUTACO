# Raw acquisition screening: seventh controlled refactor

Accepted: B40 target 16 (/records/48) and B100 target 0 (/records/0) replayed
all 40 and 100 archived observations. Their complete scientific trajectory and
mathematical checkpoint hashes equal the immutable unrefactored references.
No proposed candidate or context was replaced to obtain an archived response.
No tolerance, scientific policy, seed or reference value changed. Physical
simulator attempts/calls: zero.

The extraction moves the existing 44-line scoring block into
`_screen_and_rank_raw_candidates(raw_posterior, target, config)`. It evaluates
the existing barrier objective, orders finite scores, selects barrier or
exploration (including registered diagnostic variants), and builds raw rankings.
The plain eight-value return retains tensors and ranking objects. The main
proposal path now exposes posterior evaluation, screening, refinement and final
selection as separate scientific stages. Posterior calls, the surrounding
error handler, empty-score failure, optimizer/restart inputs, final selection,
fallback and duplicate recovery remain in place. Inlining the helper gives the
identical original module AST; its arguments and return/unpack order also match.

Before production edits, 36 CPU float32/float64 cases passed on source 677b619.
The same cases passed afterward, with 36 byte-identical captured states. They
cover barrier and uncertainty branches, stable ties, exact/adjacent thresholds,
trace and normalized-miss diagnostics, zero-radius/overflow/nonfinite scores,
empty finite pools, ownership/views, gradients, seeds, ambient RNG and typed
failure precedence. Controlled downstream restart failures expose the exact
optimizer starts and raw fallback; existing tests cover the real refiners.
An initial model-double name-resolution mistake was corrected before the
qualified unchanged-production run; it was not a production failure.

Full local before/after and final TRUBA suites each passed 396, skipped 8 and
failed 0. All 404 identities/outcomes match; the prior 368 cases are retained
and exactly 36 passing screening cases were added. No skip changed. Final post-evidence metadata/lifecycle checks also passed all
75 cases with zero physical attempts. The v2
zero-failure gate uses that complete pre-extraction inventory, pins the corrected
NEI fixture, and rejects any missing/extra/changed outcome. The historical old
NEI failure remains preserved in old evidence; it is not accepted for this code.

One scheduled CPU job (6860290) used 20 CPUs, 20 GiB and a 20-minute
limit on Barbun. It finished COMPLETED, exit 0:0, in 416 seconds, with
reported batch MaxRSS 12696676 KiB (12.108 GiB). Pytest exit and scientific
exit were independently zero. B40 gated B100 in the same allocation. The saved
baseline was not rerun and there was no automatic resubmission.

B40 preserves 25 fits/updates, 24 proposals and one recommendation; B100 preserves
85 fits/updates, 84 proposals and one recommendation. All training identities,
frozen transforms, available posterior/GP checkpoints, raw rankings, branch
choices, candidates/contexts, RNG, recovery, terminal results and logical counts
match. Fit-event order/boundaries match 50 and 170 expected events. Those expected
events are reconstructed from saved original fit records and the baseline
emitter, not original historical event logs. Final-data checkpoints are the
existing conditional cold-fit fixtures, not invented historical warm GP states.
Existing wall-time/provenance exclusions and unordered library enumeration are
the only comparison normalizations. No cross-platform equality claim is made.

The exact executed source snapshot was 8cdb6b19076044fab649469bc99e16800a8d45e1, descended from
677b619561abfb7e6d8eca4ddaa967de3e866c7b. Its four source/test/contract/profile
files match the final files byte for byte. Only portable evidence and project
state were added afterward. The tracked lifecycle prepared the source and
validated zero-call setup. A private driver expanded only the test-path list;
all replay/checkpoint calculations reused the unchanged tracked harness and
production algorithm. The driver, full captures, logs, original inputs and
source bundles are retained privately with hashes. This portable summary is
derived, not byte-identical to those raw packets. Strict environment checking
still requires the original private packets, including their library paths.

Verify portable integrity with `shasum -a 256 -c SHA256SUMS` in this directory.
Raw-file SHA-256 values are in raw-evidence-index.json; derived-file hashes are
in SHA256SUMS and manifest.json. The separately sealed private root also binds
the accepted source bundle and final manifest. Previous evidence was untouched.
Development remains on ege/repro-refactor; no GitHub push or PR update occurred.

Next recommendation only: inspect the existing post-refinement valid,
non-worsening best-outcome selection as one mathematical decision boundary.
Characterize finite/bounds filtering, comparison direction, raw-best threshold,
strict improvement and exact-tie order before considering extraction. Keep
optimizer internals, fallback and duplicate handling explicit. Do not change
the separately identified global-best-raw or terminal policies. No next refactor
was implemented. Coverage remains CPU; eight established optional skips remain.
