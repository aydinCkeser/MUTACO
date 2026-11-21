# GP returned-result bookkeeping regression

The reviewed initial milestone push published 415c9ea, 47c81dc and 6a6dfe8 by
normal fast-forward to ege/repro-refactor; existing draft PR #1 reflects 6a6dfe8.
Original private history and all prior evidence backups remain intact. No new
branch, force push, main merge, PR creation or visibility change occurred.
The new sixth extraction remains local and has not been pushed.

Inspection found a smaller cohesive boundary than all fit-result handling:
_record_gp_fit_result records every returned diagnostic, updates physical work
counters and emits the existing GP_FIT_FINISHED event. It receives the same
version/diagnostic, FitRecord list, physical counters and callback state. The
exact order is record append, executed-recovery-attempt count, fit time,
validation-posterior calls and finish event. Diagnostic identity is retained;
no copying/reinterpretation or random draw is added.

Numerical fitting/preparation exceptions remain in the preceding try and do not
create a returned record or finish event. Status/model usability decisions stay
after reporting. Physical fit success, warm-state retention/trimming, logical
training-version updates and return values remain visibly in fit_version.
The runner reads prepare/configure/fit, record returned work, then judge and
commit a usable model. No fitting, recovery or scientific policy was extracted.
Record/report resource errors still propagate outside the numerical-fit catch;
progress exceptions stay isolated. Finish reporting still precedes the current
fit's success, warm cache and logical commits.

All 18 new cases passed before and after extraction. Real graph recovery tests
check first/sequential fits and bounded newest-first warm history in both
dtypes, with normal and failing start/finish callbacks. Clearly labelled
controlled result doubles cover typed initial/update failures, raised fit calls,
missing/failed model guards, executed/skipped recovery attempts, partial state,
identities and recommendation accounting. A record-allocation fault establishes
its precedence over later failed-status handling. Doubles are contract fixtures,
not invented historical GP states. Every case checks ambient RNG preservation
and uses a fake simulator with physical entry points blocked.

The local scientific/import suite had identical 291 outcomes before/after:
282 passed, 8 skipped, the known NEI Cholesky fixture failure. All 46 metadata/
lifecycle checks passed before capture; 53 metadata/lifecycle checks passed after
adding the completed evidence. Whole-runner AST comparison is identical after inlining
the helper and restoring the diagnostic alias. No new relevant failure appeared.

One Slurm job 6855141 requested 20 CPUs, 20 GiB and a 20-minute limit. It ran
399 seconds; final batch MaxRSS was 11510484 KiB
(10.977 GiB). TRUBA tests: 279 passed, 8 skipped and the
same NEI failure, with all 18 additions passed and no changed accepted outcome.
Slurm FAILED/exit1 reflects that fixture; the independent scientific gate passed.
No original baseline was rerun, no duplicate job or automatic retry submitted.

B40 target 16 (/records/48) matched all 40 archived observations, 25 fit/update
events, 24 proposals and one terminal recommendation. It passed strict replay,
fit-event comparison and exact mathematical checkpoints before B100 began.
B100 target 0 (/records/0) matched all 100 observations, 85 fits/updates,
84 proposals and one recommendation. Every 25/85 training identity, frozen
transform, GP/posterior quantity, acquisition/candidate/context, seed/RNG,
duplicate recovery, terminal result and logical accounting agrees exactly.

| Case | Exact full replay hash | Exact checkpoint hash |
| --- | --- | --- |
| B40 | 687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782 | 12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c |
| B100 | c97b62ec714905ff97037561aad8ac3aef638abf9e189a926d6306844d3854c2 | 06338d126708d62fdb3967db3d1603eb1529270ec4c6704322b621cd6b4fa150 |

Saved original packets lack raw progress-event logs. Expected GP events are
explicitly reconstructed from their FitRecords and the preserved baseline
emitter. Both emitter ASTs were verified against aa7f4fe. An optional observer
records candidate events during the same strict replay, forwarding the original
callback and adding no fit or response call. All 50 B40 and 170 B100 GP events
match in version, status/message, IDs and phase/index metadata, with unchanged
boundaries relative to transform/evaluation/proposal/recommendation events.
Original historical event timestamps are unavailable. Only existing declared
elapsed_seconds fields are excluded. These are reconstructed expectations,
not newly discovered historical logs or exact wall-time equality claims.
Mathematical payload extraction/comparison remains unchanged.

The qualified environment matches with the existing unordered-library metadata
rule. No tolerance, reference hash, fitting setting, prior, precision or policy
changed. All real physical-simulator tripwires recorded zero attempts. Logical
archived-response evaluations retain their normal accounting.

Starting/published milestone: 6a6dfe85073475c14a2155d0527b98a218419a70.
Executed clean candidate snapshot: fe8e0aa3e2e2edb419e0202e9076f95302c62988.
Scientific baseline: aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3.
Executed production/tests and diagnostic files match this milestone byte for
byte; final documentation describes the completed run. A private clean snapshot
on the existing branch name served the lifecycle. Development stays on
ege/repro-refactor with no further push or PR update after the initial milestone.

Changed files: runner.py, runner characterization tests, refactor-validation
contract, fit_result_reference_inputs.json, optional event-observer/gate wiring
in refactor_reference.py/readability_regression.py and its batch script, the
stdlib fit_event_evidence module and its tests, project/state.toml,
project/experiment_registry.toml, and this directory's six evidence files.
GP/acquisition/recommendation implementations, configurations, manuscript and
older scientific evidence remain unchanged.

These files are derived portable summaries, not raw acceptance packets.
Original raw and derived SHA-256 values are separate in the manifest/index.
Raw logs, full event/math captures, original environment records, source bundle
and processing code are preserved privately with verified checksums. Historical
GP intermediates remain unavailable; original-source GP references are
reconstructed and the conditional final cold fit is not the historical warm
terminal GP. The known NEI fixture and unavailable CUDA/SBI coverage remain
limitations. Inherited paths in existing remote registries remain a separate
public-release concern; the milestone introduces no new operational paths.

Verify portable bytes from this directory:

    shasum -a 256 -c SHA256SUMS

Independent raw verification restores the private inputs/snapshot, checks
fit_result_reference_inputs.json and worker SHA256SUMS, reruns the unchanged
mathematical comparators and the metadata-only fit-events comparator. Sanitized
environment summaries cannot replace private raw acceptance packets.

Next recommendation only: review a narrow proposal-result validation/reporting
boundary while leaving acquisition mathematics and duplicate-recovery policy
visible. Characterize failure precedence before considering extraction. No
further refactor, GP-prior investigation, terminal correction or release work
has begun.
