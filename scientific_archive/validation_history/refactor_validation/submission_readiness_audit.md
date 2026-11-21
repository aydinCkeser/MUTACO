# Documentation, architecture and submission-readiness audit

Reviewed 28 October 2026: `ege/repro-refactor` at
`e99abf84966ac3b1be952fb675388840286334fd`; scientific baseline
`aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3`. This is a documentation audit,
not a new scientific validation, release approval or authorization to submit jobs.
Source line references below refer to the reviewed commit.

## Decision and supervisor summary

**Ready for Aydın's private technical review, with the limitations below;
not ready for external software submission or public distribution.** Nine
operation-preserving extractions have made the scientific sequence easier to
navigate. Saved evidence supports the qualified CPU environment and two complete
historical trajectories; it does not qualify every platform, policy or dependency
combination. No new numerical experiment was performed for this audit.

The highest-value next work is a portable installation/reproduction package and
release-governance decisions, rather than more small production helpers. Aydın
should separately decide whether to correct the failure-diagnostic shadowing.
No current scientific acceptance result depends on changing that diagnostic.

Live read-only Git/CLI checks confirmed the canonical repository is private,
PR #1 is OPEN/DRAFT at `7422a0caa2598d22db1d53d989fa39394fc4e99d`, and
the development remote is the same commit. Local HEAD is two commits ahead.
Live remote `main` is `06528705613a8b4cc49ca1e766eaa4400fe17714`, rather
than the locally recorded original baseline. No remote-tracking ref was updated;
this difference does not redefine the fixed scientific baseline. No push or
GitHub mutation was made. Connected-plugin private access failed; authenticated
Git CLI and `gh` supplied these read-only facts.

## Scientific execution and paper-to-code map

The current authoritative [paper-to-code map](../../../src/mutaco/README.md)
locates every scientific stage and its inputs/outputs. The remaining analysis
below is the dated audit snapshot, not current command or scheduler state.

The nine helpers expose meaningful stages. Relative to the original baseline,
`run_sequential` spans 521 rather than 629 lines, `propose_candidate` 268 rather
than 341, and the single-fit function 282 rather than 303. These AST-derived
navigation metrics are not a comprehension study. Shared mutable observation
lists, counter dictionaries, warm history, and long terminal dispatch still
require careful reading. The screening helper returns eight positional values;
this is a future API-documentation concern, not a reason to redesign it now.
Closures capture partial state and preserve exception/RNG contexts. Keep those
boundaries explicit; further tiny extractions risk more parameter plumbing.

## Verified evidence and its limits

The [latest derived manifest](../../../benchmarks/reproducibility/gp_optimizer_diagnostics_v1/manifest.json)
and preceding eight extraction summaries support:

| Accepted gate | B40 target 16, `/records/48` | B100 target 0, `/records/0` |
| --- | --- | --- |
| Strict archived observations | 40/40: 16 initial + 24 adaptive | 100/100: 16 initial + 84 adaptive |
| Fits/updates; proposals; terminal recommendation | 25; 24; 1 | 85; 84; 1 |
| Reconstructed fit-event boundaries | 50 exact | 170 exact |
| First/final-data checkpoint scientific SHA-256 | `12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c` | `06338d126708d62fdb3967db3d1603eb1529270ec4c6704322b621cd6b4fa150` |
| Full replay mathematical SHA-256 | `687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782` | `c97b62ec714905ff97037561aad8ac3aef638abf9e189a926d6306844d3854c2` |

Candidate coordinates and effective context must match before an archived
response is supplied. All saved comparisons are exact, with only predeclared
timing/provenance exclusions and unordered runtime-library enumeration.
No candidate substitution or numerical tolerance was introduced.

The latest local before/after and TRUBA selected scientific/import suites each
record **523 passed, 8 skipped, 0 failed**; 531 case identities/outcomes agree.
This is a selected suite, not the whole repository's default `pytest` collection.
The v2 gate requires zero failures; earlier baseline reports correctly retain
the old NEI fixture failure. The final quality-closure and nine-refactor reports
are cumulative evidence, not nine new experiments in this audit.

Last validation: job 6860709, COMPLETED/0:0, 444 seconds, 20 CPUs, 20 GiB
requested, 12.418 GiB reported peak RSS. Pytest and scientific exits were both
zero, independently. No new resource allocation is authorized here.

This audit rechecked all **58 files in 12 portable checksum sets**, all **47
latest validated core-file hashes**, both compressed archive/manifest hashes
and all **8 archive members/sidecars** using existing verification code.
The original B40 portable mathematical payload hashes were recomputed without
GP calculations. The latest private evidence inventory and its **130 files**
were verified read-only. Original raw/environment packets remain private;
portable summaries are derived and are not byte-identical replacements.
No reference, archive or raw backup was edited.

B40/B100 are independent historical campaigns, not nested budgets. Two cases
protect the exercised trajectories and same-environment implementation changes;
they do not prove every target/policy, scientific superiority, GPU behavior,
SBI workflows or cross-platform equality. Original historical GP iterates,
optimizer states and complete environments are absent from the compact archives.
Saved fitted states/events are reconstructed original-source references.
The final-data cold-fit checkpoint is conditional, not the historical final warm
GP. Mac historical bitwise agreement is neither required nor established.

## Reproduction and environment boundaries

- Qualified scientific reference: TRUBA Linux x86-64 CPU, Python 3.11.17;
  torch 2.12.1+cpu, BoTorch 0.18.1, GPyTorch 1.15.2, linear_operator 0.6.1,
  NumPy 2.4.6, SciPy 1.17.1, pytest 9.1.1; Torch threads 20/1 and recorded
  OpenBLAS/OpenMP builds. Full metadata is in the latest manifest. SBI is absent
  from this core-only runtime; this does not qualify NPE/SNPE training.
- Local development: the separate macOS ARM64 Python 3.11 `mutaco-sbi`
  environment and shell launchers; Windows PowerShell launchers remain supported
  entry points but were not freshly qualified in this audit. MPS/CUDA and all
  package/Python combinations permitted by broad dependency ranges are unqualified.
- Archived replay: committed observations and checked candidates/contexts,
  with graph physical entry points patched to tripwires; no new physical data.
- Physical execution: concrete simulator and campaign workers can generate new
  data. Generic `run_mutaco_target` defaults include assessment, so it is not an
  offline tutorial command. Root pytest protects temporary output, not all
  simulator entry points. Do not equate arbitrary `pytest` or a generic benchmark
  worker with the guarded selected suite.

Verified local entry points and the exact selected-suite recipe are in
[the local guide](../../../infrastructure/local/README.md). From repository root,
portable B40 integrity can be checked without fitting:

```sh
infrastructure/local/run_mutaco.sh -m infrastructure.truba.shareable_evidence verify --directory benchmarks/reproducibility/shareable_milestone_v1
```

For each other evidence directory, run `shasum -a 256 -c SHA256SUMS` from that
directory (`sha256sum -c SHA256SUMS` is the Linux equivalent). For both archives, the existing `verify_archive` function checks
compressed bytes, manifests, every member and sidecars without extraction.
Use the guarded integrity command in the local guide. Integrity is not a new
historical replay verdict.

The strict replay CLI is
`python -m experiments.paper.exploration.scalar_primary_replay --budget 40
--target 16 --output <external-output.json>` (B100 uses `--budget 100 --target 0`).
It performs numerical work; it is not an environment-free integrity check.
Historical numerical acceptance must use the qualified compute environment.
Do not execute it on a login node or infer Mac equality from successful capture.

For authorized TRUBA work, the tracked `infrastructure.truba.lifecycle
prepare-validate` validates exact-source transfer and writes a receipt; it has
no submission command. The candidate regression entry point is
`infrastructure/truba/readability_regression.sbatch`, with explicit verified
private references and an applicable v2 input manifest. It runs tests, B40 and
then B100, stopping on failed gates. A transport-only SSH config and user-owned
paths are supplied privately, not embedded here. Scheduler policy, available
account/runtime and separate submission approval must be checked again.

**Reproduction gap:** the latest full-suite selection/capture wrappers and
private reference restoration recipe are preserved privately, not all exposed
by the tracked batch entry point. Its original test selection is narrower than
523 cases. A future engineering task should publish a reviewed portable driver
and restoration specification, reusing existing engines and comparators.
Strict environment comparison still requires access to original path-bearing
raw packets; sanitized summaries alone cannot pass it. Incremental private
source bundles require their preserved base commit/bundle. A future external
researcher needs approved access or an approved portable reference-distribution
plan; do not quietly weaken this check.

## Prioritized readiness findings

No P0 blocker was found for controlled, guarded private technical review.
This is not a claim that all code, history or workflows are exhaustively audited.

| Priority | Confirmed finding and next action |
| --- | --- |
| P1 — distribution authority | No root LICENSE or CITATION.cff; `pyproject.toml`:8-20 has development version, broad dependency ranges and only generic contributor attribution. GitHub reports no license; state marks it unresolved. Obtain ownership/licensing/data-sharing approval and author-approved citation before external distribution. Do not invent these files' legal content. Existing boundary tests deliberately assert absence of an invented license. |
| P1 — portable installation | `configs/environments/mutaco-sbi-macos-arm64-resolved.txt`:28,30 contains Conda-build `file://` requirements, not credentials. The previous fresh-install command consumed that snapshot. The guide now uses existing top-level pins and labels the missing clean-install/transitive-lock qualification. JSON locks record environment facts, not a standalone installer. Build a portable, approved pinned installer and test a fresh environment later. |
| P1 — reproducibility handoff | Strict same-environment raw references remain private; latest full-suite wrappers and base-bundle restoration dependencies need a documented portable entry point. Preserve raw checks and derived/raw distinction. |
| P1 — sharing hygiene | A bounded tracked-text scan found cluster account-bearing absolute paths in 84 files, including `project/experiment_registry.toml`:271 and historical manifests. These are provenance/access metadata, not evidence of passwords. No private-key marker, quoted credential assignment or personal `/Users/` path was found by those patterns. One resolved requirements file contains local-install references. This is not a complete secret/history/binary scan. |
| P1 — scientific-method clarity | Explain registered-prior/MAP fitting, recovery and precision alongside manuscript language. Prior-free MML comparison, terminal dispatch and global-best-raw issues remain separate scientific decisions; review rather than silently correcting them. |
| P1 — GP failure diagnostics | The shadowing below discards a calculated last-valid covariance summary. Authorize a separate behavior change and tests before correction. Current accepted behavior remains protected. |
| P2 — documentation navigation | Root README's former “current” NPE result and `docs/status/CURRENT.md` described an older program/campaign as current. Both are now explicitly historical; current scientific scope is `scientific/paper_scope/README.md`. Historical result numbers were not revalidated or replaced here. The earlier quality-checkpoint line map also remains a historical snapshot; use this map now. |
| P2 — API/architecture | The public import surface mixes ordinary, diagnostic and retired comparator options; it needs a supported researcher-facing example and clearer stability/scope labels. Preserve useful profile/core/simulator separation and explicit failure commits. No seventh runner helper is recommended solely to shorten the function. |
| P2 — automation and packaging | No tracked `.github` CI workflows were found. A future zero-call CI job should use the established selection/guard, alongside metadata checks. Fresh wheel/editable-package installation and Python 3.12 support were not demonstrated by this audit. |

Private use: retain existing original manifests and path-bearing registries,
with operational credentials outside Git. Eventual public release: first seal
and approve raw provenance, then derive portable evidence/location mappings and
review the entire intended release history (including compressed/binary data).
Do not rewrite shared history or strip identity-bearing provenance casually.
Any source-history/privacy migration requires its own approved plan.

## `last_valid_covariance`: exact scope and proposed correction

In [exact_multitask.py](../../../src/mutaco/surrogate/exact_multitask.py):1220 the
outer fit variable starts at `None`. `recording_optimizer`:1258 declares only
`optimizer_exception` nonlocal. Its assignment at :1289 therefore creates a
local cell; the inner callback's `nonlocal last_valid_covariance`:1263 targets
that optimizer cell, not the fit variable. The callback actually updates only
`last_valid_state`. Python symbol-table inspection confirms this scope.

On an optimizer/caller-callback exception after at least one snapshot, the code
first records failure covariance, builds another model, loads the snapshot and
calculates last-valid covariance into the optimizer-local variable. After normal
Exception handling, the outer result passed to the summary at :1387 remains
`None`. The intuitively intended summary is the computed snapshot covariance,
not the failing current-state covariance; that intended correction is an
inference for Aydın to approve, not the existing contract.

Successful fitting and fixed conditioning reach :1324 and populate the outer
field normally. Failure before any callback has no snapshot; without an
optimizer observation, there may be no optimizer report at all. Parameter or
posterior-validation failures have their own later-stage behavior. If diagnostic
construction itself raises, it may mask the original optimizer error; a
`BaseException` outside the outer `Exception` catcher propagates. Those paths
must not be conflated with the ordinary after-callback reporting defect.

A repository-wide Python search found this key attached to returned diagnostic
and recovery-attempt dictionaries and serialized through runner `FitRecord`
(:314). No key-specific downstream production read was found. Recovery selects
on status/model availability and finite objective, not this covariance field;
`_failure_kind` runs before summary attachment. Consequently the verified effect
is missing failure-report information, with no identified change to candidate
selection, recovery or numerical calculations. External consumers are unverified.
The covariance inspection already executes, so the proposed scope correction
need not add a numerical call. This is still an observable diagnostic change.

The [13-case characterization](../../../tests/surrogate/test_optimizer_diagnostics.py):196-203
explicitly asserts `None` after optimizer/caller failures, verifies the two
covariance calculations, failure-summary alias and operation order. It also
covers before-callback failure, successful/iteration-limit/fixed/warm fits,
invalid parameters/posterior, real optimization and recovery exhaustion/success.
The older GP source predates all extractions; no fix was made here.

Minimal proposal, **requiring Aydın's authorization**: add
`last_valid_covariance` to `recording_optimizer`'s outer nonlocal declaration,
then change only the failure-diagnostic expectation to the already calculated
snapshot summary. Test success/fixed/before-and-after-callback/caller-exception
paths, unchanged diagnostic failure precedence, aliases, warnings, covariance
call counts, parameter/RNG bytes, recovery choices and posterior outputs.
Review diagnostic-hash implications explicitly; preserve old references and use
an approved version-aware diagnostic expectation. Do not update baseline hashes
or scientific tolerances merely to force a new report to match.

## Recommended next tasks and review questions

1. Package the existing zero-call scientific test selection, private-reference
   restore contract and environment preflight into portable documented tooling;
   qualify a clean install separately. Keep exact mathematical comparators.
2. Aydın/owners decide license, attribution/citation, approved data/provenance
   access, manuscript/implementation wording and whether the diagnostic repair
   is authorized. Prior-free MML, terminal-policy/global-best questions remain
   independent investigations with no implicit authorization.
3. Review submission metadata, supported platforms, resource guidance and a
   guarded CI plan. Do not begin further production refactoring solely for line
   count. Use this architecture map for a supervisor walkthrough first.

Audit verification passed: 35 relative Markdown links, syntax of three embedded
Python commands, state TOML parsing, and whitespace/diff checks. The documented
selected-suite collection has all 531 saved TRUBA test identities in exactly the
same order (including the 8 recorded skips); collection did not execute them.
The metadata/lifecycle/import-boundary subset passed 78 tests, zero failures and
zero physical-backend attempts. No full numerical suite was rerun for these docs.
All tracked production, test, package, configuration, registry, manuscript and
scientific-reference files remain unchanged relative to the reviewed source.

Audit operations: read-only Git/metadata/source inspection, AST/scope analysis,
checksum and archive verification, and guarded documentation/metadata checks. No physical
simulation, historical replay, GP fit, TRUBA job, installation, reference
regeneration or production/test modification was performed.
