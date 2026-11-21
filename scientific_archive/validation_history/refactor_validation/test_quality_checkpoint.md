# Test quality and runner readability checkpoint

Reviewed production source: `9219b7a82e5713aec27b9e0637f4e8588f5170aa`.
Comparison: original scientific baseline `aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3`.
This checkpoint changes a test fixture and documentation only. It does not
perform a seventh runner extraction. Results and immutable byte identities are
recorded in `benchmarks/reproducibility/test_quality_v1/manifest.json`.

## NEI fixture diagnosis

The contract in `scientific/primary_benchmark_protocol/README.md` requires
`promoted_float64_cholesky_zero_jitter_v1`: one strict float64 Cholesky, no
jitter, clipping or covariance repair, with typed failure if it fails. This
composite NEI policy is distinct from the scalar qLogNEI/GP jitter policy.

The original five-by-five fixture is exactly `C = V V^T`, where row i of V
is `(1, x_i, x_i^2, x_i^3)` and x is `(0, 1/4, 1/2, 3/4, 1)`. Thus C is
positive semidefinite, has exact rank four and determinant zero. Its sole null
direction is `(1, -4, 6, -4, 1)`. A float64 eigenvalue around 2.5e-16 is
roundoff, not evidence of strict positive definiteness. Precision promotion
cannot restore a missing rank. The old success expectation is a fixture defect;
a successful strict factorization of it is backend-dependent roundoff. The
registered contract propagates a Cholesky failure; it does not add an exact
rank test or guarantee detection of every mathematically singular matrix.

The corrected success fixture changes only C[0,0] from 1 to `1 + 2^-23`.
This is one exactly representable float32 input variance. Since its direction
has nonzero overlap with the sole null vector, the resulting matrix is
strictly positive definite. Exact rational arithmetic gives determinant
`9/8796093022208` and five positive leading principal minors. Its smallest
eigenvalue is approximately 1.703e-9. Native float32 Cholesky still fails;
float64 promotion succeeds. The original reconstruction tolerances and full
zero-jitter diagnostic assertions are retained. No production covariance is
perturbed. The original matrix remains in the shared test data and raw diagnostic
evidence. Separate rejection cases use two duplicate unit rows (a two-by-two
all-ones covariance) in float32 and float64. Its exact zero Cholesky pivot
exercises the documented error without depending on the rounding outcome for the old
matrix. Existing indefinite-covariance rejection coverage remains unchanged.

Before correction, the baseline and current source each gave 5 focused passes
and the same one failure, with physical entry points blocked. This directly
reproduced the issue rather than attributing it to a refactor or a platform.
The first small TRUBA full-suite run exposed two failures in the newly added
rejection cases when they still used the old rank-four matrix: its float64
Cholesky returned success on that backend. This was an overstrong new test
expectation, not an accepted regression or a reason to change the algorithm.
The positive-definite promotion case passed. Only the new rejection fixture
was changed to duplicate unit rows, and a manual NEI-only follow-up was used;
the full suite was not automatically resubmitted. Both failed cases and their
raw evidence remain recorded. See the manifest for the final local and Slurm
outcomes and the explicit limit on final-source full-suite verification.

## Paper-to-code map

The map follows the accepted manuscript's Algorithm box
(`docs/paper/latex/bandits_v6.tex`, label `app:mutaco_algorithm`) and the
`app:gp_surrogate`, `app:gp_acquisition` and `app:gp_terminal` definitions.
The runner also supports registered non-paper options; this map is for the
accepted B40/B100 profile, not a claim that every policy is the paper algorithm.
Locations below refer to the reviewed, unchanged production source.

| Manuscript stage | Current implementation |
| --- | --- |
| Input/capability/seed checks (engineering preamble) | `run_sequential`, runner.py:1171; `_validate_pre_run_options`:888 |
| 1. Generate initial Sobol design and evaluate in order | `_initial_design`:535, `_collect_initial_observations`:917; `_evaluate`:567 preserves retries, contexts and seeds |
| 1. Freeze initial means/scales and standardize data/target | `_prepare_frozen_responses`:817, called at runner.py:1228 |
| 2. Assemble committed training data and fit the multitask GP | `_prepare_gp_training_data`:959; nested `fit_version`:1253; `_build_model` in surrogate/exact_multitask.py:850 performs input normalization/task augmentation; fitting/recovery remains in that module |
| Fit diagnostics and acceptance, at every update | `_record_gp_fit_result`:1012; status/model guard, bounded warm history and logical commits stay in `fit_version` |
| 3(a-c). Screen, choose barrier/exploration branch, refine and rank | `propose_candidate` in acquisition/proposal.py:1035; called at runner.py:1346; existing mathematical objectives and restart functions remain in acquisition modules |
| 3(d). Recover an unevaluated raw point on an exact duplicate | Explicit runner.py:1380 control flow, calling `recover_best_ranked_unevaluated_raw` in acquisition/proposal.py:914 |
| 3(e). Evaluate, standardize/commit, then refit | `_evaluate` at runner.py:1432; `_standardize_and_commit_sequential_observation`:985, called at 1450; `fit_version` at 1459 |
| 4. Recommend from the final surrogate in physical response coordinates | Explicit runner.py:1472 terminal dispatch; recommendation/physical_mae.py and recommendation/_continuous.py; no new simulation |

## Readability assessment

The six helpers provide meaningful progress. The top-level function now reads
validate -> initial design/collection -> freeze -> initial fit -> sequential
proposal/evaluation/commit/refit -> terminal recommendation. Each helper has
one cohesive responsibility; none introduces a second scientific algorithm.
The frozen preparation helper deliberately returns partially completed state;
the collection and commitment helpers preserve in-place ordered commits.

The original `run_sequential` spans 629 lines versus 521 now (17% reduction).
Its AST contains 172 statements versus 132 (23% reduction, including its two
nested helpers). The whole file grew from 1437 to 1569 lines. These are
navigation measures, not proof of comprehension or a completed architecture.
The numerical work and failure boundaries are easier to locate, but this is
still a long orchestration function, not a compact rendering of the paper.

The main remaining costs are:

- Mutable observations/records, untyped counter dictionaries and warm history
  are shared across scopes. `_collect_initial_observations` takes 11 inputs,
  including substantial callback/accounting plumbing; `_record_gp_fit_result`
  takes seven. More tiny helpers would repeat this cost without clarifying the
  science. A general state framework would expand risk and is not proposed.
- Setup and the nested `finish` closure occupy much of the preamble. `finish`
  intentionally observes current partially committed state; `fit_version`
  intentionally shares the frozen transform, current observations and warm
  history. Their closures are useful lifecycle boundaries, not automatically
  mistakes to replace with classes.
- The sequential body interleaves attempt recording, result usability,
  duplicate recovery, success commits and progress events. These have distinct
  failure/commit semantics. In particular a failed proposal is recorded before
  it becomes a committed successful proposal. The physical-accounting name
  describes adapter operations; an archived adapter response is not a new
  physical simulator execution. Read the separate tripwire count alongside it.
- Terminal policy dispatch remains long and mixes scientific choices with
  context, seeds and reporting. Keep it explicit pending its separate policy
  investigation; shortening it must not silently resolve those issues.

The proposed proposal-result reporting extraction is lower value now. A helper
for a few counters would not expose another manuscript stage. Combining the
whole region would absorb early returns, replacement of `proposals[-1]` during
duplicate recovery, and the commit/report boundary, obscuring useful control
flow. Leave this region explicit for the next milestone.

The highest-value next bounded area is **raw screening and branch selection in
`acquisition/proposal.py`**, not another reporting helper or GP policy change.
Review whether its existing raw-posterior/objective/ranking sequence can expose
one clear screening result before refinement, while retaining the current
failure-construction boundaries. This corresponds directly to Algorithm 3(a-c).
Do not combine it with optimizers, duplicate recovery or terminal selection.
Before any such extraction, characterize barrier and exploration paths, empty
or nonfinite pools, diagnostic policy variants, tied ranking, RNG, posterior
call counts and failure precedence, then require the existing exact B40/B100
gates. If doing so requires redesigning result/error state, first document the
acquisition phase map instead. GP attempt fitting and recovery already have
separate entry points; priors/optimization are a more sensitive next target.

## Test-gate compatibility

Historical reference packets and earlier reports correctly retain their old
NEI failure. Do not rewrite them. The historical `readability_regression.tests_gate`
explicitly requires that failure and the earlier test inventory; it will reject
this intentional failure-to-pass correction and two additional singular cases.
This test-only job uses an explicit zero-failure pytest verdict, preserving that
legacy comparator. Before a future production extraction, version its software
test expectations using this checkpoint's JUnit identities and counts, with
only this declared correction and two additions allowed. Keep mathematical
comparators, environment fingerprints and all B40/B100 references unchanged.
No acquisition/GP/terminal code was changed during this review.
