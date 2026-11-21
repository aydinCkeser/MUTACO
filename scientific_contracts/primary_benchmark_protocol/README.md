# MUTACO--NEI--Sobol primary benchmark protocol

**Classification:** normative, frozen scientific design

**Authority subject:** the prospective primary comparison of canonical MUTACO,
objective-aligned composite Monte Carlo noisy expected improvement (NEI), and
nonadaptive scrambled-Sobol search on the frozen Paper-120 graph-oscillator
targets

The machine-readable authority is
[`mutaco_nei_sobol_primary_v1.json`](../../experiments/paper/studies/graph_paper120/studies/mutaco_nei_sobol_primary_v1.json).
This contract freezes scientific behavior but authorizes no implementation,
target construction, qualification, source transfer, TRUBA submission, or
population campaign. The production NEI numerical tier is intentionally not
fixed here; it must be selected by the frozen numerical qualification rule
below and then recorded in a new approval-gated declaration.

## Scientific question and interpretation

At independent inference budgets `B=40` and `B=100`, every method attempts to
match the same hidden-parameter-generated, frozen physical target response
`y_star` without receiving the hidden generating parameter.

The primary endpoint is one fresh held-out physical-response MAE at the frozen
terminal recommendation. The comparisons have deliberately different scopes:

- **MUTACO versus NEI** is a controlled acquisition-policy comparison. Both
  arms use the same multi-output GP family, input normalization, frozen
  initial-response standardization, fitting and recovery policy, and
  continuous terminal optimizer.
- **MUTACO versus Sobol** is an end-to-end comparison of adaptive model-based
  inversion with nonadaptive surrogate-free space-filling search. It does not
  isolate acquisition or sample allocation alone.

No manuscript or report may describe the three-way comparison as sharing a
common surrogate or terminal estimator.

## Cohort, budgets, and shared initial geometry

The primary cohort is the existing 120-target Paper-120 v3 ledger. Hidden
generating parameters remain unavailable to every search method and may be
used only for explicitly secondary analysis.

`B=40` and `B=100` are independent campaigns rather than prefixes of one
trajectory. MUTACO and NEI allocate `16 + 24` and `16 + 84` calls respectively.
Sobol uses exactly `B` calls.

For each `(campaign, target, budget)`, one method-independent scrambled Sobol
engine generates a single ordered sequence of length `B`. The exact first 16
points, in preserved order and dtype, are supplied to all three methods.
Sobol alone continues with points 17 through `B` from that same engine. It is
forbidden to construct the initial design and Sobol continuation from separate
engines or seeds. The full sequence identity and the shared-prefix identity
must be persisted.

The parameter locations are shared, not their simulator realizations. Search
simulator contexts are method-addressed, and each logical evaluation receives
fresh named coordinate-addressed randomness.

## Canonical MUTACO arm

MUTACO uses the registered graph profile for the applicable budget without a
scientific modification. Its compatibility criterion, barrier refinement,
operator-norm uncertainty geometry, spectral-uncertainty fallback, GP policy,
and response transform remain governed by the MUTACO D contract. Extending the
surrogate boundary for NEI joint sampling must leave all existing MUTACO
identities, candidates, diagnostics, random streams, and call accounting
unchanged.

## Composite Monte Carlo NEI arm

The registered policy identifier is
`composite_mc_nei_physical_mae_v1`. It is sequential `q=1` NEI over the same
latent standardized multi-output GP used by MUTACO.

For evaluated points `Q_t=(theta_1,...,theta_n)` and candidate `theta`, each
Monte Carlo replicate is one joint latent posterior realization over every
point-task pair in `(Q_t, theta)`. Observation noise is not newly sampled. The
learned likelihood affects latent posterior uncertainty, including uncertainty
at evaluated points.

Every sampled standardized response is transformed to physical coordinates
with the run's transform fitted once from the complete 16-point initial design.
For sample `s`,

```text
incumbent_s = min_i mean(abs(Y_s(theta_i) - y_star))
candidate_s =       mean(abs(Y_s(theta)   - y_star))
improvement_s = max(incumbent_s - candidate_s, 0)
NEI(theta) = mean_s improvement_s
```

The target is fixed. Independent marginal sampling, observation-noise samples,
standardized L2 objectives, scalar-GP surrogates fitted to observed MAE, and
simulator reranking are prohibited.

Within one acquisition-optimization invocation, common base samples and the
sampled incumbent realizations are fixed across candidate comparisons. A
conditional-Gaussian implementation may cache a validated baseline
factorization and sample candidates conditionally; it must be mathematically
equivalent to the required joint latent distribution. Exact evaluated-point
duplicates are rejected and must have zero improvement up to the registered
numerical tolerance.

The registered baseline factorization policy is
`promoted_float64_cholesky_zero_jitter_v1`. The already validated,
structurally symmetric latent covariance is promoted to float64 before its
single strict Cholesky factorization. No diagonal jitter, eigenvalue clipping,
or other covariance perturbation is applied. Baseline sampling, conditional
solves, and conditional-covariance arithmetic use the promoted factorization;
the public acquisition value returns to the candidate dtype. Failure of the
promoted strict factorization remains a typed `NEIAcquisitionError`.

Finite non-worsening local refinement is accepted. If refinement fails or
worsens the acquisition while at least one finite unevaluated raw candidate
exists, the highest-ranked such raw candidate is returned. Exact acquisition
ties resolve to the earliest raw-candidate index. A run fails only when the
joint posterior cannot be evaluated under the accepted numerical contract or
no valid finite unevaluated candidate can be produced. Fallback status and
reason are retained.

## Sobol arm and terminal decisions

Sobol is response-independent throughout search. It fits no GP, calls no
acquisition, and never allows an observed response to change a later point.

After budget exhaustion, MUTACO and NEI both use
`continuous_posterior_mean_physical_l1_mae` with identical controls and exact
method-independent target-budget-addressed initial optimizer geometry:

```text
argmin_theta mean(abs(mu_final_physical(theta) - y_star)).
```

The common projected-Adam mechanism tests convergence only after evaluating a
post-update feasible state and compares scores from distinct states. Each
restart retains its best finite non-worsening post-update state rather than
assuming its final iterate is best. If no restart improves or ties its finite
raw start, the best finite raw candidate is returned.

Sobol uses `evaluated_best_observed_physical_mae_v1`: it selects the evaluated
point with minimum observed physical-response MAE, resolving an exact tie by
earliest evaluation. It fits no post-hoc GP and performs no refinement.

Every terminal decision is zero-call and must be durably frozen before any
assessment. Method-neutral artifacts distinguish posterior-mean terminal
scores from observed-response terminal scores.

## Randomness ownership

The exact coordinate ownership is:

| Object | Coordinates after the versioned family identity |
|---|---|
| Sobol design and shared prefix | campaign, target, budget, `sobol_design` |
| Search simulator evaluation | campaign, target, budget, method, evaluation index, `simulation` |
| GP fit and recovery | campaign, target, budget, method, training version, attempt, `gp_fit` |
| MUTACO proposal | campaign, target, budget, method, iteration, `mutaco_proposal` |
| NEI Monte Carlo samples | campaign, target, budget, method, iteration, optimization stage, `nei_mc` |
| NEI optimizer | campaign, target, budget, method, iteration, `nei_optimizer` |
| Continuous terminal starts | campaign, target, budget, `continuous_terminal` |
| Held-out assessment | campaign, target, budget, replicate, `heldout` |

The method coordinate is deliberately absent from shared initial geometry,
continuous terminal starts, and held-out assessment. It is present in every
search simulator stream. Budget is always present. Ambient Python, NumPy,
Torch, and CUDA RNG state cannot influence scientific output.

## Held-out endpoint and descriptive trajectory

After all method recommendations or failures for one target-budget pair are
frozen, every valid recommendation receives one assessment under the same
method-independent stochastic context. It is outside `B`, cannot feed back,
and is named **paired one-realization held-out physical-response MAE**. It is
an aggregate stochastic-performance endpoint, not a noise-free per-target
error.

The graph execution adapter restores that context as a complete
`SimulationContext`: the frozen assessment family is the scientific run ID,
`purpose` is `paired_held_out_assessment`, `logical_index` is the replicate,
`repeat_index` is zero, and the evaluation seed is the held-out coordinate's
derived seed. These adapter fields remain method-independent and do not alter
the held-out seed coordinate above.

Every run also retains `best_observed_mae_so_far`. This is a noisy,
selection-affected descriptive trajectory. It is not interchangeable with the
held-out endpoint and cannot support an undeclared hitting tolerance or a
held-out accuracy-versus-calls claim.

## NEI numerical qualification

Five off-cohort targets from the same frozen target-generation protocol are
used for NEI-only independent B40 and B100 qualification trajectories: 700
search calls total, excluding separately accounted target construction. No
held-out endpoint or MUTACO/Sobol performance is used to select NEI numerical
settings. Target identities, tier values, and snapshot positions must be
declared before execution.

The saved immutable GP snapshots cover post-initial, intermediate, and late
states from a predeclared high-effort qualification trajectory. For each
snapshot, every candidate produced by every tier plus the high-effort reference
candidate is evaluated using four independent reference batches of 4,096 joint
latent samples each. All candidates within a reference batch use common base
samples.

Let `A_ref_max` be the largest reference acquisition over this union,
`A_ref_selected` the reference value of a tier's candidate, and

```text
s_run = max(median initial-design observed physical MAE, 1e-12)
normalized_regret = max(A_ref_max - A_ref_selected, 0)
                    / max(A_ref_max, 0.01 * s_run).
```

The lowest-effort tier qualifies only if optimizer success is at least 95%,
median normalized regret is at most 0.01, the 90th percentile is at most 0.05,
and there is no systematic boundary, duplicate, covariance, or raw-pool
failure. Production `N_MC`, raw count, local starts, steps, learning rate, and
other controls are frozen only after this rule passes. Reference estimates are
qualification artifacts and are not required in population runs.

After numerical settings are frozen, an independent one-target, three-method
B40 smoke uses 120 search calls and three paired assessments. Combined with the
700-call numerical qualification, the planned qualification ceiling is 823
calls, excluding target construction. Every scientific phase remains a
separate explicit authorization gate.

An assessment-only rescue is permitted only for an engineering failure after
the immutable all-method decision freeze and before any held-out simulator call
begins. It must bind to the hash-verified freeze and original declaration,
reuse the same method-neutral held-out coordinate, evaluate each frozen valid
recommendation exactly once, persist append-only rescue records, and execute no
search or terminal optimization. A rescue is a new declared execution attempt;
it never changes the failed attempt or retroactively marks it complete. Any
prior held-out simulator call makes this narrow rescue ineligible.

The pre-execution Q1 freeze is
`experiments/graph_inversion/configs/paper/primary_nei_q1_preparation_v1.json`.
It fixes a new five-point no-response theta family, snapshot positions
`(16,28,40)` and `(16,40,70,100)`, and three lowest-effort-first candidate
tiers: `(64,64,4,8,0.05)`, `(128,128,8,16,0.04)`, and
`(256,256,12,24,0.03)` for `(N_MC, raw candidates, local starts, optimizer
steps, learning rate)`. The highest tier owns the qualification trajectories;
the high-effort reference candidate uses `(1024,512,16,48,0.025)`, while its
candidate-union scoring still uses four independent 4,096-sample batches. The
frozen rule selects the lowest qualifying tier. The five target
responses are not constructed by this freeze and require exactly five
separately authorized calls.

Every Q1 execution attempt is append-only and declaration-addressed. Numerical
tier selection requires one complete matrix produced under a single declared
qualification identity; partial or completed records from failed historical
attempts cannot be combined with a later declaration.

## Statistical estimands and failures

For comparator `C` and budget `B`, the paired difference is

```text
D_j(B,C) = E_hold(j,B,MUTACO) - E_hold(j,B,C),
```

so negative values favor MUTACO. The primary summary is the median paired
target-level difference. Supporting summaries are mean difference, IQR, win
probability, and marginal method distributions. Deterministic paired bootstrap
intervals resample whole targets; cross-budget summaries keep both budgets for
one target in the same cluster. B40 and B100 are reported separately. Holm
correction is required only if the study makes formal superiority tests across
the four budget-comparator combinations.

Every contrast reports counts for both completed, only MUTACO completed, only
the comparator completed, and neither completed. Numerical paired differences
use jointly completed targets and must appear with that completion table. A
failure-aware sensitivity treats a one-sided failure as losing the pair.
Complete-case results cannot silently stand in for the full declared cohort.

## Artifacts and accounting

Plots are derived only from machine-readable artifacts. Required records
include the campaign manifest, one target-method-budget record, every simulator
evaluation with physical response and named context, adaptive GP/acquisition
diagnostics, zero-call terminal metadata, paired assessment evidence, phase
timings, endpoint tables, trajectories, paired differences, and runtime tables.
NEI production records retain actual acquisition values, configuration, seeds,
base-sample identity and shape, baseline size, optimizer/fallback status, and
timing; they need not compute high-precision reference acquisition values.

At the Q1 evidence boundary, the complete trajectory attempt is recursively
canonicalized once before semantic identity or persistence. Finite values retain
the canonical JSON semantics used by existing identities; signed infinities are
represented by the reserved, versioned
`__mutaco_evidence_nonfinite_float_v1__` tag, while NaN is forbidden with an
exact field path. Attempt and record identities are computed from that canonical
object, independent of pretty formatting. A pre-hash target-position/budget
address permits a serialization-safe failure envelope to be persisted without
embedding the failed result. Scientific validators remain responsible for
deciding whether infinity is admissible in a particular field.

The full primary campaigns contain 50,400 in-budget search calls and 720
held-out calls, totaling 51,120 scientific calls. B40 has 123 calls per target
and 14,760 total. B100 has 303 calls per target and 36,360 total. Target
construction, terminal optimization, analysis, and plotting add zero campaign
calls. No automatic campaign retry is authorized.

## Implemented zero-call foundation

The approved implementation foundation is versioned independently from any
scientific execution. Core implementation resides in the additive joint
posterior and NEI/recommendation surfaces under `src/mutaco/`; primary schemas,
shared design construction, target execution, and numerical-qualification
declarations reside under `experiments/paper/primary_benchmark/`; reference NEI
and Sobol method runners reside in
`experiments/benchmarking/methods/primary_search.py`. Historical Paper-v3
protocols and records remain unchanged. These modules authorize no campaign and
contain no frozen production NEI numerical tier.
The Q1/Q2 layer additionally provides capped append-only worker orchestration,
global collectors, tier selection, compact export, and a validation-only TRUBA
profile. Its submission surface refuses execution while the five frozen target
responses are absent or authorization remains false.
