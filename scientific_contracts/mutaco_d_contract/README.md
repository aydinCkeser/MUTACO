# MUTACO D mathematical contract

**Classification:** normative, current

**Authority subject:** simulator-independent MUTACO mathematics

**Implementation:** `src/mutaco/`
**Generated references:** [policy support](POLICY_SUPPORT.md),
[posterior numerics](POSTERIOR_NUMERICS_SUPPORT.md),
[likelihood support](LIKELIHOOD_SUPPORT.md), and
[candidate support](CANDIDATE_POLICY_SUPPORT.md). The machine-readable
[policy surface and compatibility matrix](POLICY_SURFACE.md) classifies every
selectable policy and identity-bearing internal mechanism.

This is the living mathematical specification for the supported MUTACO D
solver. Dated plans and decision records explain provenance but do not
override this contract. Simulator physics, graph-specific profiles, benchmark
budgets, and manuscript claims are outside this document.

## Coordinates and frozen response transform

For physical response component \(y_j\), fit the transform exactly once from
the complete initial design:

\[
\bar y_j=\frac{1}{n}\sum_{i=1}^{n}y_{ij},\qquad
s_j=\max\!\left(\operatorname{sd}_{\mathrm{ddof}=1}(y_{1j},\ldots,y_{nj}),
s_{\min}\right),
\]

\[
y_{\mathrm{std},j}=\frac{y_j-\bar y_j}{s_j}.
\]

The positive `scale_floor` \(s_{\min}\), ordered response identity, mean,
empirical scale, effective scale, dtype, and transform identity are recorded.
The transform is never refitted from sequential observations.

The selectable ablation policy `identity_physical_v1` omits initial-design
centering and scaling by using fixed zero offsets and unit scales. Physical
responses and targets therefore pass unchanged through the same surrogate,
acquisition, and recommendation interfaces. It records a distinct transform
policy and identity, is not a new method implementation, and is not the D
default. It is approved only for explicitly labelled response-standardization
ablations. The historical fitted-transform policy remains
`initial_design_zscore_v1`; an absent policy retains that behavior and its
existing transform identities. Declared `ddof` values other than 1 are
rejected rather than silently ignored.

## Exact multitask GP boundary

The surrogate consumes physical parameter vectors and frozen standardized
responses. Its public boundary supplies:

- posterior standardized mean \(\mu_{\mathrm{std}}(\theta)\);
- latent standardized covariance \(\Sigma_{\mathrm{latent,std}}(\theta)\);
- a separate observation/likelihood-noise covariance when explicitly asked.

The approved additive joint-latent boundary supplies a point-major mean
`[H,M]` and full covariance `[H*M,H*M]` across every supplied point and task.
It exposes cross-point covariance for composite Monte Carlo acquisitions while
retaining the same fitted model, transform, dtype/device, latent-noise boundary,
and mixed-precision covariance validation. Calling it cannot modify canonical
MUTACO posterior summaries, fitted parameters, identities, or random streams.

Likelihood noise is never part of D confidence geometry. The default
likelihood is task-diagonal Gaussian noise. Shared-isotropic Gaussian noise
remains an exact generic compatibility API; it is not exposed in ordinary
graph configuration and has no current JSON/TOML declaration.

The simulator-independent fitting default is `cold_only`. The explicitly
selected `graph_m20_sequential_recovery_v1` policy is an approved specialized
recovery contract, not a replacement default. Its fixed standardized
task-variance floor is

\[
\kappa_{\min}=5.991468697175151\times10^{-6}.
\]

That floor protects the positive task-kernel normalization denominator. It is
fixed, identity-bearing, and visible to optimization. It is not adaptive
jitter, likelihood noise, eigenvalue clipping, or a change to the GP model.
The complete warm/cold/four-restart/fixed-conditioning order is specified by
Decision 0021 and [posterior numerics](POSTERIOR_NUMERICS_SUPPORT.md).

## Posterior covariance numerical contract

The D-default graph-capable numerical pathway is
`float32_fit_float64_covariance`:

1. observations, learned parameters, marginal-likelihood optimization, and
   posterior means remain native Torch float32;
2. a cached fixed-parameter float64 mirror materializes latent covariance;
3. the float64 covariance is checked for finiteness, material asymmetry, and
   positive semidefiniteness;
4. only within-tolerance structural roundoff may be symmetrized;
5. the covariance is returned in float32 and strictly validated again.

This changes numerical evaluation of a fitted covariance, not GP training or
the covariance function. It consumes no additional fit attempt, simulator
call, or random draw. Material asymmetry and material negative eigenvalues
remain typed failures. There is no pseudoinverse, adaptive floor, adaptive
jitter, or silent float64 fitting fallback.

`native_float32` and `full_float64_diagnostic` remain diagnostic-only. The
selected policy, both validation stages, dtypes, tolerances, promotion, cache,
and timings are recorded.

## D confidence geometry and acquisition objective

For standardized residual

\[
r(\theta)=\mu_{\mathrm{std}}(\theta)-y^\star_{\mathrm{std}},
\]

define

\[
q(\theta)=\lVert r(\theta)\rVert_2^2,\qquad
\lambda(\theta)=\lambda_{\max}
\!\left(\Sigma_{\mathrm{latent,std}}(\theta)\right),
\]

\[
\rho^2(\theta)=\beta^2\lambda(\theta),\qquad
\Delta(\theta)=\frac{\rho^2(\theta)-q(\theta)}{\rho^2(\theta)}.
\]

Scientific confidence feasibility is \(q-\rho^2\le 0\). The matched-barrier
objective is

\[
J(\theta)=\lVert r(\theta)\rVert_2-\alpha\log\Delta(\theta),
\qquad \Delta(\theta)>\tau_{\mathrm{interior}}.
\]

The default `fixed_beta_v1` behavior remains the finite positive scalar
`beta=2.0`. The approved optional
`joint_chi_square_response_dimension_v1` policy resolves

\[
\beta=\sqrt{\chi^2_{M,c}},
\]

from declared response dimension `M` and coverage `c` in `(0,1)`. Its
coverage, dimension, chi-square quantile, float64 Torch bisection provenance,
and resolved beta are identity-bearing and validated against the simulator
response specification. It is a conservative spherical joint-response
multiplier under operator-norm geometry, not an exact covariance ellipse or a
theorem-labelled confidence sequence. For the graph M=20 profile at coverage
0.95, the quantile is `31.410432844230918` and beta is
`5.604501123581913`. `alpha` is finite and positive.
`tau_interior` is finite and nonnegative and defines the stricter numerical
log-barrier domain; it does not move the scientific confidence boundary.

When no finite raw point lies in the strict barrier interior, the default
exploration policy maximizes
\(\lambda_{\max}(\Sigma_{\mathrm{latent,std}})\). Diagnostic trace
exploration is registered but not a default. Candidate generation, refinement,
tie ordering, and recovery are described in the generated candidate support
table.

## Terminal recommendation policies

All terminal policies minimize their stated score and make zero simulator
calls and zero GP fits.

### Core default: evaluated standardized L1

`final_posterior_standardized_l1` searches evaluated points only:

\[
S_{\mathrm{std,L1}}(\theta_i)=
\sum_j\left|\mu_{\mathrm{std},j}(\theta_i)-
y^\star_{\mathrm{std},j}\right|.
\]

It uses posterior mean only and resolves an exact tie by earliest evaluation.

### Compatibility: evaluated expected standardized L2

`evaluated_posterior_expected_l2` searches evaluated points only:

\[
S_{\mathrm{expected,L2}}(\theta_i)=
\lVert\mu_{\mathrm{std}}(\theta_i)-y^\star_{\mathrm{std}}\rVert_2^2+
\operatorname{tr}\Sigma_{\mathrm{latent,std}}(\theta_i).
\]

It requires latent covariance and resolves an exact tie by earliest
evaluation.

### Physical squared-L2 policies

Both physical-L2 policies use exactly the same mean-only score:

\[
S_{\mathrm{phys,L2}}(\theta)=
\lVert\mu_{\mathrm{phys}}(\theta)-y^\star_{\mathrm{phys}}\rVert_2^2,
\qquad
\mu_{\mathrm{phys}}=\bar y+s\odot\mu_{\mathrm{std}}.
\]

- The compatibility API `evaluated_posterior_mean_physical_l2` searches
  evaluated points and uses the earliest exact tie. It is retained for generic
  callers and old evidence, but is not an ordinary graph-paper option.
- `continuous_posterior_mean_physical_l2` searches the complete bounded
  parameter domain using deterministic coordinate-seeded Sobol initialization
  and projected refinement. It accepts finite non-worsening refinement,
  otherwise returns the best finite raw candidate, and fails if none exists.

They differ only in search domain. They do not mix standardized and physical
coordinates or add uncertainty to the score. Both are dispatched directly by
the generic sequential runner as well as exposed through the public
recommendation API.

### Approved continuous physical-MAE policy

`continuous_posterior_mean_physical_l1_mae` searches the complete bounded
parameter domain and minimizes

\[
S_{\mathrm{phys,MAE}}(\theta)=\frac{1}{M}\sum_{m=1}^{M}
\left|\mu_{\mathrm{phys},m}(\theta)-y^\star_{\mathrm{phys},m}\right|.
\]

It uses only the posterior mean, deterministic coordinate-seeded scrambled
Sobol initialization, and projected refinement. Finite non-worsening refined
candidates are preferred; otherwise the best finite raw candidate is
returned. It makes zero simulator calls and zero GP fits. The stable
experiment-local diagnostic identifier remains unchanged in immutable Prompt
12/13 evidence and is not silently remapped to this core identifier.

The continuous physical-L2 and physical-MAE policies share one internal
bounded optimizer. The selected policy supplies the loss and policy-addressed
seed coordinate; this implementation sharing is not a third scientific policy.
Their public identifiers, formulas, result documents, and zero-call contracts
remain distinct.

Each projected-Adam restart evaluates convergence only between distinct
feasible states: the raw-start score and the first post-update score, then
successive post-update scores. A pre-update score cannot be compared with
itself to terminate the first iteration. Every restart retains its best finite
non-worsening post-update state; a later worsening or failed update cannot
displace that state. If no restart produces such a state, selection falls back
to the best finite raw candidate. Restart histories persist the actual update
count, best update index, best score, termination state, and typed failure.

### Approved surrogate-free observed physical-MAE policy

`evaluated_best_observed_physical_mae_v1` searches only evaluated points and
minimizes the physical-response MAE computed from the stored observed
simulator response. It uses no surrogate, GP fit, refinement, or simulator
call. Exact score ties resolve to the earliest evaluation. It is approved for
explicitly declared surrogate-free controls and is not a MUTACO D default.

## Composite Monte Carlo NEI policy

`composite_mc_nei_physical_mae_v1` is an approved simulator-independent q=1
acquisition over the joint latent multi-output posterior. Each sample retains
baseline--candidate and point--task covariance, is transformed to physical
response coordinates, and computes sample-wise improvement in physical MAE.
New observation noise, independent marginal sampling, and standardized-L2
substitution are prohibited. Common base samples remain fixed within one
optimization invocation. Failed or worsening local refinement recovers the
highest-ranked finite unevaluated raw candidate under
`best_ranked_finite_unevaluated_raw_v1`; exhaustion remains a typed failure.
The primary benchmark contract owns the numerical qualification and campaign
selection of this optional policy.

Its baseline conditional-sampling factorization uses the primary-contract
numerical policy `promoted_float64_cholesky_zero_jitter_v1`: promotion changes
arithmetic precision only and applies no covariance regularization.

## Duplicate policies

`reject_exact_duplicate_v1` is the conservative simulator-independent
default. `best_ranked_unevaluated_raw_v1` is an approved optional deterministic
recovery: when the refined proposal exactly duplicates an evaluated point, it
returns the highest-ranked finite raw candidate not already evaluated. The
recovery makes no new posterior call, simulator call, GP fit, or random draw;
exhaustion remains a typed failure. No near-duplicate tolerance is implied.

## Sequential accounting and randomness

For \(S\) completed sequential rounds, the in-memory runner performs exactly
\(S+1\) successful scientific GP fits, one per committed training-data
version. Each round makes one proposal and one accepted simulator evaluation.
Retries are allowed only for explicitly typed retryable simulator failures and
reuse the same logical evaluation seed.

The orchestration layer owns evaluation seeds. Scientific simulators derive
named coordinate-addressed child streams and must not depend on ambient
Python, NumPy, Torch, or CUDA RNG state. Dtype is scientific identity;
resolved device and timings are execution provenance.

The generic runner accepts an explicit terminal recommendation context when an
experiment must supply a target identity, nonzero target index, or a separate
policy-family seed authority. Ordinary callers retain the historical default
context. Graph-paper execution supplies the real target index and graph-paper
randomness family so the runner performs the configured physical terminal
policy exactly once without changing the identity-addressed inference streams.

## Supported versus quarantined behavior

The generated support tables are the machine-aligned list of active,
compatibility, diagnostic, and internal policies. Compatibility identifiers
remain callable through generic or explicitly compatibility-labelled APIs but
are not ordinary graph-paper selections. Diagnostic identifiers are imported
through `mutaco.diagnostics` or experiment-owned diagnostic modules; legacy
core enum members remain exact reader/reproduction shims. Unknown identifiers
fail validation without fallback.

Full-stream common random numbers, same-realization assessment,
`continuous_expected_physical_l2_diagnostic`, legacy-scale search, and the
retained graph oracle are historical/diagnostic evidence. They are excluded
from ordinary graph-paper configuration and are not D defaults.

## Provenance

Decisions 0002-0004 define the original policy, mathematical, GP, and design
contracts. Decisions 0017 and 0020-0025 record later duplicate, covariance,
recovery, diagnostic-quarantine, mixed-precision, physical-L2, and core
promotion changes. In a conflict, this current normative contract and the
machine registry govern supported behavior; dated decisions explain why it
changed.
