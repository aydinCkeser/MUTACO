# Scalar qLogNEI primary benchmark protocol

Status: normative candidate under zero-call implementation validation. No
scientific execution is authorized by this document.

## Comparison and estimand

The primary comparison has three methods at independent simulator budgets
`B40` and `B100` over the frozen 120-target cohort:

- MUTACO: the registered structured, compatibility-aware method.
- scalar qLogNEI: conventional objective-level noisy Bayesian optimization.
- Sobol: nonadaptive, space-filling search without a surrogate.

For target response `y*` and a stochastic response `Y(theta)`, the common
scientific loss is

`L(theta, Y) = mean_j |Y_j(theta) - y*_j|`

in physical response coordinates. The scientific estimand is
`R(theta) = E[L(theta, Y)]`. One noisy simulator response gives an unbiased
observation of `R(theta)`; it is not itself the latent risk.

The comparison is end-to-end: structured response modelling and
compatibility-aware selection versus a standard scalar objective-level BO
baseline versus nonadaptive search. It does not isolate acquisition policy
while holding the surrogate representation fixed.

## Scalar qLogNEI

The baseline uses the pinned, official BoTorch implementation:

- `SingleTaskGP` with input normalization and scalar outcome standardization;
- training response `-L(theta, Y)` so larger is better;
- `qLogNoisyExpectedImprovement` with a Sobol QMC normal sampler;
- official `optimize_acqf` bounded optimization;
- a deterministic 256-point raw Sobol fallback, scored by the same qLogNEI
  acquisition, when the optimized result is nonfinite or an exact duplicate;
- sequential `q=1` selection.

Here `q=1` means one new design and therefore one simulator call per BO
iteration. It does not mean one posterior sample. `n_mc` controls the number
of Monte Carlo samples used to estimate the acquisition.

The 16-point scrambled Sobol initial design is shared by all methods for each
target-budget campaign. Scalar qLogNEI then chooses `B-16` designs
sequentially. New population evidence must persist the optimized candidate and
value, their finiteness, any exact-duplicate index, the exact fallback trigger,
the returned candidate source, fit/proposal failures, timing, seeds, and
library warnings.

The frozen production settings are `n_mc=256`, `raw_samples=256`, 10
restarts, 64 acquisition-optimizer iterations, and 100 GP-fit iterations.
Terminal optimization uses 1,024 raw samples, 10 restarts, and 64 optimizer
iterations. The canonical settings document and its semantic identity are
`experiments/graph_inversion/configs/paper/scalar_qlognei_primary_settings_v1.json`
and `scalar-qlognei-primary-settings-v1-0f64f21df9b925a5508d90420f1179941174d300d0d38a11ac162044f3540a6d`.

## Precision and factorization policy

Both adaptive arms use float32 end to end in this comparison. Scalar qLogNEI
must not silently promote a failed production calculation to float64.

The official GPyTorch float32 Cholesky policy is frozen with initial jitter
`1e-6` and at most three factorization attempts. Every emitted jitter warning
and magnitude is recorded. If these bounded attempts do not recover the
factorization, the trajectory terminates with a typed failure before another
simulator call. Float64 replay is permitted only as a zero-call diagnostic and
cannot rescue or alter a scientific trajectory.

Cholesky jitter is numerical regularization of a matrix factorization. It is
distinct from the GP likelihood's learned observation noise and its parameter
constraint. Evidence must not describe either quantity as the other.

This policy intentionally has no production sequence of float32, then
float64, then custom jitter. Such a sequence would make numerical precision
method-dependent and would obscure the comparison.

## Terminal recommendations

After its simulator budget is exhausted, scalar qLogNEI continuously maximizes
the scalar GP posterior mean of negative physical MAE, equivalently minimizing
posterior mean scalar loss. MUTACO continuously minimizes physical-response
posterior-mean MAE under its maintained surrogate. Both terminal optimizations
make zero simulator calls and use the same declared numerical geometry.

Sobol remains surrogate-free and recommends the earliest evaluated design
with minimum observed physical-response MAE. No post-hoc GP is fitted to
Sobol for the primary comparison.

## Held-out assessment and accounting

For each target and budget, all three frozen recommendations are evaluated
under one shared, method-independent held-out simulator context. This is one
fresh noisy realization per recommendation, paired by target and budget. It
is not a 120-replicate estimate at one theta: 120 is the number of distinct
targets in the population comparison.

Each target-budget campaign therefore uses `3B` search calls and three
held-out calls. Terminal optimization uses zero calls. Across 120 targets:

- B40: 14,400 search + 360 assessment = 14,760 calls.
- B100: 36,000 search + 360 assessment = 36,360 calls.
- full comparison: 51,120 calls.

Budgets are independent campaigns. Automatic scientific retry is forbidden;
attempt evidence is append-only.

## Decision-side timing

All three methods persist the same non-overlapping `perf_counter` timing
fields:

- `simulator`: time inside the search simulator-call boundary;
- `gp_fit`: all GP-fit boundaries, and exactly zero for Sobol;
- `acquisition`: all adaptive proposal boundaries, and exactly zero for Sobol;
- `terminal`: continuous recommendation for the adaptive methods and
  evaluated-best selection for Sobol;
- `other`: the residual preprocessing and orchestration time;
- `total`: the outer method boundary through the frozen recommendation.

The five components must sum to `total` within numerical clock tolerance.
Nested method-specific diagnostics may be more detailed, but are not summed
again into the common fields. In particular, MUTACO proposal
`timings.total_seconds` is the proposal contribution; its candidate-generation,
raw-scoring, and refinement subfields are descriptive children of that total.

Held-out assessment, evidence serialization, scheduler time, and Q1-only
qualification work are outside decision-side `total` and must be reported
separately. Timing supports descriptive compute characterization; a strong
compute-efficiency claim additionally requires controlled hardware and worker
placement.

## Historical boundary

The former composite multi-output NEI protocol and all results, failures,
logs, declarations, and hashes remain immutable historical engineering
evidence. They are ineligible for scalar-qLogNEI qualification, numerical
selection, or population performance claims and must not be pooled with a
future scalar-qLogNEI campaign.

Before scientific execution, the scalar implementation settings, exact source,
environment, declaration, target ledger, and call cap require separate
zero-call validation and explicit submission authorization.

## Current prequalification state

The frozen implementation passed fit, official qLogNEI proposal, and terminal
optimization replays on five hash-verified archived failure-state prefixes at
baseline sizes 16, 34, 40, 60, and 66. All remained float32, used no precision
fallback, preserved ambient Torch RNG state, and made zero simulator calls.
The compact evidence is
`results/paper/graph_scalar_qlognei_zero_call_validation_v1/manifest.json`.

Because no archived failed trajectory reached 100 observations, a separate
synthetic exact-100-point regression covers the B100 covariance, official
proposal, and terminal path without calling a simulator. This synthetic check
is numerical capacity evidence only, not scientific performance evidence.

The completed one-target B40 prequalification used 120 search calls plus three
paired held-out assessments, for 123 calls total. The separately prepared
one-target B100 prequalification declaration is capped at 300 search calls plus
three paired held-out assessments, for 303 calls total. Its TRUBA profile has
execution disabled. B100 preparation and exact-source validation authorize no
scientific submission; that remains a separate explicit gate.

The separately authorized B100 attempt, job `6248220`, subsequently completed
all 303 calls with zero failed calls and no automatic retry. All three method
records and the automatic collection completed. Scalar qLogNEI remained
float32 and used no precision fallback; 4 of 84 adaptive proposals used its
frozen deterministic raw-candidate fallback after optimizer-level failures.
This diagnostic is part of the method evidence and is not silently recoded as
an official optimizer proposal. The compact archival manifest is
`results/paper/graph_scalar_primary_b100_prequalification_smoke_6248220/manifest.json`.
As with B40, this one-target result is engineering prequalification evidence,
not population-performance evidence.

Zero-call forensics across the four B40 and four B100 fallback states reproduced
five exact historical fallbacks; all five were triggered by an optimized point
that exactly duplicated the baseline. The fallback returned finite, unique
points, preserved ambient Torch RNG state, and remained on the same qLogNEI
objective. Three Linux states did not reproduce on Windows despite matching
library versions, so their historical invalidity subtype is irrecoverable: the
old records retained only the compound `ScalarQLogNEIError` class. This is an
evidence-provenance defect, not a demonstrated acquisition-policy defect. It
must be repaired without changing candidate selection before a population
campaign is frozen. The forensic evidence is
`results/paper/graph_scalar_qlognei_fallback_forensics_v1/manifest.json`.

That diagnostic-only repair is now complete. Future proposal evidence records
the rejected optimizer candidate and acquisition value when finite, explicit
nonfinite-kind tags otherwise, exact baseline-duplicate indices, and the
underlying exception type and message. Official optimizer success records a
null trigger. The GP, qLogNEI construction, optimizer settings, deterministic
raw-candidate fallback generation and ranking, seeds, float32 policy, and
terminal recommendation are unchanged. Historical records remain immutable;
their identities were verified unchanged. The zero-call repair evidence is
`results/paper/graph_scalar_qlognei_fallback_provenance_repair_v1/manifest.json`.
No population preparation or scientific submission is authorized by this
repair.

## Population preparation

The B40 and B100 Paper-120 population campaigns are now locally prepared from
the frozen cohort under separate declarations and campaign seeds. Each budget
uses a 120-task Slurm array with one target and one exclusive 20-core node per
task; worker packing, automatic retry, and Slurm requeue are disabled. The
declared caps are 14,760 calls for B40 and 36,360 for B100. A zero-call
collector is designed as a separate `afterany` dependency so failed worker
evidence is still inspected rather than erased by a success-only dependency.

All 120 target files and hashes, both complete shard maps, declaration and plan
identities, method identities, settings identity, call accounting, and the
fallback-provenance requirement passed local zero-call validation. The compact
preparation evidence is
`results/paper/graph_scalar_primary_population_preparation_v1/manifest.json`.
The declarations remain unexecuted and their TRUBA profiles have execution
disabled. Exact-source transfer, remote directory creation, and scientific
submission require later explicit gates.

The B40 and B100 exact sources were subsequently materialized in separate
TRUBA namespaces at commit `ce41544651b72a5197985aebb507898f4d4ec9da`.
Each imported the pinned SBI environment, revalidated all 120 one-target
shards, reproduced its declaration and plan identities and call preview, and
reported zero scientific calls and no submitted job. The output roots remain
absent. These validations authorize neither campaign; B40 and B100 submission
remain separate explicit decisions.

The first authorized B40 submission attempt was rejected by Slurm before job
creation because the v1 physical array `0-119%120` exceeded the cluster's
`MaxArraySize=100`. The preflight completed, but no worker output, scheduler
job, collector, or simulator call was created. This is preserved as a rejected
execution attempt, not a scientific trajectory.

Execution revision v2 represents the same 120 logical workers as two physical
arrays: task IDs `0-99` map to logical ranks `0-99`, while task IDs `0-19`
with offset 100 map to ranks `100-119`. A single collector depends `afterany`
on both arrays. Exact one-time rank coverage, both array specifications, fresh
B40/B100 exact sources, and all scientific invariants passed zero-call
validation. The repair evidence is
`results/paper/graph_scalar_primary_array_limit_repair_v1/manifest.json`.
Fresh submission authorization is required.

The authorized v2 B40 submission attempt created array A as job
`6248524`, then Slurm rejected array B with `AssocGrpSubmitJobsLimit`. Sixty
tasks briefly allocated before the orphaned array was cancelled; the worker
output root remained absent, so the attempt made zero scientific calls. Array
B and the collector were never created. Scheduler logs remain immutable
failed-attempt evidence.

Execution revision v3 uses one physical array `0-59%60`. Physical task `i`
runs logical workers `i` and `i+60` sequentially in separate Python processes,
each with the unchanged one-target shard, 20-core setting, 123-call cap, and
worker evidence directory. A nonzero first worker does not suppress the second;
the physical task returns nonzero after both attempts if either failed. The
array is submitted held, the collector is attached with `afterany`, and the
array is released only after both job identifiers validate. A pre-release
failure cancels every partially created job. The 60 array elements plus one
collector remain below the demonstrated 100-job association limit. This is an
execution-only repair: all scientific policies, seeds, targets, ordering, and
the 14,760-call global cap are unchanged.

The first v3 scheduler test-only pass accepted the array but rejected the
collector because Barbun requires every job, including a zero-call collector,
to request 20 cores per node. No job or call was created. Revision `v3r1`
retains the same execution geometry and gives the collector the site-required
20-core allocation in a fresh namespace.

Revision `v3r2` replaces hold-and-release with a ten-minute delayed-start
transaction.  The submitter creates the 60-element array, attaches the
20-core zero-call `afterany` collector, and cancels every partially created job
if the transaction cannot be completed before the array becomes eligible.
This geometry completed the full B40 population campaign and is therefore the
demonstrated scheduler design for B100 as well.

The B100 population preparation has been regenerated in a fresh `b100-v3r2`
namespace with the same execution-revision-4 geometry: 60 exclusive 20-core
nodes, with physical task `i` running logical target workers `i` and `i+60`
sequentially.  Each logical worker retains its independent 303-call cap, so the
global bound remains 36,000 search calls plus 360 paired held-out calls, or
36,360 calls total.  All 120 zero-call shard previews, the common B40/B100
cohort and randomness contract, declaration and plan identities, timing fields,
append-only collector, compact-export schema, and deterministic analysis
adapter passed the preparation audit.  The frozen analysis specification uses
seed `92024100` and 10,000 percentile-bootstrap resamples.  The compact audit is
`benchmarks/paper/graph_scalar_primary_b100_population_preparation_v3r2/manifest.json`.
The earlier B100 `v2` source remains historical scheduler-preparation evidence;
its scientific policy is unchanged, but its 100-plus-20 two-array submission
layout is superseded.  No source transfer, TRUBA job, or simulator call was made
by this audit.

The exact B100 `v3r2` execution source was subsequently materialized from
commit `649235bcb7e8773f5f4b0425ad111314d7d3b1f9` at
`/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b100-v3r2/source-exact`.
The tracked lifecycle validated the clean commit, disjoint source/output/log
paths, pinned runtime, declaration and plan identities, complete 120-shard
coverage, and 36,360-call ceiling.  Its receipt records zero scientific calls,
no submitted job, and no submission authority.  Scientific execution remains
a separate explicit gate.

## B100 population result

Jobs `6251851` and `6251852` completed the frozen B100 campaign without a
retry.  All 60 physical array tasks, 120 logical target workers, and the
collector completed successfully.  The final ledger contains 36,000 search
calls and 360 paired held-out assessments: 36,360 attempted and successful
calls, zero failed calls, and 360 target--method records.  The independent
archive audit verified all 842 JSON/hash pairs, the scheduler inventory and
logs, the compact transport hashes, exact cohort order, exact paired assessment
contexts, and the complete 120-target by three-method grid.

The primary held-out physical-response component-mean MAE is:

| Method | Mean | Median | Target wins |
|---|---:|---:|---:|
| MUTACO | 0.0196113 | 0.0152868 | 80 |
| scalar qLogNEI | 0.0255210 | 0.0233433 | 33 |
| Sobol | 0.0590539 | 0.0579599 | 7 |

The paired MUTACO-minus-qLogNEI mean difference is `-0.00590966`; its frozen
10,000-resample 95% percentile-bootstrap interval is
`[-0.00824921, -0.00366315]`, with MUTACO better on 84 targets and qLogNEI
better on 36.  MUTACO-minus-Sobol is `-0.0394426`
(`[-0.0448504, -0.0341739]`; 111 versus 9), and qLogNEI-minus-Sobol is
`-0.0335329` (`[-0.0390637, -0.0281684]`; 110 versus 10).

All qLogNEI trajectories completed in float32 with no precision fallback and
no unrecovered algorithm failure.  Of 10,080 adaptive proposals, 10,078 used
BoTorch's optimized candidate and two used the frozen deterministic raw
fallback.  Direct numerical warnings reported jitter no larger than `1e-4`.
Some recovered optimizer-retry warning text mentions internal escalation up to
`0.1`; this is retained separately from direct jitter events and from the two
explicit raw fallbacks.  It is diagnostic evidence, not an exclusion rule.

The tracked result is
`benchmarks/paper/graph_scalar_primary_b100_v1/manifest.json`.  The full compact
export and raw worker evidence remain outside Git under their immutable local
and TRUBA archive roots.  Archival and deterministic analysis added zero
simulator calls and submitted no jobs.
