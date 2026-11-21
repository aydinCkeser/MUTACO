# Draft TRUBA design: MUTACO--NEI--Sobol primary benchmark

**Status:** qualification diagnosis; non-executable; no submission authority

**Scientific authority:**
[`scientific_contracts/primary_benchmark_protocol/README.md`](../../scientific_contracts/primary_benchmark_protocol/README.md)

This document records the proposed TRUBA implementation boundary and open
engineering decisions. It is not an environment profile, population
declaration, command approval, exact-source receipt, or scientific launch
authorization. No shell command in this design is approved for execution.

## Current implementation boundary

The zero-scientific-call foundation now provides:

- `mutaco-primary-benchmark-population-v1` declarations and
  `mutaco-primary-target-method-record-v1` records;
- budget-addressed target/method and method-independent held-out contexts;
- one-sequence shared-prefix Sobol construction;
- callback-based target-centric all-method freeze-before-assessment execution;
- `mutaco-nei-numerical-qualification-declaration-v1` with the 700-call preview
  and no assessment surface;
- joint latent point-task posterior covariance, composite physical-MAE NEI,
  raw recovery, surrogate-free Sobol terminal selection, and reference method
  runners;
- deterministic endpoint, trajectory, paired-difference, and runtime table
  reconstruction from primary records;
- a frozen five-point Q1 theta declaration, constructed response ledger,
  post-construction numerical declaration, three numerical tiers, exact
  snapshot schedule, deterministic tier selector, capped append-only Q1 and
  primary worker orchestration, collectors, and compact export;
- a tracked Q1 validation profile and exact-source validation dispatch whose
  submission path remains hard-blocked until frozen responses and separate
  authorization exist;
- deterministic fake-simulator validation of 123/303-call previews, B40
  execution accounting, shared geometry, method-distinct search randomness,
  paired assessment, and zero-call terminal decisions.

The Q1 response ledger and post-construction declaration now exist. The first
Q1 qualification attempt stopped consistently across all five workers after
their 16-call initial designs: 80 simulator calls succeeded, no trajectory
record or tier result was produced, and the wrapper preserved only the outer
exact-budget failure. Zero-call reproduction identified the position-16
snapshot-state hash as the failure boundary: exact GP state includes legitimate
positive-infinity constraint bounds, which strict canonical JSON rejected.
The evidence layer now encodes signed infinities explicitly while continuing
to reject NaNs, and it persists full failed trajectory attempts before budget
validation. The real GP/snapshot reproduction now completes the position-16
snapshot and three tier observations. There is no selected production tier or
tracked Q2/B40/B100 profile. The new retry declaration and exact source pass
zero-call validation and the focused TRUBA encoding regression. The tracked
real-GP compute smoke passed on `barbun13` as job `6244651`, producing the
position-16 snapshot, three tier observations, and four explicitly encoded
positive infinities from 16 deterministic synthetic responses with zero
simulator calls. The scientific retry remains separately unauthorized.
The later authorized retry, job `6244676`, failed after 436 successful calls
and zero simulator failures. One worker completed B40 and B100; four workers
failed at target-dependent B40/B100 trajectory-attempt stable hashes because a
broader returned result still contained non-finite evidence. No qualification
result or production tier exists. Zero-call diagnosis identified expected
`-inf` sentinels in `result.proposal_diagnostics[*].raw_values[*]`, outside the
earlier snapshot-only encoder. The evidence layer now has one recursive
path-aware canonicalizer for complete attempts, reserved versioned
signed-infinity tags, exact-path NaN rejection, semantic hashes derived from the
canonical object, and pre-hash safe failure envelopes. Production-shaped
B40/B100 lifecycle and identity regressions pass with zero simulator calls.
Historical attempts stay immutable. Any new declaration, source preparation,
or retry remains separately unauthorized.

The separately approved zero-call preparation gate now freezes a new canonical
qualification declaration and companion run declaration. They preserve every
scientific field from the original Q1 contract, change only the declaration
identity/output address, name `ae8dfdf` as the evidence-repair ancestor, and
make both historical failed jobs ineligible for tier selection. Exact-source
materialization and validation remain required before any retry authorization.
That lifecycle gate now passes at exact source commit `00c81dd`: the clean
remote tree imports the new declaration/profile, validates all five frozen
targets and the graph configuration, reports pairwise-disjoint source/log/output
paths, and executes no simulator call or submission. Scientific retry remains
unauthorized.

## Proposed campaign decomposition

The production study should use separate B40 and B100 declarations and jobs.
This preserves their independent campaign and seed identities and avoids an
implicit B40 prefix of B100.

Each scientific work unit should own one target-budget pair and execute all
three methods before creating the all-method decision freeze and paired
assessment. This keeps shared initial geometry, terminal-start geometry, and
held-out pairing inside one causal unit. Sharding a target's methods across
independent jobs would require a new durable cross-job freeze protocol and is
not the preferred design.

The initial candidate production layout is 40 workers with three targets per
worker:

| Campaign | Calls per target | Calls per three-target worker | Global calls |
|---|---:|---:|---:|
| B40 | 123 | 369 | 14,760 |
| B100 | 303 | 909 | 36,360 |

This is a logical shard proposal, not frozen scheduler geometry. NEI joint
sampling and optimization may change CPU time and memory enough that the
number of simultaneous workers, CPUs per worker, and workers per node must be
qualified independently of the scientific shard identities.

### Prequalification engineering integration smoke

The approval-gated declaration
`primary_prequalification_b40_integration_smoke_v1.json` exercises the actual
one-target three-method B40 path before Q1 is complete. It uses the already
declared high Q1 controls under the explicit identifier
`tier_high_provisional_engineering_only`; it does not select a production tier
and is ineligible for Q1 tier selection and primary performance reporting. The
one frozen Q1 off-cohort target is reused without a construction call. Its hard
cap is 120 search calls plus three paired assessments, for 123 total calls.

The common worker preserves the shared 16-point geometry, method-distinct
search contexts, method-independent terminal starts for MUTACO and NEI, one
all-method freeze before assessment, and one paired held-out context. NEI
proposal evidence includes the registered promoted-float64 baseline
factorization diagnostic; MUTACO and NEI terminal evidence includes complete
restart histories. Sobol fits no surrogate. The tracked profile remains
disabled and scientific submission is a separate authorization gate.

The first authorized execution, job `6247524`, completed all 120 search calls
and preserved the immutable all-method freeze, then failed before the first
held-out simulator call because the assessment mapping lacked the production
`SimulationContext` address fields. Its assessment-only rescue declaration is
bound to that freeze and permits no search rerun. Preparation and exact-source
validation are zero-call gates; any future three-call execution requires a
separate explicit authorization.

## Qualification phases

### Phase Q0: zero-call contract validation

The local pinned `mutaco-sbi` test suite and tracked TRUBA validate-only path
must check declarations, identities, call previews, shard coverage, output/log
separation, and worker dry runs without invoking the simulator. Exact-source
preparation remains owned by `infrastructure.truba.lifecycle prepare-validate`.
It must produce a fresh clean detached source, a new append-only remote root,
and a receipt reporting zero scientific calls and no submitted job.

### Phase Q1: NEI numerical qualification

Five separately frozen off-cohort targets run independent NEI-only B40 and
B100 trajectories. The hard search-call ceiling is 700. There are no held-out
assessments and no MUTACO/Sobol comparator results. Saved artifacts include
training data, frozen response transforms, fitted GP state and identity,
snapshot position, tier candidates, reference-batch seeds and values,
normalized regret, optimizer/fallback status, timing, and memory provenance.

The qualification runner is not yet designed. Reusing the current Paper-v3
all-method assessment runner would incorrectly require terminal assessment and
method-complete decision freezes. The implementation should either generalize
the declaration/runner with an explicit numerical-qualification protocol or
add a narrowly typed qualification runner that cannot be mistaken for primary
population evidence.

### Phase Q2: independent B40 production-path smoke

After the NEI production tier is frozen, one off-cohort target runs all three
methods through the actual production pathway. Its hard cap is 120 search
calls plus three paired assessments, totaling 123. Together Q1 and Q2 have a
planned 823-call ceiling, excluding separately declared target construction.

The previously considered 783-call alternative would reuse one provisional
Q1 NEI trajectory and add only MUTACO, Sobol, and assessment. It is rejected
as the default because it would not test the final frozen NEI tier through the
real all-method production runner. It remains only a documented alternative
until the implementation review closes it definitively.

The prequalification engineering smoke above is not this formal Q2 phase. Q2
still requires a Q1-selected production tier and a newly declared independent
target.

### Phase Q3: resource qualification

Before population preparation, run zero-simulator-call compute benchmarks on
representative frozen GP snapshots. Measure NEI baseline factorization,
conditional candidate sampling, raw scoring, refinement, terminal optimization,
peak resident memory, Torch thread scaling, and artifact volume. Resource
qualification cannot select scientific NEI settings; it sizes execution for
the already frozen tier.

## Worker and scheduler design

Scientific shard identity should remain target-position based and independent
of scheduler rank. A worker may process three targets serially while numerical
kernels use pinned CPU threads. The final layout must satisfy Barbun's
20-core-per-node allocation rule without reviving hard-coded thread guards in
scientific workers.

Values that remain open pending Q3 are:

- scientific workers per job and scheduler tasks per node;
- CPUs and Torch threads per worker;
- memory per node and whether B40/B100 need different profiles;
- job wall-time limits;
- whether known-problematic historical node exclusions remain warranted;
- whether one worker should retain all three targets after a terminal
  engineering failure or emit completed append-only slots and stop.

Profiles must supply execution controls; core methods cannot branch on paper
choices or scheduler layout. The submission wrapper validates by default.
`--submit` or an equivalent surface remains a separate explicit scientific
authorization gate and must not be added to the exact-source lifecycle command.

## Outputs, collection, and transfer

Worker roots, scheduler logs, and source roots are distinct and append-only.
Every worker persists a start record, declaration and source identities,
per-target method attempts, all-method decision freezes, assessment randomness,
terminal record, and exact call ledger even after failure. Automatic campaign
retry is prohibited; any recovery uses a new declaration and output root after
explicit review.

Unlike earlier compact Paper-v3 exports, the primary compact export must retain
the complete 20-component response trajectory and required GP/acquisition
diagnostics for every call. It should exclude reconstructable Monte Carlo
sample tensors and production GP model tensors. Q1 qualification exports are a
separate class and retain the frozen GP snapshots required for convergence
audit.

Global collection must verify:

- exact target-method-budget coverage and absence of duplicate slots;
- B40/B100 declaration and seed-family separation;
- one shared 16-point geometry identity per target-budget pair;
- method-distinct search evaluation contexts;
- exact 40/100 search counts for completed methods;
- zero terminal calls and assessment only after the all-method freeze;
- method-independent held-out contexts;
- balanced attempted/successful/failed call ledgers;
- artifact hashes and reconstruction of endpoint and trajectory tables.

Transfer to the local Paper Lab remains a separate verified operation after
successful global collection. Plotting and statistical reports consume only
the fetched, consolidated evidence and add zero scientific calls.

## Implementation work packages for later approval

1. **Implemented:** add budget-addressed three-method primary records and
   numerical-qualification declarations.
2. **Implemented:** add caller-supplied shared Sobol-prefix plumbing and the
   response-independent continuation arm.
3. **Implemented:** add joint latent posterior sampling and registered composite
   NEI without changing canonical MUTACO behavior.
4. **Implemented:** add method-appropriate terminal decisions and trajectory
   artifact schemas.
5. **Implemented at the orchestration boundary:** Q0/Q1/Q2 validation, hard
   caps, append-only worker persistence, collectors, compact export, and
   deterministic analysis tables.
6. **Implemented:** the canonical graph Q1 driver, exact five-call construction,
   frozen response ledger, and post-construction declaration.
7. **Implemented:** exact-source preparation and zero-call validation for Q1.
8. **Failed attempts preserved; evidence diagnosis and repair complete:** job `6244497`
   stopped after 80 successful initial-design calls. The signed-infinity
   snapshot serialization defect and swallowed failure evidence are repaired
   with zero scientific calls. The new append-only declaration, output root,
   exact-source validation, and the one-job zero-simulator compute smoke are
   complete. The authorized retry job `6244676` then stopped after 436
   successful calls at a broader non-finite trajectory-result hashing boundary.
   The complete Q1 evidence boundary, deterministic semantic hashing, safe
   failure envelopes, and production-shaped B40/B100 zero-call regressions are
   now repaired and verified. A one-target B40 integration smoke subsequently
   exercised MUTACO, NEI, and Sobol; its separately rescued held-out results
   remain engineering-only. A fresh five-target Q1 attempt is declared for
   zero-call exact-source validation, with all historical Q1 attempts excluded
   from tier selection. Exact commit `f39df98` passed that validation with no
   simulator calls or submission. Scientific submission remains separately
   unauthorized.

No work package is authorized by this design document.
