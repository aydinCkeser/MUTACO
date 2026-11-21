# Cross-experiment evidence and deterministic analysis

**Classification:** normative analysis machinery; scientific evidence remains
owned by its native immutable campaign.

The implementation in
`experiments.paper.common.evidence_analysis` enforces the vocabulary and
ownership rules in the cross-experiment scientific contract. Native evidence
is adapted into one hashed row per target, budget, and method. Adaptation does
not relabel or modify the source evidence.

Each row keeps the four error quantities distinct. Both component-mean MAE and
sum-L1 are recorded when a quantity is available; an unavailable quantity has
an explicit reason and is never imputed. Metrics are recomputed from physical
responses whenever the native evidence contains the response and frozen target.

The scalar-primary adapter validates the full ordered Paper-120 cohort, exact
method grid, budget-length trajectories, call ledgers, terminal-policy
provenance, metric recomputation, and byte-equivalent paired assessment
contexts. The historical Paper-v3 adapter validates endpoint and call-ledger
structure but records its assessment-context provenance as partial: only the
later SNPE fragment retained the detailed context in the consolidated bundle.
This limitation is not promoted into a new pairing claim.

Historical MUTACO evidence is labelled as valid evidence for the executed
1,024-point screen plus one-step terminal policy and protocol-noncompliant for
the current corrected-refinement claim. NPE and SNPE retain their native
posterior-sample-MAP policy label; that operation is not falsely classified as
a GP terminal stage.

Analysis options, including the bootstrap algorithm, seed, and resample count,
are frozen in a hashed specification. Pair-specific bootstrap streams are
derived deterministically from that analysis seed and method names. Analysis
randomness never enters the simulator or changes scientific evidence.

The mechanism-ablation adapter is intentionally prospective and rejects
nonempty input until that study's native record schema is separately frozen.

The tracked B40 acceptance package is reproduced with:

```text
python -m experiments.paper.common.evidence_analysis_cli --repository-root . --check
```

The command verifies the native export hash, regenerates every row, statistic,
audit, and package identity in memory, and compares both files and SHA-256
sidecars. It performs no simulator operation.

## Scalar-primary B40/B100 synthesis

The tracked cross-budget synthesis is materialized by
`experiments.paper.common.cross_budget_synthesis_cli`.  It consumes only the
frozen B40 and B100 canonical analysis rows and reports.  Before synthesis it
checks their fixed file hashes and evidence identities, validates the complete
target--method grids, and reproduces both native reports from their respective
tracked analysis specifications.

B40 and B100 are independent campaigns on the same ordered Paper-120 cohort.
Consequently, target identity may be used as a blocking factor for a
descriptive B100-minus-B40 endpoint summary, but B100 is not a continuation of
B40.  The synthesis cannot support a per-run monotonicity claim.  Its seeded
cross-budget bootstrap intervals quantify descriptive target-blocked
uncertainty and are not promoted to new confirmatory hypotheses.

Confirmatory interpretation remains within budget: the primary endpoint is
paired held-out physical-response component-mean MAE. Search observations,
terminal-surrogate errors, numerical fallback counts, and symmetric persisted
timings are secondary or descriptive diagnostics. Missing terminal quantities
are retained as unavailable and never imputed. Retired vector-NEI and repaired
MUTACO--SBI evidence are excluded from this synthesis.

The package is reproduced with:

```text
python -m experiments.paper.common.cross_budget_synthesis_cli --repository-root . --check
```

This command performs no simulator calls, GP fits, search executions, terminal
optimizations, held-out assessments, or scheduler submissions.
