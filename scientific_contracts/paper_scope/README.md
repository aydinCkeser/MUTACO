# Paper experimental scope

**Classification:** normative paper-scope checkpoint

This document classifies the result-bearing Paper-120 evidence that may support
the current paper. It does not change any recommendation, simulator response,
evidence identity, or historical record.

## Current main-paper evidence

The primary optimization benchmark is complete on the ordered Paper-120 v3
cohort at independently executed budgets B40 and B100. At each budget it
compares MUTACO, scalar qLogNEI, and scrambled Sobol using fresh,
target-paired held-out physical component-mean MAE. B100 is not an extension of
the B40 trajectories.

The repaired SBI comparison is complete for MUTACO, NPE, and SNPE at B40 and
B100. Its repaired, budget-specific paired assessment is the authoritative SBI
endpoint; superseded historical assessments remain provenance only.

The controlled B40 terminal-domain mechanism ablation is complete. B' is the
best evaluated point under a freshly refitted final structured GP's
posterior-mean physical MAE, and C' is the continuous minimizer of the identical
objective under the same fitted GP. C' had the lower posterior score for all
120 targets. In the separately frozen v2 paired held-out assessment, mean MAE
was `0.0359636` for B' and `0.0257310` for C'; mean C'-minus-B' was
`-0.0102326`, with paired 95% percentile-bootstrap interval
`[-0.0146451, -0.00602673]` and win/tie/loss counts `75/0/45` for C'. The
population relative mean-MAE change was `-28.45%`.

The permitted claim is narrow: continuous terminal optimization demonstrated
a population-level physical benefit over evaluated-only recommendation in the
controlled B40 B'/C' refit experiment. This does not establish the effect for
B100, the archived historical recommendation, or terminal optimizers in
general.

## Supporting and diagnostic evidence

The completed response-standardization comparison is supporting evidence. Its
arms used distributionally matched but arm-specific learning streams, including
different initial-design realizations. Its paired held-out endpoint is valid,
but it is not a fixed-trajectory or shared-initial-data causal ablation.

Historical terminal-lineage forensics, exact-state recovery qualifications,
cross-budget synthesis, convergence diagnostics, and other zero-call audits
remain supporting or diagnostic evidence within their declared claim
boundaries. Qualification-only, prequalification, smoke, and off-cohort runs
are engineering evidence rather than Paper-120 population evidence.

## Excluded, historical, and incomplete evidence

- A scalar-MUTACO ablation is not planned for the main paper. Immediate
  scalarization changes the statistical estimand because, in general,
  `E[MAE(Y,y*)] != MAE(E[Y],y*)`; scalar MAE also mixes response-mean mismatch
  with intrinsic realization variability.
- A B100 terminal-domain ablation is optional reviewer-requested evidence, not
  part of the current main-paper program.
- Response standardization is not part of the main mechanism claim.
- Composite or multi-output NEI is a retired historical comparator. Failed and
  cancelled NEI qualifications remain historical provenance and must not be
  revived or pooled as current baseline evidence.
- The abandoned target-0 exact-GP/Barbun reconstruction path is engineering
  history, not scientific result evidence.
- Terminal held-out v1 is permanently incomplete: one attempted call, one
  `attempted_outcome_unknown`, no persisted success, and no retry.
- Superseded historical SBI assessments and all qualification-only,
  prequalification, smoke, failed, cancelled, and off-cohort records retain
  their registered provenance but are not current population endpoints.

Immutable evidence is classified in place; it is not deleted or rewritten.

## Randomness and assessment families

All completed Paper-120 comparisons use the same ordered Paper-120 v3 cohort,
target responses, physical simulator implementation, stochastic law, response
definition, and endpoint. Learning and search generally use method-specific
realizations. Thus, *matched learning noise* means the same stochastic
law/configuration, not identical random draws.

Within each declared comparison, frozen recommendations use a fresh
target-specific held-out context shared across methods or policies. For one
graph-simulator evaluation seed, quenched disorder, initial state/phase, and
thermal noise are all regenerated. Independent evaluators consume the common
context without sharing mutable simulator state.

The current assessment families are:

| Evidence family | Budget | Root seed | Pairing scope |
|---|---:|---:|---|
| scalar primary | B40 | `92021324` | MUTACO/qLogNEI/Sobol within target |
| scalar primary | B100 | `92021424` | MUTACO/qLogNEI/Sobol within target |
| repaired SBI | B40 | `92021540` | MUTACO/NPE/SNPE within target |
| repaired SBI | B100 | `92021600` | MUTACO/NPE/SNPE within target |
| terminal-domain v2 | B40 | `92021841` | B'/C' within target |

These families share the physical stochastic law but not the same held-out
realizations, so numerical MAEs are descriptively comparable across families
but are not paired across families.

Historical Paper-v3 SBI B40, historical Paper-v3 SBI B100, and the
standardization comparison reused the same target-context map rooted at
`92020402`. Each assessment was fresh relative to its own learning data, but
those assessments are not independent of one another across experiments:

```text
fresh relative to learning != independent across experiments
```

## Remaining work

No further simulator-bearing main-paper experiment is presently planned. The
Paper-120 B40/B100 SBI-zoo extension is complete: its off-cohort qualification,
240 shared datasets, 480 NLE/NRE recommendations, 480 repaired-family held-out
evaluations, and deterministic comparison analysis all passed their frozen
gates.

Remaining work is zero-call scientific review, figure and table selection,
evidence synthesis, and manuscript integration. These are not new experiments.

The manuscript randomness paragraph describing disorder reuse across queried
parameter vectors is known to be inconsistent with the authoritative
implementation. Correcting that paragraph remains pending because the current
manuscript worktree contains unrelated user edits; this checkpoint does not
stage or overwrite them.
