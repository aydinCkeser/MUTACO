# Terminal-domain ablation

## Scientific question

This zero-simulator-call ablation asks whether continuous terminal optimization
improves the final fitted MUTACO surrogate objective relative to restricting the
recommendation to the 40 evaluated B40 points.

For each Paper-120 target, the archived scalar-primary B40 trajectory is fixed.
One deterministic final structured GP is refitted from its 40 observations.
Both policies use that exact model, target, response inverse transform, dtype,
device, and component-mean posterior physical-MAE score:

- **B'** selects the earliest evaluated point attaining the minimum score.
- **C'** uses the canonical bounded continuous terminal optimizer.

Only the terminal candidate domain differs. The estimand is
`L_post(C') - L_post(B')`; negative values favor continuous optimization in
surrogate space.

## Claim boundary

This is a controlled recommendation-domain mechanism ablation on a newly
refitted final surrogate. C' is not the archived historical recommendation,
and comparing C' with that historical recommendation has no causal meaning.
These zero-call results do not establish held-out physical improvement. That
claim requires a separately declared target-paired held-out assessment of B'
and C' under one frozen method-independent context.

## Frozen implementation

The machine-readable protocol is
`experiments/paper/studies/graph_paper120/analysis/terminal_domain_ablation_b40_refit_v1.json`.
The implementation is
`experiments/paper/common/terminal_domain_ablation.py`, and the compact result
evidence is
`benchmarks/paper/terminal_domain_ablation_b40_refit_v1/manifest.json`.

The analysis must construct no simulator and make zero simulator calls. It may
fit GPs and optimize the posterior objective. Exact equality uses tensor value
equality; near equality uses normalized-parameter L-infinity distance at most
`1e-6`. Reported displacement and nearest-evaluated distances use normalized
parameter coordinates and the L2 norm.

## Held-out assessment attempt v1

The separately frozen local B'/C' held-out assessment passed its zero-call
preflight but is incomplete. Its first write-ahead slot (target 0, B') was
invoked once; the frozen outcome writer then rejected a read-only simulator
metadata mapping before an outcome could be persisted. The slot is therefore
`attempted_outcome_unknown` and cannot be repeated under that declaration.
No other slot was invoked, no retry occurred, and no confirmatory inference is
permitted. Any future execution requires a new declaration and separate
authorization.

## Held-out assessment v2

V2 used a new declaration, new target-specific paired context family, and the
same frozen B'/C' recommendations. The engineering repair recursively thawed
strict read-only simulator metadata before canonical JSON persistence; it did
not change simulator behavior or scientific values. All 240 local calls
completed with zero failures, outcome-unknown slots, or retries.

Across all 120 complete pairs, mean held-out physical MAE was `0.0359636` for
B' and `0.0257310` for C'. The mean C'-minus-B' difference was `-0.0102326`;
the frozen 10,000-resample paired percentile-bootstrap interval was
`[-0.0146451, -0.00602673]`. C' was better on 75 targets, tied on zero, and B'
was better on 45. Under the frozen decision rule, this supports a physical
benefit from continuous terminal optimization relative to evaluated-only
recommendation under the shared refitted final surrogate.

The paired target-level bootstrap quantifies variation across the Paper-120
target/context pairs; it does not separately estimate within-target Monte
Carlo uncertainty.
