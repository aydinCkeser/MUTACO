# Held-out assessment protocol

Final comparative response error uses one fresh held-out forward assessment at
each method's final theta. It uses the same declared fidelity for all methods,
is not fed back into inference, and is accounted outside inference/search
budgets. Parameter error requires no simulator assessment.

For ordinary graph-paper campaigns, the authoritative assessment is one fresh
canonical sparse M=20 evaluation. Dense force grids are visualization-only,
belong to separately accounted representative-figure workflows, and must not
replace sparse assessment or enter population performance metrics. When a
representative dense replay is required, its canonical force points must
reproduce the sparse realization under the graph simulator's isolated dense
RNG extension; supplemental conditions remain descriptive visualization data.

Existing benchmark and Paper-100 records implement this protocol with
method-addressed assessment seeds, strict outside-budget accounting, persisted
responses, and failure retention. Those identities and results remain
immutable.

The prospective
[`paper_final_decision_and_paired_assessment_v1`](../paper_comparison_protocol/README.md)
protocol instead uses `paired_held_out_assessment_v1`: after every method's
recommendation or failure is frozen, each valid recommendation receives one
fresh assessment call under the same target-addressed, method-independent
stochastic context. Method inference streams remain independent. Paired CRN
changes cross-method covariance, not call accounting, and cannot feed back
into a recommendation.

Individual campaigns must declare which versioned assessment policy applies,
as well as fidelity, seed coordinates, and assessment count, before execution.
