# Cross-experiment scientific semantics

**Classification:** normative, prospective except where explicitly historical

**Authority subject:** paper-comparison vocabulary, cohort ownership, and
randomness ownership across budgets and method families

The machine-readable authority is `contract.json`. Its canonical constructor
and validator are
`experiments.paper.common.cross_experiment_contract`. This contract changes no
historical recommendation, random seed, simulator response, or evidence
identity.

## Distinct error quantities

The generic phrase *terminal error* is prohibited in scientific reports unless
it is immediately qualified as one of the following quantities:

\[
E_{\rm obs}=\min_{i\leq B}L(y_i,y^\star),
\]

the best error among noisy in-search simulator observations;

\[
E_{\rm raw}^{\rm GP}=L(\mu_B(\theta_{\rm raw}),y^\star),
\]

the final surrogate-mean error at the best candidate from the declared raw
terminal screen;

\[
E_{\rm term}^{\rm GP}=L(\mu_B(\hat\theta),y^\star),
\]

the final surrogate-mean error at the returned terminal recommendation; and

\[
E_{\rm heldout}=L(y^{\rm assess}(\hat\theta),y^\star),
\]

the physical error from a fresh held-out simulator evaluation of the frozen
recommendation. Surrogate errors and simulator errors are never substituted
for one another. A sum-L1 value and a component-mean MAE must also remain
explicitly distinguished.

Recommendation provenance uses the labels `best_observed`,
`raw_terminal_screen`, `historical_one_step_terminal`,
`corrected_refined_terminal`, and `heldout_assessment`. The last label denotes
an assessment operation, not a new recommendation policy.

## Historical evidence status

Evidence status and policy conformance are separate axes. In particular, the
historical MUTACO--SBI recommendation is valid evidence for the policy that
actually ran—1,024-point GP screening followed by one erroneous Adam update
per restart—and is simultaneously protocol-noncompliant for a claim about the
intended corrected multi-step optimizer. Historical files, identities, and
labels remain immutable. A future corrected comparison creates new versioned
evidence.

## Common paper cohort

The common backbone for the scalar primary population, a repaired Paper-120
SBI comparison, and a Paper-120 mechanism ablation is the ordered target list
in the [Paper-120 v3 manifest](../../experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json).
The frozen `contract.json` records this manifest's SHA-256 and original
repository address, `data/cohorts/paper120_v3/manifest.json`. These comparisons also require a declared common physical
simulator definition and configuration, parameter domain, response
coordinates, assessment fidelity, and metric schema.

Existing primary declarations call the SHA-256 of the complete budget-specific
MUTACO configuration a `simulator_configuration_identity`. That legacy field
also covers method, budget, campaign, and runtime settings, so B40 and B100 are
expected to have different values. Cross-budget physical compatibility is
instead checked from the versioned simulator-science projection containing
only `dtype` and the complete `simulator` block. The complete configuration
hash is still verified independently; it is neither discarded nor relabelled.

Numerical qualification may use an explicitly declared off-cohort target set.
Qualification evidence cannot be pooled with the primary population or
silently described as Paper-120 population evidence.

## Randomness ownership

There is no universal global seed. Every root seed and coordinate belongs to a
typed scientific role. The machine-readable contract distinguishes:

| Role | Method in owner | Pairing scope |
|---|---:|---|
| target construction | no | cohort target |
| shared initial design | no | target and budget within a comparison |
| search evaluation | yes | method-specific |
| acquisition or training | yes | method-specific |
| terminal recommendation | yes | method-specific |
| paired held-out assessment | no | target and budget within a comparison |

Method-independent ownership means that all methods in the declared paired
comparison receive the same random context; it does not eliminate simulator
calls. Method-addressed ownership prevents adaptive searches from accidentally
sharing or colliding random streams. A shared initial design is used only when
the study declaration says it is scientifically compatible.

The phrases *same noise configuration* and *same noise realization* are not
interchangeable. The former means the same physical stochastic law, noise
parameters, response definition, and simulator configuration. The latter means
the same evaluation seed/context. Paper learning and search streams are
generally method-addressed: competing methods are matched in stochastic law but
do not receive identical realizations. Common random numbers apply within a
declared held-out comparison after all recommendations have been frozen.

For the graph simulator, a new evaluation seed regenerates all three named
families: quenched disorder, initial state/phase, and thermal noise. The current
protocol does not hold quenched disorder fixed across sequentially queried
parameter vectors while changing only dynamical noise.

Terminal implementations may share a comparison-level root seed while using
method- or policy-addressed namespaces. Their semantic owners remain
method-specific because their candidate geometries need not align.

The scalar B40 and B100 campaigns are independent campaigns on the same
cohort. Their design, inference, terminal, and assessment root seeds must be
distinct across budgets unless a future contract explicitly declares a nested
cross-budget design. No such nesting is claimed for the current study.

Cross-budget synthesis may block B100-minus-B40 endpoint differences by the
shared target identity. Such differences remain descriptive comparisons of
independent campaigns. They are not within-run trajectories, do not establish
monotone improvement with budget for an individual execution, and do not
replace the within-budget paired held-out primary contrasts. Evidence from a
different study family, including retired vector-NEI attempts and the repaired
MUTACO--SBI assessment, cannot be pooled into the scalar-primary synthesis.

Current result-bearing assessment families use roots `92021324` (scalar B40),
`92021424` (scalar B100), `92021540` (repaired SBI B40), `92021600` (repaired
SBI B100), and `92021841` (terminal-domain B40 v2). Methods are paired within
each family, but these five families do not define cross-experiment pairs.
Their MAEs are descriptively comparable under the common cohort, simulator law,
and endpoint.

Historical Paper-v3 SBI B40, historical Paper-v3 SBI B100, and the completed
standardization comparison reused the target-context map rooted at `92020402`.
Those assessments are fresh relative to their respective learning data, but
they are not independent of one another across experiments. This historical
reuse is not a defect in their within-experiment paired comparisons and must
remain visible whenever cross-experiment independence is discussed.

## Freeze-before-assessment order

For each target and budget the mandatory causal order is:

```text
frozen target
  -> method-addressed searches
  -> all recommendations or failures frozen
  -> one method-independent assessment context
  -> one simulator evaluation per valid recommendation
  -> physical metrics and analysis
```

Assessment randomness, responses, and metrics cannot influence search,
surrogate fitting, acquisition, recommendation, fallback, or retry selection.

## Machine validation

The validator answers, without deriving or changing a scientific seed, which
study, target, budget, method, role, and index own an operation. It validates
the complete ordered Paper-120 cohort, simulator/configuration backbone,
metric and assessment policies, distinct role seeds, and independence of the
B40/B100 campaign pair. The future common evidence pipeline must consume this
contract rather than reimplement these rules.
