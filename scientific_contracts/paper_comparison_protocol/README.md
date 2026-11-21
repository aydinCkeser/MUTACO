# Paper comparison final-decision protocol

**Classification:** normative, current

**Authority subject:** versioned MUTACO--SBI final decisions, metrics,
assessment identities, and call accounting for future paper comparisons

## Protocol and record identities

The first approved protocol is
`paper_final_decision_and_paired_assessment_v1`. Target-method records governed
by it use schema `mutaco-paper-target-method-record-v3`.

The protocol and schema are prospective. Existing
`mutaco-paper-target-method-record-v2` records retain their original
posterior-mean SBI recommendations, method-addressed assessment identities,
legacy metric names, content identities, hashes, and interpretation. A v2
record is never relabelled or reused as v3 evidence.

## Method-appropriate zero-call decisions

Every final decision is frozen before assessment and makes zero simulator
calls.

### Upgraded graph-paper MUTACO

MUTACO retains the authoritative
`continuous_posterior_mean_physical_l1_mae` terminal policy from the current
MUTACO D contract. This protocol does not alter its formula, `16 + 24`
allocation at `B=40`, joint-chi-square confidence selection, terminal
optimizer, terminal seed coordinate, GP, acquisition, or inference streams.

### NPE and SNPE

The first approved SBI decision is
`posterior_sample_map_physical_v1`. For the target-conditioned learned direct
posterior and configured final posterior sample population
`S = (theta_0, ..., theta_{K-1})`, it returns

\[
\hat\theta
=
\underset{\theta_k\in S}{\arg\max}\;
\log \hat p(\theta_k\mid y^\star).
\]

The score is the support-masked unnormalized learned posterior score evaluated
in physical parameter coordinates. The density is defined with respect to
Lebesgue measure in those physical coordinates. Estimator-internal
standardization remains owned by the learned density API. No transform
Jacobian is added to the optimization score: a Jacobian would change the
measure and therefore
the requested mode. Any theta-independent leakage normalization at fixed
target response is omitted from the argmax score.

The v1 policy uses exactly the already configured posterior sample population;
for the graph scientific protocol, `K=10000`. It performs no continuous
refinement and introduces no optimizer, restart, learning-rate, tolerance, or
boundary-refinement controls. It is recorded as `sample_map`, never as exact or
continuous MAP. Exact score ties resolve to the earliest candidate index. No
finite in-support score is a typed decision failure; posterior mean, median,
likelihood-only selection, simulator reranking, and silent fallback are
prohibited.

Each record persists at least the policy identifier, `sample_map` label,
candidate count, selected candidate index, selected score, score identity,
physical bounds, coordinate system, measure, tie rule, refinement=`none`, and
zero simulator-call accounting.

NPE and SNPE are the only families approved for the first implementation.
Cold-start target-specific NPE is the initial Paper-120 SBI comparator. SNPE
requires separate approval of its sequential allocation and qualification
before campaign inclusion.

For scientific identity compatibility, cold-start NPE's single round containing
the full declared budget is intrinsic to the method and is not serialized as a
separate method-configuration identity field. The allocation remains explicit
in execution records. SNPE's declared multi-round allocation is a scientific
choice and remains part of its method-configuration identity.

Accepted profile documents that pin derived scientific scalars are
authoritative after their formula, dimension, provenance, and tolerance checks
pass. A platform-local recomputation validates those values but must not replace
them in the resolved profile, canonical identity, or downstream seed family.

NLE, SNLE, NRE, and SNRE remain deferred to optional Prompt 20F and are not
ordinary paper MAP policies. Any future NLE/SNLE posterior MAP must maximize

\[
\log \hat p(y^\star\mid\theta)+\log p(\theta),
\]

and any future NRE/SNRE posterior MAP must maximize

\[
\log \hat r(y^\star,\theta)+\log p(\theta).
\]

Likelihood or ratio alone is never called posterior MAP. Activating any of
these deferred families requires a new approved numerical policy.

## Paired held-out assessment identity

The assessment policy is `paired_held_out_assessment_v1`. Its family identity
contains the protocol version, method-independent target-record identity,
simulator-definition and configuration identity, declared assessment fidelity,
and declared assessment campaign seed. It contains no method,
recommendation-policy, inference, or execution-selection identity.

For assessment replicate `a`, derive

```text
SeedCoordinate(
    "paired_held_out_assessment",
    (target_record_identity, a)
)
```

The initial protocol fixes `a=0`. For a fixed target and replicate, every
method receives the same assessment scientific-run identity and evaluation
seed. Coordinates differ across targets and replicates. Method inference
families remain method-addressed and independent; paired CRN applies only to
the fresh final assessment.

For the graph simulator, the shared assessment evaluation seed is the root of
three named randomness families: quenched disorder, initial state, and thermal
noise. Every method at a fixed target therefore receives the same underlying
disorder variates, initial phases, and thermal standard-normal trajectory.
Theta-dependent transformations of those shared variates remain method-
specific. New unified-runner graph records persist the complete assessment
seed manifest plus namespace-level coordinate identities and manifest hashes;
consolidation rejects mismatched assessment roots or namespace identities.

The mandatory causal order is:

```text
frozen target
    -> independent method-specific inference
    -> method-appropriate zero-call decision
    -> all recommendations or failures frozen and persisted
    -> method-independent paired assessment context constructed
    -> one fresh assessment call per valid recommendation
    -> metrics
```

Assessment randomness and responses cannot influence training, acquisition,
posterior fitting, terminal optimization, MAP selection, fallback, or
reranking. A failed method remains failed; valid recommendations are not
silently changed to manufacture a complete pair.

## Explicit physical metrics

For physical response dimension `M`, v3 records use only the following
authoritative names:

```text
response_error_l1_sum
response_error_mae
response_error_l2
parameter_error_l2
```

\[
\mathrm{response\_error\_l1\_sum}
=\sum_{j=1}^{M}|y_j^{\mathrm{assess}}-y_j^\star|,
\]

\[
\mathrm{response\_error\_mae}
=\frac{1}{M}\mathrm{response\_error\_l1\_sum},
\]

\[
\mathrm{response\_error\_l2}
=\lVert y^{\mathrm{assess}}-y^\star\rVert_2,
\]

and, when hidden truth is available,

\[
\mathrm{parameter\_error\_l2}
=\lVert\hat\theta-\theta^\star\rVert_2.
\]

Response metrics use raw physical response units. Parameter L2 uses raw
physical parameter coordinates and is an identifiability diagnostic only; it
cannot select or rerank a recommendation. Metric calculation reuses the one
assessment response and consumes no additional simulator call.

New v3 records do not write ambiguous legacy metric names. Backward-compatible
readers continue to interpret v2 `target_error_l1` as its immutable L1 vector
sum, `target_error_l2` as physical response L2, and `design_error_l2` as raw
physical parameter L2.

## Four-category simulator-call ledger

Each target records these categories separately:

| Category | MUTACO | NPE/SNPE | Budget treatment |
|---|---:|---:|---|
| target construction or reuse | one shared record | one shared record | outside method budget |
| method-specific inference or training | declared `B` | declared `B` | inside inference budget |
| terminal recommendation | 0 | 0 | model-side computation |
| paired assessment and metric evaluation | 1 | 1 | outside inference budget |

Sharing assessment randomness does not remove a call: each valid
recommendation is evaluated at its own theta. Target construction is charged
once when newly executed and zero when an accepted frozen target record is
reused. Physical attempts, retries, and failures are retained separately and
never hidden inside nominal scientific counts.

## Qualification boundary

Implementation prompts use toy or synthetic tests first. The first real graph
qualification is intended to reuse frozen Paper-100 target 0 under new v3
identities, a separately presented exact command, and a strict call cap. It is
engineering evidence and cannot replace any historical Paper-100 record.

This contract authorizes no implementation, qualification, target generation,
or campaign.

## Locked B100 qualification profile

The approved B100 engineering qualification reuses frozen Paper-120 target
positions 0, 49, and 99. MUTACO allocates 16 initial and 84 sequential calls;
target-specific NPE uses one 100-call round; target-specific SNPE uses rounds
`[34, 33, 33]`. Each method then receives one paired held-out assessment call
using the existing assessment campaign and its shared disorder and thermal-noise
coordinates. The exact qualification cap is therefore 900 inference calls plus
9 assessments, or 909 physical calls. Passing this gate is operational evidence,
not a performance claim. Its completed records may be retained in a later
Paper-120 B100 comparison if their locked identities remain exact.

## Prepared B40 response-standardization ablation

The prospective response-standardization ablation compares two labelled arms,
`mutaco_standardized` and `mutaco_unstandardized`, through the same MUTACO
adapter and Paper-v3 runner. The former retains the historical
initial-design `ddof=1` fitted transform; the latter selects the explicit
`identity_physical_v1` transform. Both retain B40 allocation `16 + 24`, the
joint-chi-square confidence profile, GP and recovery settings, continuous
posterior-mean physical-MAE terminal policy, frozen Paper-120 targets, and
one held-out assessment per target. Inference remains method-addressed; only
the final assessment uses paired common random numbers.

The completed population comparison must therefore be interpreted as using
distributionally matched but arm-specific learning streams. The two arm
identities generate different initial-design realizations as well as distinct
subsequent evaluation streams. Its paired held-out endpoint is valid supporting
evidence, but the experiment is not a fixed-trajectory or shared-initial-data
causal ablation of standardization alone.

The assessment comparison configuration remains
`paper-comparison-config-v1-a5da263af0ad0975602297f8f17e486f56ce6950aae3256fa40cbd5f17253d19`,
campaign seed `92020402`, and replicate zero. Therefore every target reuses
the exact B40/B100 assessment coordinate and the same quenched-disorder,
initial-state, and thermal-noise manifests. The three-target engineering
qualification selects cohort positions `0`, `49`, and `99` and has a strict
cap of 240 inference calls plus six assessments, or 246 physical calls. This
preparation authorizes neither the qualification nor the full 9,840-call
campaign.
