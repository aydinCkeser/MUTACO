# Paper-120 SBI-zoo extension contract

**Classification:** normative candidate; physical qualification and Paper-120
P1/P2/P3 execution passed, and the deterministic comparison analysis is complete

**Authority subject:** the final planned simulator-bearing paper experiment:
single-round NLE and single-round NRE at independent B40 and B100 budgets

## Scientific purpose and scope

The extension adds exactly two SBI estimators to the repaired Paper-120
MUTACO--NPE--SNPE comparison:

- single-round neural likelihood estimation (NLE); and
- single-round neural ratio estimation (NRE).

The purpose is bounded breadth across posterior-, likelihood-, and
likelihood-ratio estimation. It is not a general SBI leaderboard. SNLE and
SNRE are excluded from the main-paper plan because their proposal-dependent
rounds add sequential-policy and numerical complexity without closing a
necessary methodological gap after SNPE.

Candidate names are determined by the executed simulation design, not by the
installed `sbi` class name. A method trained once on one immutable
prior-predictive dataset is NLE or NRE even if the installed API exposes that
class through an `SNLE` or `SNRE` alias. The sequential label requires at least
two simulation rounds and a later proposal distribution that depends on an
earlier fitted posterior.

## Authoritative comparison backbone

The extension uses the canonical ordered Paper-120 v3 cohort, frozen target
responses, 20-component physical response ordering, canonical physical
simulator and stochastic law, and component-mean physical held-out MAE. B40
and B100 are independent campaigns, not nested trajectories or datasets.

Historical MUTACO, NPE, and SNPE searches, recommendations, and repaired
held-out outcomes remain immutable and are not rerun. Learning realizations
are not claimed to be shared between the historical methods and the new
methods. Their comparability is by cohort, per-target simulator-call budget,
physical stochastic law and configuration, response definition, and endpoint.

## Training-data contract

The retained NPE artifacts do not contain exact training theta/response pairs.
They retain seeds, context identities, diagnostics, recommendations, and call
accounting, but neither the local 120-target B40 raw records nor the B100
compact evidence provides a reusable immutable pair dataset. Reconstructing
responses from seeds would be a new simulator campaign and must not be called
historical dataset reuse.

For each target and budget, the extension therefore generates one new
method-independent prior-predictive dataset:

- B40: exactly 40 simulator calls in one prior-predictive round;
- B100: exactly 100 simulator calls in a separate prior-predictive round.

The B40 dataset is not a prefix of B100. Targets and budgets have distinct,
coordinate-addressed dataset and simulator-context identities. Each row
persists canonical float32 physical theta and response bytes, response order,
context identity, and integrity hashes. Once complete, the dataset is frozen
before estimator training. NLE and NRE consume the identical ordered bytes for
that target--budget. Their training and posterior-sampling randomness remains
method-addressed.

This design controls simulation evidence for NLE versus NRE. Comparisons with
historical NPE, SNPE, and MUTACO remain matched-law, equal-budget end-to-end
comparisons; they are not represented as shared-learning-realization
comparisons.

The authoritative prior is the historical independent physical BoxUniform law
in coordinate order `(weight_mean, pairwise_coupling_mean, bias_scale_mean,
pairwise_coupling_CV)`, with lower `(0.5,0.5,0.5,0.0)` and upper
`(2.0,2.0,2.0,1.0)`. Dataset theta is generated directly on CPU as float32 by
a coordinate-seeded `torch.Generator` and
`lower + torch.rand(...) * (upper-lower)`. The sampling implementation is
half-open at the upper endpoint while density and validation support use the
closed box. Invalid or nonfinite samples fail without rejection or replacement.
Budget is a seed coordinate, so B40 and B100 are independent and not nested.

The artifact schema `mutaco-sbi-zoo-shared-dataset-v1` stores `[B,4]` theta and
`[B,20]` responses as little-endian, C-contiguous float32 bytes plus canonical
JSON metadata. NaN and infinity are forbidden. Its SHA-256 covers
length-prefixed metadata, theta, and response bytes in that order. A complete
sibling directory is atomically renamed into its append-only destination, and
reload rehashes every file and the composite payload before exposing tensors.
NLE and NRE consumer receipts must contain the identical dataset identity and
content hash.

Each scientific row is first published as an append-only
`mutaco-sbi-zoo-slot-outcome-v1` artifact. Canonical metadata binds its
little-endian float32 response bytes to the declaration, call ledger, target,
budget, scientific slot, ordered row, theta hash, stochastic context, response
ordering, simulator, and simulator configuration. Success is permitted only
after atomic publication, reload, complete SHA-256 verification, and equality
with the write-ahead intent. Dataset assembly accepts exactly one validated
ledger-success artifact for every row in frozen row-index order; filesystem or
event-arrival order is never scientific ordering.

An invocation exception is `failed`. Any exception after an invocation returns
is `attempted_outcome_unknown`, including canonicalization, persistence,
reload, or success-event failure. Attempted-without-terminal replays as the
same unknown state. If an outcome is durable but its success event is not, the
orphan is retained for audit and is never deleted, overwritten, reconciled,
reused, or assembled. Failed and either explicit or implicit unknown slots are
non-retryable and consume budget.

## Estimator and training contract

The planned existing-project settings are CPU float32, MAF for NLE, MLP with
`num_atoms=2` for NRE, learning rate `5e-4`, validation fraction `0.2`, batch
size `min(64, B)`, early-stopping patience `20`, maximum `200` epochs, and no
fallback. Qualification is pass/fail, not a hyperparameter search. A training,
sampling, scoring, serialization, or numerical failure is retained as a typed
failure; there is no automatic scientific retry.

Pinned `sbi==0.26.1` defaults are resolved rather than leaving `maf` and `mlp`
as incomplete identities. NLE uses five MAF transforms, width 50, two
non-residual tanh blocks per autoregressive transform, fixed masks, zero
dropout, no batch normalization, identity response embedding, independent
theta/response standardization, random permutations between transforms, and a
standard-normal base. NRE uses a two-hidden-layer width-50 MLP, with LayerNorm
then ReLU after each hidden linear layer, identity theta/response embeddings,
independent standardization, one output logit, and `num_atoms=2`.

Training uses Adam with resolved Torch defaults, no scheduler, gradient clipping
at norm 5, a seeded `torch.randperm` 80/20 split, seeded
`SubsetRandomSampler` loaders, `drop_last=true`, and zero loader workers. The
machine contract serializes all effective loader and optimizer fields.
`sbi 0.26.1` loops while `epoch <= max_num_epochs`; the implementation passes
`199` to enforce the scientific maximum of 200 completed epochs. This is an
API-semantics translation, not tuning.

The MCMC policy is resolved before physical qualification.
`slice_np_vectorized` uses 20 chains, 100 warmup steps, thinning 1, one worker,
spawn context, and `init_strategy=proposal`. Under pinned `sbi`, initialization
makes one sequential draw from the physical BoxUniform for each chain in chain
order; no candidate pool is involved. One named method--target--budget seed
controls initialization and sampling. Its full 63-bit value seeds Torch and its
value modulo `2^32` seeds the NumPy global RNG used internally by the pinned
vectorized slice sampler. Both ambient RNG states are restored afterward.
Invalid initial state is a typed failure.

Exactly 10,000 retained samples total are requested. Pinned `sbi` obtains 500
retained samples per chain, stores chains-by-samples-by-parameters, flattens in
C-order (chain-major), and returns physical theta after inverse transformation.
Candidate index is `chain_index*500 + within_chain_index`. Any chain failure
fails the aggregate recommendation; partial chains, insufficient samples,
retry, and fallback are forbidden. Qualification tests this frozen policy and
may not choose or tune it.

## Authoritative sample-MAP decision

`posterior_sample_map_physical_v1` is scientific intent, not an incidental
implementation choice. Posterior mean, posterior median, simulator reranking,
silent fallback, and continuous MAP refinement are prohibited.

For the configured ordered sample population
`S=(theta_0,...,theta_9999)`, retain finite candidates inside the closed
physical parameter box and select the earliest candidate attaining the
largest finite posterior-density score. Exact score ties use earliest sample
order. Absence of any valid candidate or finite score is a typed
recommendation failure.

NLE scores in physical theta coordinates

\[
s_{\mathrm{NLE}}(\theta)
=\log \hat p(y^\star\mid\theta)+\log p(\theta),
\]

and NRE scores

\[
s_{\mathrm{NRE}}(\theta)
=\log \hat r(y^\star,\theta)+\log p(\theta).
\]

The score is the unnormalized physical-coordinate posterior potential. An
MCMC-internal unconstrained-coordinate Jacobian is not added to this final
sample-MAP score. Sampling mechanism, sample count, randomness, support mask,
bounds, sample order, score identity, tie rule, and failure behavior are part
of every recommendation identity.

The closed NLE/NRE `score_api` is
`sbi_mcmc_posterior_potential_physical_no_transform_jacobian_v1`. NLE requires
`score_identity=sbi-0.26.1-likelihood_plus_prior-physical-theta-v1`; NRE
requires `score_identity=sbi-0.26.1-ratio_plus_prior-physical-theta-v1`.
`method_configuration_identity` remains separate estimator and training
provenance. Historical NPE/SNPE retain their unchanged direct-posterior
`score_api=sbi_direct_posterior_log_prob_physical_v1` and posterior-derived
score identity. A method/API/identity/scoring-path mismatch fails with
`SbiZooScoringContractError`.

## Qualification gate

Zero-call tests first cover package/API identities, deterministic shared-data
serialization, call ledgers, the two posterior-score formulas, physical
support masking, earliest exact ties, 10,000-candidate ordering, typed failure,
and mock assessment persistence. Synthetic SBI tests may train only synthetic
models and construct no scientific simulator.

The engineering tests install a fail-closed tripwire around canonical physical
graph-simulator construction and evaluation. This is additional to call
counters: any attempted construction or invocation fails immediately.
Synthetic estimator fits and tiny MCMC API smoke runs are engineering checks,
not physical qualification or Paper-120 execution.

The smallest physical qualification uses one accepted off-cohort Paper-100
target. It generates one independent 40-row dataset and one independent
100-row dataset, then trains both methods and freezes their recommendations.
The qualification cap is exactly 140 simulator calls and zero held-out calls.
It must not derive, load, inspect, or evaluate repaired Paper-120 held-out
contexts or outcomes. Failure of either method at either budget stops
progression; changing a method or control requires a separately authorized
declaration.

The canonical execution package resolves that target mechanically as historical
Paper-100 target position 0,
`target-record-v1-0344e748d87e1d5e323cb53eb67bc7c0995c229cbaa474ccf44a0c3c228817d8`.
Its archived source SHA-256 is
`7c1f7ead8b1b8919105de6d41f0e9b460ce7a56d9552f49651a4e661231b2793`.
The self-contained execution declaration retains that identity and archive
location without modifying historical evidence. The record is outside
Paper-120, is qualification-only, and is ineligible for population analysis.

`experiments.graph_inversion.sbi_zoo_qualification` owns declaration,
dataset, method, collection, and scheduler-record commands. Dataset theta and
row contexts call the frozen randomness APIs directly; the execution layer
does not derive a second randomness policy. The tracked TRUBA surface is
`sbi_zoo_qualification_v1.env` plus
`submit_sbi_zoo_qualification.sh` and the no-requeue, exclusive one-node batch
template. One job executes B40, independent B100, and then the four method
components sequentially, stopping on the first failure. Physical submission
remains separately gated.

Provenance distinguishes
`scientific_contract_source=d57313a2e9affdb1e6db30a045a8b1a6ef27da33`
from the later `qualification_execution_source`: the former identifies the
frozen scientific implementation and contract from which the execution
package was derived; the latter will identify the complete exact source
actually materialized and executed. Declarations, logs, ledgers, models,
recommendations, receipts, and collections live outside the clean source
checkout in append-only roots.

The Q0--Q2 physical qualification completed on TRUBA as job `6261857` from
`qualification_execution_source=bffa31ec729dfb90fa294c8c02f47634f0a0c7be`
(tree `0d6cd46c412144bf446aac29f026c4a3301fee52`). All 40 B40 and
100 independent B100 calls succeeded, with no failures, unknown outcomes, or
retries. All four NLE/NRE budget recommendations passed the 10,000-candidate
contract. The append-only qualification evidence root is
`/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/sbi_zoo_qualification/q0q2-v1-bffa31e`.

## Paper-120 P1 execution package

P1 is dataset generation only: 120 ordered targets each produce and immediately
freeze B40, followed by an independently addressed B100. The complete campaign
declares 4,800 B40 slots, 12,000 B100 slots, and 240 dataset artifacts. NLE/NRE
training, recommendation generation, MCMC, and held-out evaluation are absent.

`experiments.graph_inversion.sbi_zoo_p1` owns declaration, exact cohort-to-worker
mapping, slot enumeration, per-target worker progression, dataset freeze,
restart inspection, population collection, and the machine-checkable P2 gate.
The TRUBA surface is `sbi_zoo_p1_v1.env`, `submit_sbi_zoo_p1.sh`, and separate
worker/collector batch templates. The Candidate A worker geometry is `barbun`,
one logical worker per allocation, 20 runtime threads, 20 requested Slurm CPUs,
64 GB, 30 minutes, physical array `0-59%20`, non-exclusive and no-requeue.
Node colocation is scheduler-controlled rather than part of the scientific or
execution contract. Successful historical Barbun jobs support an expected
`AllocCPUS=20`; the authoritative observed value must be taken from `sacct`
after execution. The 64 GB request is an empirically accepted Barbun submission-policy
representation, not an estimate of application demand; qualification peak RSS
was about 599 MiB. The dependent collector uses 20 CPUs, 64 GB, 15 minutes,
non-exclusive and no-requeue. Physical task `i` runs logical targets `i` and `i+60` sequentially
in separate Python processes. The second target still runs after a nonzero
first target; the physical task reports nonzero after both terminate if either
failed. Thus no more than 20 logical target workers run simultaneously.

The original `0-119%20` submission was rejected before job creation because
TRUBA currently has `MaxArraySize=100`. It created no worker, collector, or
scientific call. The 60-by-2 repair follows the completed scalar-primary B40
and B100 Paper-120 precedent and also keeps the array plus collector below the
observed association submitted-job limit. The rejected declaration is not
reused: a repaired exact source requires a fresh lifecycle receipt and freshly
bound declaration identities before any physical call.

A later `20 CPU / 4 GB / non-exclusive` P1 submission was also rejected before
job creation by Barbun's accounting/QOS policy and consumed zero calls. Its
preflight and declaration remain historical zero-call provenance and cannot be
reused. The resource repair changed scheduler representation only and therefore
required a new exact source, lifecycle receipt, append-only root, and declaration
before separate P1 execution authorization.

An earlier worker submission, job 6263056, was also cancelled with zero
allocation and zero scientific calls. The subsequent
`fd8a9199b96ad1f84c43badffbee5308c8e50306` execution package
submitted worker array 6263281 and collector 6263282, but both were cancelled
with no allocation, started worker, scientific call, dataset, or held-out access.
The immutable `p1-preflight-v3-fd8a919` root, lifecycle receipt, and declaration
are zero-call historical evidence and cannot be edited, regenerated, republished,
or reused. Candidate A changed only worker exclusivity. The final execution
therefore required a new commit and tree, append-only exact-source root,
lifecycle receipt, and P1 declaration identity and hash.

Restart may continue only a new or intent-only slot. Durable success is skipped;
failed and explicit or implicit `attempted_outcome_unknown` states are permanent
hard stops. The P2 gate required the read-only collector to validate all 240
datasets and all 16,800 successful calls.

The final P1 execution used source
`55ce098f53c6e14749dcbc1aa8fa54b2de902026`, worker array `6263793`, and
collector `6263794`. It passed with 120 B40 datasets, 120 independent B100
datasets, 16,800 successful calls, and no failed, unknown, retry, or replacement
calls. The frozen collection identity is
`mutaco-paper120-sbi-zoo-p1-collection-v1-07baf458f14d4332db73c136736822113934e208cdc419c1b4842267397dbd7e`.

## Paper-120 P2 and P3 completion

P2 used source `ea582fe46b0b1a3e819d0eb66931805215cfe5e5`, worker array
`6264281`, and collector `6264282`. It consumed the 240 frozen P1 datasets
read-only and produced 480 frozen NLE/NRE recommendations with zero simulator
calls. Its collection identity is
`mutaco-paper120-sbi-zoo-p2-collection-v1-7bc671817b515d988b13b8e842e3f28c34688744c8949db4d4a0324f0981590a`.

P3 used source `593b70afb677dd21766f93e14113debe5dac5ba2`, worker array
`6265116`, and collector `6265117`. All 480 frozen recommendations received
exactly one repaired-family held-out evaluation; all 480 calls succeeded with
no failure, unknown outcome, retry, or replacement. Its collection identity is
`mutaco-paper120-sbi-zoo-p3-collection-v1-d35fd77ef3a3e03e00f784f682315dd7314288ffa13cd48d418846bfdfe36589`.

The canonical deterministic comparison analysis is complete in
`benchmarks/paper/sbi_zoo_comparison_v1/manifest.json`. No active TRUBA job or
further simulator-bearing main-paper experiment remains.

## Test-set firewall and inferential status

The repaired held-out outcomes are already known for historical MUTACO, NPE,
and SNPE. They are forbidden for method choice, architecture or hyperparameter
selection, qualification, debugging, stopping, fallback, or decisions to keep
or alter a method.

Before assessment, freeze implementation/source identity, all settings and
randomness identities, the 240 immutable training datasets, and all 480 B40
and B100 recommendation records and hashes. The assessment is a
**prospectively frozen SBI-zoo extension using the existing repaired-SBI
assessment family**. NLE and NRE each receive one first exposure to that
family. The complete five-method family is not described as an entirely
previously unseen test set.

## Held-out assessment

After recommendation freeze, B40 uses the exact repaired-SBI target-context
map rooted at `92021540` and B100 uses the map rooted at `92021600`. Each new
recommendation receives one fresh simulator invocation using a fresh simulator
instance and the exact pre-existing target-specific held-out context. This is
not an independent stochastic realization: the common context intentionally
provides paired common random numbers with archived MUTACO, NPE, and SNPE.

Every invocation has write-ahead intent and attempted accounting followed by
success, explicit failure, or `attempted_outcome_unknown`. No slot can be
retried under the declaration. Existing method outcomes are reused unchanged;
only NLE and NRE recommendations receive new held-out calls.

## Analysis

B40 and B100 are analyzed separately. Report mean and median physical MAE,
target-wise NLE-minus-MUTACO and NRE-minus-MUTACO differences, established
target-level paired percentile-bootstrap 95% intervals, and useful
win/tie/loss counts. These are prospectively prespecified secondary paired
comparisons with ordinary 95% intervals. No familywise claim that MUTACO beats
all SBI methods is authorized, and budgets are neither pooled nor treated as
nested.

## Exact call budget

Paper-120 shared training datasets require `120*40=4,800` B40 and
`120*100=12,000` B100 calls. Two new methods at two budgets require 480 new
held-out calls. The Paper-120 cap is therefore 17,280. Adding the 140-call
off-cohort qualification gives a maximum new simulator-call cap of 17,420.
Sharing the NLE/NRE dataset saves 16,800 calls relative to separate training
streams. No historical NPE-dataset-reuse saving is claimed.

## Authorization and exclusions

The scientific implementation, qualification, Paper-120 P1/P2/P3 execution,
and deterministic comparison analysis are complete. This completed state does
not authorize reruns, additional simulator calls, new SBI fits or
recommendations, held-out access, or new TRUBA submission. SNLE, SNRE,
posterior-mean decisions, continuous MAP,
historical dataset reconstruction, new held-out families, reruns of existing
methods, B40/B100 pooling, and familywise superiority claims are excluded.
