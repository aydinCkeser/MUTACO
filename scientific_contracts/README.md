# Scientific contracts

This directory explains what MUTACO computes, what its experiments compare, and which conclusions the recorded evidence supports. A scientific contract makes choices such as response ordering, the fitting objective, the final recommendation rule and the assessment budget explicit. These choices must remain identifiable when files are renamed or experiments are reproduced.

Start with the [mathematical contract](mutaco_d_contract/README.md) to understand the algorithm, the [paper scope](paper_scope/README.md) to understand the claims, and the [scientific registry](scientific_registry.toml) to find each contract's recorded status. The [core guide](../src/mutaco/README.md) connects the scientific stages to their implementations. The [migration map](../docs/decisions/repository_migration_map.md) records changed addresses and any historical paths that must remain available.

## 1. What belongs here

The contracts describe mathematical definitions, scientific interfaces, experiment designs, method comparisons and interpretation rules. The directory also retains the policy support tables and reference material needed to state those contracts precisely.

| Material | Owner and reason |
| --- | --- |
| Mathematical and comparison rules | This directory; a reader should be able to identify the intended quantities before inspecting a runner. |
| Scientific contract status | `scientific_registry.toml`; preserve the existing classifications and version identities when relocating it. |
| Sequential inference implementations | [`src/mutaco/`](../src/mutaco/); the installed package owns the reusable algorithm. |
| Physical forward model | [`simulators/graph-simulator/`](../simulators/graph-simulator/); it defines how graph parameters produce physical responses. |
| Experiment programs and configurations | [`experiments/`](../experiments/); they select targets, budgets, policies and study procedures. |
| Results still used in current research | [`results/`](../results/), including completed results from earlier dates; their age does not make them obsolete. |
| Superseded or unused historical material | [`scientific_archive/`](../scientific_archive/), with its original status, provenance and relevant context preserved. |
| Saved-evidence verification | [`reproducibility/`](../reproducibility/); checking stored evidence is a separate operation from generating new scientific observations. |
| Manuscript and explanatory documentation | [`docs/paper/`](../docs/paper/) and [`docs/`](../docs/); these explain and present the scientific work. |

Some original paths are part of a source-hash, archive-member or historical loader contract. The migration map must distinguish a new canonical owner from an address retained for compatibility. A clean directory listing alone does not establish reproducibility.

## 2. Read status before using a contract

Four questions must be answered separately: is the definition authoritative for a particular protocol, is its implementation available, has it been executed under the required conditions, and is its evidence accepted for the current paper?

| Recorded role | How to interpret it |
| --- | --- |
| **Normative** | Defines the required meaning or procedure within its stated version and scope. It does not make every implementation or result automatically valid. |
| **Candidate / prospective** | Describes a proposed or bounded scientific program. Check its own qualification records and acceptance conditions before making a broader claim. |
| **Historical** | Records an earlier policy, implementation or investigation. Preserve the context needed to interpret its results; do not relabel them as outputs of the current policy. |
| **Diagnostic / supporting** | Answers a narrower question, such as numerical parity or sensitivity. Its conclusion may support the paper without measuring the main comparison endpoint. |

These descriptions help read the existing registry; they do not replace its machine-readable vocabulary. Combined labels such as “normative candidate” must be retained with their scope. Execution status and scientific authority can differ: a candidate contract may have completed executions, while a normative design may have unexecuted branches.

Use the registry, the relevant versioned contract, the current paper-scope document and the corresponding evidence manifest together. If an older header and a later record disagree, record that discrepancy for review instead of silently promoting, retiring or rewriting a scientific claim. Historical execution flags are records of their original milestones.

## 3. The quantities the algorithm works with

MUTACO seeks parameters whose forward response matches a supplied target. Let `P` be the number of parameters, `M` the number of ordered response components, `N` the number of committed observations and `H` the number of query points.

| Quantity | Shape | Meaning |
| --- | --- | --- |
| Parameter vector $\theta$ | `[P]` | One point in the declared bounded physical parameter domain $\Theta$. |
| Ordered designs | `[N, P]` | Evaluated parameter vectors in their recorded observation order. |
| Physical response $y$ and target $y^\star$ | `[M]` | Components in the order declared by the response specification. |
| Training responses | `[N, M]` | Responses paired with the corresponding ordered designs. |
| Latent posterior mean at queries | `[H, M]` | Predicted standardized response of the fitted surrogate. |
| Per-query task covariance | `[H, M, M]` | Latent covariances of the response components at each query point. |
| Full joint latent covariance | `[H*M, H*M]` | Covariances across both points and components; point-major index `i*M + m` identifies component `m` at point `i`. |

Names, bounds, units and ordering are scientific data. A reorder can change the target-matching problem even when tensor sizes still agree. Physical coordinates mean the coordinates declared by the simulator; a unit label does not apply an automatic unit conversion. Dtype, device and contiguity requirements belong to the corresponding numerical interfaces.

The [simulator contract](simulator_protocol/README.md) describes the minimum `[P] -> [M]` interface, declared batched capabilities, context metadata and retry behavior. A batched `[H, P] -> [H, M]` implementation must preserve what counts as one parameter evaluation. Logical committed observations, physical attempts and held-out assessments are distinct accounting quantities.

### Frozen response coordinates

The initial observations establish a componentwise center $c_m$ and effective scale $s_m$. The reference standardization policy uses the initial sample standard deviation with `ddof=1`, bounded below by the configured scale floor:

$$
\widetilde y_m=\frac{y_m-c_m}{s_m},\qquad
s_m=\max\{\operatorname{sd}_{\mathrm{ddof}=1}(y_{1,m},\ldots,y_{N_0,m}),s_{\min}\}.
$$

Here $N_0$ is the initial-design size and the standard deviation is taken over its observations for each fixed component $m$. The same frozen transform is used for later observations, the target and the inverse transformation of predictions. Its identity binds the statistics to the initial observations and component order. The scale floor handles nearly constant components; it is not additional observation noise.

The no-standardization policy has its own ablation identity. Recomputing the transform after each new observation or replacing the recorded statistics with numerically similar values would change the scientific procedure.

## 4. Follow one MUTACO run

### Step 1 — initialize and freeze the response transform

Generate the configured initial design within the parameter bounds, evaluate its ordered rows, and freeze the response transform. The paper profile uses a scrambled Sobol design with 16 initial observations; these values are profile choices rather than universal requirements of every supported call. Named seed coordinates identify the random streams used for design, fitting, proposal generation and simulation.

### Step 2 — fit a correlated response surrogate

The fitter receives physical parameter inputs and standardized responses. It normalizes parameters to the model domain and expands each observation into task-indexed rows before fitting the correlated exact GP. The model provides latent mean and covariance information for the acquisition rule, with a separate interface for predictions that include observation noise. Per-point covariance blocks cannot substitute for the full joint covariance required by a joint-sampling method.

The optimizer minimizes the negative exact marginal log likelihood **including the registered log-prior terms**. This is MAP-type fitting. The qualified B40/B100 profile allows 12 fitting iterations. A different iteration limit in a generic configuration describes a different profile. The manuscript's marginal-likelihood wording is an identified author-review item; the reorganization must preserve the implemented objective and keep that wording issue visible. A successful fit record establishes the checks actually performed, not an unconditional statement that the numerical optimizer converged.

### Step 3 — choose and evaluate another parameter vector

In standardized response coordinates, let $r(\theta)=\widetilde\mu(\theta)-\widetilde y^\star$ and let $\Sigma(\theta)$ be the latent task covariance. For the operator-norm radius policy,

$$
\rho^2(\theta)=\beta^2\lambda_{\max}\!\left(\Sigma(\theta)\right),\qquad
\Delta(\theta)=\frac{\rho^2(\theta)-\lVert r(\theta)\rVert_2^2}{\rho^2(\theta)}
\quad\text{when }\rho^2>0.
$$

Compatibility uses $\lVert r\rVert_2^2\leq\rho^2$. The barrier objective is minimized in its strict interior, $\Delta>\tau_{\mathrm{interior}}$:

$$
J(\theta)=\lVert r(\theta)\rVert_2-\alpha\log\Delta(\theta).
$$

The paper profile’s exploration objective maximizes the largest latent covariance eigenvalue. The proposal code specifies screening, refinement, branch selection, fallbacks and exact-duplicate handling; those details are part of the behavior to preserve. At zero radius, the implementation treats exact target agreement as compatible but does not place it inside the positive barrier domain.

The paper's beta policy derives a nominal response-space radius from a chi-square quantile with `M` degrees of freedom. `M` is the response dimension, not the number of parameters or observations. This calibration does not itself assert a time-uniform or frequentist coverage theorem. Fixed beta and trace-based alternatives retain their separate policy identities.

After an actual evaluation succeeds and its response is committed, the runner refits as specified. Retries, failed attempts and unsuccessful proposals retain their recorded outcomes and accounting.

### Step 4 — make the final recommendation

The paper policy minimizes the physical MAE of the final surrogate's posterior-mean response over the continuous parameter domain. Here $B$ denotes the final committed-observation budget:

$$
\widehat\theta\approx\arg\min_{\theta\in\Theta}
\frac{1}{M}\sum_{m=1}^{M}\left|c_m+s_m\widetilde\mu_{B,m}(\theta)-y_m^\star\right|.
$$

This is an approximate optimization of a specified score. It is neither the posterior expectation of absolute error nor a fresh simulator evaluation. The sequential runner's final refit already includes the final committed observation. The final recommendation uses that model without adding an otherwise redundant fit.

The generic API default is an evaluated-point standardized L1 policy. It must not be confused with the paper's continuous physical-MAE profile. Physical posterior-mean squared L2, expected standardized squared L2 and best observed physical MAE also represent different quantities, domains or information sources.

## 5. Forward models, comparators and assessment

The graph forward model evolves coupled oscillators and returns ordered physical observables. The analytic benchmark models are controlled forward maps used to study target matching: the handwritten map supplies explicit nonlinear channels, while the GP-landscape model uses a frozen random-feature construction. That landscape is a forward problem; the GP trained inside MUTACO is the surrogate used to solve an inverse problem.

The owners are [`simulators/graph-simulator/`](../simulators/graph-simulator/) for graph physics and [`experiments/analytic_target_matching/forward_models/`](../experiments/analytic_target_matching/forward_models/) for analytic forward maps. [`experiments/graph_forward/`](../experiments/graph_forward/) exercises and characterizes physics; [`experiments/graph_inversion/`](../experiments/graph_inversion/) searches for parameters. [`experiments/benchmarking/`](../experiments/benchmarking/) provides shared comparison machinery, and [`experiments/paper/`](../experiments/paper/) owns paper-specific protocols and studies.

| Method or quantity | Distinction to preserve |
| --- | --- |
| MUTACO | Uses correlated response predictions, compatibility geometry and its configured acquisition/terminal policies. |
| Scalar qLogNEI | Fits a scalar objective based on negative physical MAE; its final scalar posterior-mean decision differs from MUTACO's multicomponent posterior-mean MAE. |
| Retained composite Monte Carlo NEI | Samples correlated joint latent responses to estimate physical-MAE improvement. It is a separate supported implementation and historical program, not another name for scalar qLogNEI. |
| Sobol comparator | Uses its declared design and observed-response final-decision rule. A sampled observation and a surrogate prediction are different information sources. |
| SBI comparisons | Preserve the declared inference method and phase boundaries: P1 creates simulation data, P2 trains on stored data and freezes decisions, P3 performs held-out assessment. |

The [retained graph fixture capture script](../simulators/graph-simulator/audits/capture_retained_graph_fixture.py) is a historical producer. Reading the saved JSON fixture is the offline comparison path.

The [held-out assessment protocol](assessment_protocol/README.md) specifies a fresh forward assessment at each final recommendation, using the same declared fidelity across compared methods. Its response is not fed back into inference. Use the [comparison protocol](paper_comparison_protocol/README.md) and [cross-experiment semantics](cross_experiment_contract/README.md) to distinguish prediction error, held-out response error, parameter error and any other recorded estimand. A low value for one does not establish a low value for another.

## 6. Directory guide

Each subfolder describes a different question. The scientific registry and the versioned evidence specify the scope of each classification.

| Subfolder | What to read it for |
| --- | --- |
| [`acceptance/`](acceptance/README.md) | The boundaries of acceptance checks: packaging, strict interfaces, identities, seeds and analytic replay have specific coverage limits. |
| [`assessment_protocol/`](assessment_protocol/README.md) | Independent final response assessment, common fidelity and separation from the search budget. |
| [`cross_experiment_analysis/`](cross_experiment_analysis/README.md) | Normative deterministic analysis and B40/B100 synthesis, while each campaign retains ownership of its underlying evidence. |
| [`cross_experiment_contract/`](cross_experiment_contract/README.md) | Error vocabulary, evidence statuses, recommendation provenance and cohort semantics. `contract.json` is a versioned scientific record. |
| [`graph_oscillator/`](graph_oscillator/README.md) | Scientific and retained physical-model characterization. `reference_fixtures/retained_cpu_small_v1.json` is an immutable historical reference, not a newly selected model setting. |
| [`mutaco_d_contract/`](mutaco_d_contract/README.md) | The current normative mathematical contract, coordinate rules, GP boundaries, policy support and generated compatibility tables. |
| [`paper_comparison_protocol/`](paper_comparison_protocol/README.md) | Final-decision semantics, method-appropriate decision rules, protocol identities and records. |
| [`paper_scope/`](paper_scope/README.md) | Which evidence supports current paper claims and which remains supporting, diagnostic or historical. |
| [`primary_benchmark_protocol/`](primary_benchmark_protocol/README.md) | The frozen MUTACO–NEI–Sobol design and its declared cohort/budget geometry. Interpret its original composite-NEI program through the current registry and scope. |
| [`repaired_mutaco_sbi_salvage/`](repaired_mutaco_sbi_salvage/README.md) | The bounded historical repair design and corrected-terminal lineage; preserve its prospective/execution distinctions and source identities. |
| [`sbi_zoo_extension/`](sbi_zoo_extension/README.md) | The SBI extension contract and P1/P2/P3 boundaries. The inspected document combines a normative-candidate classification with completed execution and analysis records. |
| [`scalar_qlognei_primary_benchmark_protocol/`](scalar_qlognei_primary_benchmark_protocol/README.md) | The distinct scalar-primary comparison and estimand. Its candidate header and later execution evidence must be read with their recorded dates and scope. |
| [`simulator_protocol/`](simulator_protocol/README.md) | The forward-call interface, metadata, batching capability, retry and accounting contract. |
| [`terminal_domain_ablation/`](terminal_domain_ablation/README.md) | The bounded question of continuous versus evaluated-point terminal optimization. A zero-physics refit study still performs GP computation; held-out assessment is a separate stage. |

The [same-environment validation contract](../scientific_archive/validation_history/refactor_validation/README.md) still defines the exact replay acceptance gate. The [terminal-lineage investigation](../scientific_archive/validation_history/historical_terminal_lineage_forensics/README.md) describes the historical implementations used by the original runs. Evidence still needed for current accepted results belongs with, or remains explicitly linked from, the relevant result package; moving an old report does not retire valid results.

### Generated policy documentation

Within `mutaco_d_contract/`, the initial-design, likelihood, candidate-policy, mathematical-policy and posterior-numerics support documents describe distinct interfaces. `POLICY_SURFACE.json` and `POLICY_SURFACE.md` expose the recorded policy surface. `generate_policy_surface.py` is a documentation producer that imports the policy registry; this directory is therefore mostly contracts and reference material, but not entirely passive text.

Generated rows must remain derived from their owning registry. Do not edit IDs, support labels or table rows independently to make a migrated directory look consistent. Preserve current, compatibility and historical distinctions and verify the generator's paths before publishing regenerated documents.

## 7. From manuscript to implementation and evidence

The [MUTACO preprint v2](https://arxiv.org/abs/2610.04619v2) and [core README](../src/mutaco/README.md) should be read together. The manuscript uses `app:mutaco_algorithm` for the algorithm appendix. These are the callable names used to find each implementation.

| Scientific subject | Contract and implementation entry points |
| --- | --- |
| Initial design and response coordinates | `mutaco_d_contract/`; `generate_initial_design`, `FrozenResponseTransform`, `StandardizedTarget`. |
| Correlated GP and latent moments | `mutaco_d_contract/`; `fit_exact_multitask_gp`, `ExactMultiTaskGPSurrogate`, the latent posterior summary types. |
| Compatibility and acquisition | `mutaco_d_contract/`; `evaluate_confidence_geometry`, `MatchedBarrierObjective`, `evaluate_exploration`, `propose_candidate`. |
| Sequential lifecycle | `mutaco_d_contract/` and `simulator_protocol/`; `run_sequential`, observation/fit/proposal records and named seed coordinates. |
| Final recommendation | `paper_comparison_protocol/`; `recommend_continuous_physical_mae` and the explicitly selected alternative policy functions. |
| Physical model and validation | `graph_oscillator/`, the graph simulator guide and `experiments/graph_forward/`. |
| Comparative claims | `paper_scope/`, the method-specific protocol, `results/experiment_registry.toml` and the applicable versioned result manifest. |

For each reported result, trace the contract version, target/cohort identity, method and terminal-policy identity, source revision, environment, seed declarations, call accounting and evidence manifest. Historical source hashes remain attached to their original commits and paths. A file move can change source provenance even if the scientific expressions and data bytes are preserved.

Accepted older results and new results share the `results/` owner. Superseded or unused records use `scientific_archive/`. Completed packages remain immutable and versioned; a repair, requalification or genuinely new run gets its own record. Keep required failure and diagnostic context with the evidence it explains. The [reproducibility guide](../reproducibility/README.md) distinguishes saved-evidence checks from numerical replay.

## 8. Maintaining these contracts

Update navigation and ownership when files move, while preserving versioned scientific meanings. Changes to formulas, numerical tolerances, component ordering, seed addresses, policy defaults, terminal selection or accepted evidence require their own scientific decision and provenance. Do not regenerate golden outputs or rewrite historical manifests merely to make a structural migration pass.

The source inventories identify unresolved issues, including GP fitting terminology, the continuous terminal optimizer's existing raw-candidate fallback behavior, and a B100 parameter-L2 discrepancy in a figure workflow. Preserve those issues as recorded limitations; moving or documenting code does not resolve them or confer a new qualification.
