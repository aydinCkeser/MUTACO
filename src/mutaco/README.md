# MUTACO core: scientific navigation and paper-to-code map

Start with [sequential_search/sequential_runner.py](sequential_search/sequential_runner.py), function
`run_sequential`. This is the production sequential algorithm. Public consumer
imports are in [api.py](api.py) and [__init__.py](__init__.py); experiment bindings
live outside the core. This is the authoritative navigation map. The earlier
[submission-readiness audit](../../scientific_archive/validation_history/refactor_validation/submission_readiness_audit.md)
is a dated review/evidence snapshot.

The manuscript anchors are Algorithm `app:mutaco_algorithm` and appendices
`app:gp_surrogate`, `app:gp_acquisition`, `app:gp_terminal` in
[MUTACO preprint v2](https://arxiv.org/abs/2610.04619v2).
The map describes the accepted graph B40/B100 profiles; the generic API supports
other registered options. Terminology differences do not authorize policy fixes.

## Follow one sequential run

| Scientific step | Source / important symbol | Inputs → outputs and boundary |
| --- | --- | --- |
| Select options | [paper_policies.py](../../experiments/graph_inversion/paper_policies.py), `load_graph_policy_profile`; [mutaco_reference.py](../../experiments/benchmarking/methods/mutaco_reference.py), `run_mutaco_target` | Experiment profile → typed intent and graph binding; one shared core |
| Validate and address randomness | [sequential_runner.py](sequential_search/sequential_runner.py), `run_sequential`, `_validate_pre_run_options`; [deterministic_seed_derivation.py](randomness/deterministic_seed_derivation.py), `SeedDeriver` | Capabilities/options/root seed → validated run and coordinate-addressed seeds |
| Scrambled Sobol initialization | [initial_design.py](design/initial_design.py), `generate_initial_design`; runner `_initial_design` | Bounds/count/seed → ordered physical points; CPU generation then explicit transfer |
| Collect initial observations | runner `_collect_initial_observations`, `_evaluate` | Points/contexts → ordered raw observations, attempts and partial commits; 16 in these profiles |
| Freeze response statistics | [response_standardization.py](response/response_standardization.py), `FrozenResponseTransform`; runner `_prepare_frozen_responses` | Initial responses → frozen means/scales, identity and standardized observations; statistics stay fixed |
| Assemble GP data | runner `_prepare_gp_training_data` | Ordered observations + transform → input/target tensors and training identities |
| Correlated exact multitask GP | [exact_multitask_gp.py](surrogate/exact_multitask_gp.py), `_build_model`, `fit_exact_multitask_gp`, `fit_exact_multitask_gp_with_recovery` | Normalized/task-augmented inputs → model/diagnostics; registered priors, constraints, seeded attempts/recovery |
| Record fit outcome | GP `_summarize_fit_optimizer_result`; runner `_record_gp_fit_result`, nested `fit_version` | Fit result → progress/events/accounting; numerical fitting remains separate |
| Raw scoring and screening | [candidate_proposal.py](acquisition/candidate_proposal.py), `propose_candidate`, `_screen_and_rank_raw_candidates`; [acquisition_objectives.py](acquisition/acquisition_objectives.py), [confidence_geometry.py](acquisition/confidence_geometry.py) | Posterior moments/target → compatibility, barrier/exploration scores and raw ranking |
| Choose branch and refine | proposal `_barrier_restart`, `_exploration_restart` | Strict compatibility interior → barrier; otherwise configured uncertainty exploration; seeded starts |
| Select non-worsening outcome | proposal `_select_nonworsening_refined_outcome` | Finite in-bounds refined outcomes + raw best → outcome/raw fallback; exact ties retain earliest order |
| Recover a duplicate | runner explicit duplicate path; proposal `recover_best_ranked_unevaluated_raw` | Proposed point + evaluated points/ranking → unevaluated fallback or existing failure |
| Evaluate and commit | runner `_evaluate`, `_standardize_and_commit_sequential_observation` | Candidate/context → raw evaluation first, then frozen standardization and ordered commitment |
| Update GP | runner nested `fit_version` | Enlarged ordered observations → next fit; repeat until budget exhausted |
| Terminal recommendation | [physical MAE terminal recommendation](recommendation/physical_mae_terminal_recommendation.py), `recommend_continuous_physical_mae`; [continuous terminal optimizer](recommendation/_continuous_terminal_optimizer.py) | Final GP/transform → continuous posterior-mean physical-MAE recommendation; no new observation |

Production fitting is MAP-type with registered priors. Accepted reference
profiles use 12 optimization iterations; this is not prior-free MML. Fresh
held-out physical assessment is owned by the experiment protocol. Seeds,
response order, precision and failure accounting are scientific contracts.

## Supporting modules

| Package / module | What to look for |
| --- | --- |
| [parameter and response specifications](parameter_and_response/README.md) | Parameter/response specifications, observations and run intent |
| [scientific policy registry](policies/scientific_policy_registry.py), [policy surface](policies/policy_surface.py) | Registered option identities and admissibility |
| [identity](identity), [randomness](randomness) | Canonical identities and named seed coordinates |
| [posterior](posterior) | Moment shapes, covariance validation and diagnostics |
| [simulator/simulator_adapter_contracts.py](simulator/simulator_adapter_contracts.py) | Capabilities, `SimulationContext`, checked adapter calls; no physical kernel |
| [recommendation](recommendation/README.md) | Paper terminal objective, shared evaluated designs and continuous optimizer |
| [device and resources](device_and_resources), [observability](observability), [runtime](runtime) | Device/precision selection, CPU parallelism and resource sampling, alongside run events and runtime controls |
| [file management](file_management/README.md), [file_management/atomic_file_writes.py](file_management/atomic_file_writes.py) | Artifact records and atomic writing |
| [comparators/composite_mc_nei.py](comparators/composite_mc_nei.py) | Composite NEI scores joint samples of the response-vector GP. The paper comparator fits a scalar model in the [benchmark method](../../experiments/benchmarking/methods/scalar_qlognei.py). |

Concrete physics lives in [the graph simulator](../../simulators/README.md) and [analytic forward models](../../experiments/analytic_target_matching/forward_models/README.md). The
[guarded test selection](../../infrastructure/local/README.md#guarded-current-scientificimport-suite)
is narrower than all repository tests. Exact B40/B100 evidence and limitations
are indexed in [results](../../results/README.md#code-regression-evidence).
The pre-existing shadowed GP diagnostic and terminal-policy questions remain
separate researcher decisions. The [migration ledger](../../docs/decisions/repository_migration_map.md) documents compatible imports and the scope of checks for this tree.
