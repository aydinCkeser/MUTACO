# Research experiment catalog

Read the [scientific execution map](../src/mutaco/README.md) for the algorithm,
[paper scope](../scientific_contracts/paper_scope/README.md) for claim authority and
[evidence index](../results/README.md) for existing results. This catalog
identifies source entry points, not commands authorizing new runs. Use approved
[local launchers](../infrastructure/local/README.md); some older sub-guides
retain historical interpreter/setup instructions and unavailable archive paths.

**Execution vocabulary:** “analytic” evaluates an analytic test forward;
“physical” can run the graph-oscillator kernel; “offline” consumes existing
artifacts without simulator calls. Import safety does not imply execution safety.
Never discover programs by importing the whole tree. Default pytest also
includes real forward-model workflows outside the guarded selected suite.

## Shared values, methods and paper decisions

[contracts/](contracts/README.md) owns only the shared call-count value and
compatible validation exception. [benchmarking/contracts.py](benchmarking/contracts.py)
keeps adapter/result interfaces and benchmark budget/method rules;
[benchmarking/methods/](benchmarking/methods) implements the methods.
[protocol_v3.py](paper/protocol/protocol_v3.py) keeps paper-specific decisions,
paired assessment, schemas and scientific identities. Paper package exports
load their owning modules on demand, without an SBI initialization prerequisite.
Execution remains distinct from [offline analysis](paper/analysis) of saved results.

## Analytic and generic benchmarking programs

| Program / purpose | Principal entry points | Inputs → artifacts; paper relation | Can execution evaluate a simulator? |
| --- | --- | --- | --- |
| [Analytic target matching](analytic_target_matching/README.md): transparent handwritten and frozen-GP examples | [run_handwritten.py](analytic_target_matching/run_handwritten.py), [run_gp_landscape.py](analytic_target_matching/run_gp_landscape.py), [run_suite.py](analytic_target_matching/run_suite.py) | [Configurations](analytic_target_matching/configs/) → result/history JSON and SVG under ignored `runs/analytic_target_matching`; acceptance/mechanism evidence | Yes, analytic forward; no graph physics |
| [Noisy hidden-target cohort](analytic_target_matching/noisy_cohort/README.md) | [run.py](analytic_target_matching/noisy_cohort/run.py); [plot.py](analytic_target_matching/noisy_cohort/plot.py) offline | [Noisy cohort config](analytic_target_matching/configs/noisy_cohort_100.json) → target bank, per-target history/assessment, metrics and boxplots; analytic support | Run: analytic; plot: offline |
| [Analytic SBI comparison/calibration](analytic_target_matching/sbi_benchmark/README.md) | [run.py](analytic_target_matching/sbi_benchmark/run.py), [run_calibration.py](analytic_target_matching/sbi_benchmark/run_calibration.py), [run_prompt5d.py](analytic_target_matching/sbi_benchmark/run_prompt5d.py); summarize/plot separate | [Benchmark configs](analytic_target_matching/configs/benchmark_campaigns/) → ignored `runs/benchmarks/analytic`; smoke/calibration differ from completed Prompt-5D research, whose original bundles are not all present here | Analytic forward and SBI training; retained completed/failed runs must not be overwritten |
| [Paper handwritten](paper/studies/analytic_handwritten/README.md) and [frozen-GP landscape](paper/studies/analytic_gp_landscape/README.md) | [handwritten/run.py](paper/studies/analytic_handwritten/run.py), [landscape/run.py](paper/studies/analytic_gp_landscape/run.py) | Colocated [handwritten configs](paper/analytic_handwritten/configs) / [landscape configs](paper/analytic_gp_landscape/configs) → benchmark evidence; declared paper-support families, not graph population results | Analytic forward and configured methods |
| Shared benchmark engine | [benchmarking/api.py](benchmarking/api.py), `run_campaign`; [calibration.py](benchmarking/calibration.py) | Typed/configured adapter, methods and budgets → run manifests, target/method records, assessment and timing | Yes, adapter-dependent; `engine/reference.py:replay_reference` reexecutes scientific work, unlike strict archived-response replay |
| Historical compatibility | [analytic/prompt5_acceptance.py](analytic/prompt5_acceptance.py), [prompt5_support.py](analytic/prompt5_support.py), [benchmarking/prompt5d.py](benchmarking/prompt5d.py) | Retained old invocation/import paths delegate to the owning program; no second MUTACO | Same as delegated program |

## Graph model and Paper-120 research

| Program / purpose | Principal entry points | Input/configuration → existing evidence | Physical execution boundary / paper relation |
| --- | --- | --- | --- |
| [Forward-model qualification](graph_forward/README.md) | [run.py](graph_forward/run.py), [run_parity.py](graph_forward/run_parity.py), [profile.py](graph_forward/profile.py), [validate_cuda.py](graph_forward/validate_cuda.py); [plot.py](graph_forward/plot.py) offline | Concrete graph package/configuration → raw/summary arrays and hash manifest in requested output | Run/parity/profile/CUDA invoke physical kernels; plotting loads artifacts; engineering qualification, not inverse-method performance |
| [Graph inversion and Paper-v3 population](graph_inversion/README.md) | [run_paper_v3_worker.py](graph_inversion/run_paper_v3_worker.py), [paper_v3_callbacks.py](graph_inversion/paper_v3_callbacks.py), [paper_policies.py](graph_inversion/paper_policies.py) | [Paper declarations/configs](paper/configs/graph_inversion), [profiles](graph_inversion/configs/profiles), [frozen cohort](graph_inversion/frozen_targets/paper120_v3) → [historical B40](../results/paper/graph_paper120/paper120) / [B100](../results/paper/graph_paper120/b100_v1) | Workers and target-generation/freezing paths can evaluate physics. Historical SBI assessments are superseded by repaired endpoints |
| Current scalar qLogNEI/Sobol primary comparison | [primary_benchmark/scalar_protocol.py](paper/primary_benchmark/scalar_protocol.py), [execution.py](paper/primary_benchmark/execution.py), [worker.py](paper/primary_benchmark/worker.py); [population preparation](graph_inversion/prepare_scalar_primary_population.py) | [Study specification](paper/studies/graph_paper120/studies/mutaco_scalar_qlognei_sobol_primary_v1.json) + graph declarations → [B40](../results/paper/graph_scalar_primary_b40_v1), [B100](../results/paper/graph_scalar_primary_b100_v1), [synthesis](../results/paper/scalar_primary_b40_b100_synthesis_v1) | Execution/worker physical; preparation alone is not submission; current main-paper optimization benchmark |
| Historical composite-NEI qualification | [primary_benchmark/qualification.py](paper/primary_benchmark/qualification.py), [run_primary_q1_worker.py](graph_inversion/run_primary_q1_worker.py), [diagnosis](graph_inversion/diagnose_nei_baseline_factorization.py) | [Historical protocol](../scientific_contracts/primary_benchmark_protocol/README.md), retained NEI declarations → failed/qualification/repair manifests in [paper evidence](../results/paper) | Worker physical; diagnostics have their declared zero-call scopes. Retired comparator; do not revive or pool as current evidence |
| Repaired SBI terminal/recovery/assessment | [common/repaired_sbi_recovery.py](paper/common/repaired_sbi_recovery.py), [terminal preparation](paper/common/repaired_sbi_terminal_preparation.py), [assessment worker](graph_inversion/run_repaired_sbi_assessment_worker.py) | [Analysis specs](paper/studies/graph_paper120/analysis) + frozen recommendations/source identities → [repaired assessment](../results/paper/repaired_mutaco_sbi_assessment_6251303/v1) | Recovery/terminal reconstruction fits stored data; assessment execution invokes physics. Authoritative paired SBI endpoint; [contract](../scientific_contracts/repaired_mutaco_sbi_salvage/README.md) |
| SBI-zoo extension: NLE/NRE | [sbi_zoo_qualification.py](graph_inversion/sbi_zoo_qualification.py), [sbi_zoo_p1.py](graph_inversion/sbi_zoo_p1.py), [p2](graph_inversion/sbi_zoo_p2.py), [p3](graph_inversion/sbi_zoo_p3.py), [comparison](paper/studies/graph_paper120/analysis/sbi_zoo_comparison.py) | [Extension spec](paper/studies/graph_paper120/analysis/sbi_zoo_extension_v1.json) → shared data, recommendations, held-out assessments and [comparison evidence](../results/paper/sbi_zoo_comparison_v1) | Qualification/P1 data/P3 assessment can invoke physics; P2 trains on stored data; comparison offline. [Scope](../scientific_contracts/sbi_zoo_extension/README.md) |
| Response-standardization comparison | [Paper-v3 worker](graph_inversion/run_paper_v3_worker.py), profiles and [study](paper/studies/graph_paper120/studies/b40_standardization_ablation_v1.json) | [Evidence](../results/paper/graph_paper120/b40_standardization_ablation_v1), [report](../docs/paper/results/graph_paper120_b40_standardization_ablation_v1.md) | Physical campaigns; supporting evidence with arm-specific learning streams, not fixed-trajectory ablation |
| Controlled B40 terminal-domain study | [terminal_domain_ablation.py](paper/common/terminal_domain_ablation.py), [heldout v2](paper/common/terminal_domain_heldout_assessment_v2.py) | [Refit spec](paper/studies/graph_paper120/analysis/terminal_domain_ablation_b40_refit_v1.json), [v2 assessment spec](paper/studies/graph_paper120/analysis/terminal_domain_heldout_assessment_b40_v2.json) → [refit](../results/paper/terminal_domain_ablation_b40_refit_v1), [v2 endpoint](../results/paper/terminal_domain_heldout_assessment_b40_v2) | Stored-data refit/optimization versus physical held-out workers; v1 remains incomplete provenance; narrow B40 mechanism claim |
| Target-98 oscillator post-analysis/sensitivity | [analysis.py](paper/studies/oscillator_physics_post_analysis/analysis.py), [guide](paper/studies/oscillator_physics_post_analysis/README.md) | [Config](paper/studies/oscillator_physics_post_analysis/configs/target098_sensitivity_v1.json) → [archive-only outputs](../results/paper/oscillator_physics_post_analysis/target098_v1), [sensitivity/provenance](paper/oscillator_physics_post_analysis/outputs/target098_sensitivity_job6352257) | `zero-call` reads archives; `execute-worker` is an explicitly gated eight-call physical design; illustrative physics support |
| Twelve-target local sensitivity | [study.py](paper/studies/oscillator_physics_multi_target_sensitivity/study.py), [guide](paper/studies/oscillator_physics_multi_target_sensitivity/README.md) | [Maximin-12 config](paper/studies/oscillator_physics_multi_target_sensitivity/configs/paper120_maximin12_sensitivity_v1.json) → [worker outputs/plotting data/provenance](paper/oscillator_physics_multi_target_sensitivity/outputs/paper120_maximin12_job6353452) | Declaration/preflight versus separately gated 96-call worker; local-physics diagnostic, not an inverse-method score |

The generic [run_experiment.py](graph_inversion/run_experiment.py) is a retained,
unsupported historical interface: import and `--help` are safe, but execution
fails on missing historical reporting modules **before** configuration loading
or any campaign. No replacement analysis was invented. Use the
[supported entry-point guide](graph_inversion/README.md#supported-entry-points).
`paper/primary_benchmark/worker.py` is a Python orchestration library, not a CLI.

## Offline evidence, figures and reproducibility

| Program / purpose | Entry points and inputs | Outputs / scientific scope / simulator boundary |
| --- | --- | --- |
| Paper Lab | [workflow guide](paper/workflow/README.md), [cli.py](paper/workflow/cli.py); [studies](paper/studies) | Validate/ingest/verify/report/promote evidence; no simulator calls. Commands can still write/transfer evidence; use their explicit authorization flags |
| Offline summaries and cross-study analysis | [analysis package](paper/analysis), [evidence_analysis_cli.py](paper/analysis/evidence_analysis_cli.py), [cross_budget_synthesis_cli.py](paper/analysis/cross_budget_synthesis_cli.py); [analysis specs](paper/studies/graph_paper120/analysis) | [Cross-experiment results](../results/paper/cross_experiment_analysis), [synthesis](../results/paper/scalar_primary_b40_b100_synthesis_v1); deterministic statistics, persisted timings and evidence checks, no GP fits or physical calls |
| Historical terminal lineage | [terminal_lineage_forensics_cli.py](paper/historical_workflows/terminal_lineage_forensics_cli.py) and study specs above | [Forensic results](../results/paper/historical_terminal_lineage_forensics); retained source/provenance-specific analysis with separate claim boundaries |
| Exact-source/model reconstruction | [exact_source_bootstrap.py](paper/common/exact_source_bootstrap.py), [exact_model_replay.py](paper/common/exact_model_replay.py), [exact_terminal_generation.py](paper/common/exact_terminal_generation.py) | Restore specified scientific source and fit stored observations; GP numerical work, not a portable plotting shortcut or physical campaign |
| Offline rendering and figure framework | [five persisted-input renderers](paper/figures/README.md#persisted-input-renderers), [figure registry](paper/figures/figure_plan.toml) | Offline plots from frozen results, with safe help and legacy entry-point compatibility; source arrays, original figures and registry permissions remain unchanged |
| Physical/mixed exploration and frozen producers | [exploration/safety guide](paper/exploration/README.md), [retained-producer boundaries](paper/figures/README.md#deliberately-separate-producers) | Reconstruction can invoke physics. Source-recorded producers and the currently failing scatter loader remain in place; these are not interchangeable with offline renderers |
| Strict code-regression replay/diagnostics | [scalar_primary_replay.py](paper/exploration/scalar_primary_replay.py), [replay_divergence.py](paper/exploration/replay_divergence.py); [archives](../results/paper/scalar_primary_trajectory_archives_v3r2/README.md) | Guarded archived observations only: verify proposed candidate/context before returning a response, stop on mismatch. GP fits occur; physical entry points are blocked. [Saved acceptance evidence](../results/README.md#code-regression-evidence), not new performance experiments |

[analysis](paper/analysis) owns offline aggregation, timing reports and the two
analysis CLIs; see the [package boundaries and legacy commands](paper/README.md#offline-analysis).
[protocol](paper/protocol/README.md) owns campaign, target-freezing and append contracts. [Historical workflows](paper/historical_workflows/README.md) links source-specific reconstruction and repair programs. [Response reconstruction](paper/response_reconstruction/README.md) identifies which commands can evaluate physics. The old common module paths still serve historical loaders and external callers. The catalog does
not authorize repetitions of any completed campaign, figure overwrites, new
TRUBA submissions or changes to the frozen inputs/accepted scientific policies.
