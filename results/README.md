# Scientific results and evidence

Use this index to inspect existing data and results without rerunning science.
The [paper scope](../scientific_contracts/paper_scope/README.md) controls current claims;
[experiment_registry.toml](experiment_registry.toml) records lineage.
A result directory is not execution authorization. Keep original bytes, values,
sidecars and provenance intact; derived portable summaries are not raw evidence.

## Inputs and paper benchmark performance

The frozen input is the ordered [Paper-120 v3 cohort](../experiments/graph_inversion/frozen_targets/paper120_v3),
with graph [target/method declarations](../experiments/graph_inversion/configs/paper)
and [MUTACO profiles](../experiments/graph_inversion/configs/profiles).
B40 and B100 are independent campaigns, not prefixes of one optimization run.
Held-out response MAE is distinct from search error and posterior scores.

| Scientific question / evidence family | Existing results and authority |
| --- | --- |
| Primary MUTACO / scalar qLogNEI / Sobol, B40 | [Population manifest](paper/graph_scalar_primary_b40_v1/manifest.json); [frozen trajectory archives](paper/scalar_primary_trajectory_archives_v3r2/README.md) |
| Primary comparison, B100 | [Manifest](paper/graph_scalar_primary_b100_v1/manifest.json), [analysis report](paper/graph_scalar_primary_b100_v1/analysis_report.json), [target/method rows](paper/graph_scalar_primary_b100_v1/analysis_rows.jsonl), [integrity audit](paper/graph_scalar_primary_b100_v1/integrity_audit.json) |
| Cross-budget primary synthesis | [Manifest](paper/scalar_primary_b40_b100_synthesis_v1/manifest.json), [report](paper/scalar_primary_b40_b100_synthesis_v1/synthesis_report.json), [rows](paper/scalar_primary_b40_b100_synthesis_v1/cross_budget_rows.jsonl) |
| Repaired MUTACO / NPE / SNPE paired assessment | [B40 report](paper/repaired_mutaco_sbi_assessment_6251303/v1/analysis_report_b40.json), [B100 report](paper/repaired_mutaco_sbi_assessment_6251303/v1/analysis_report_b100.json), [manifest](paper/repaired_mutaco_sbi_assessment_6251303/v1/manifest.json); [repair contract](../scientific_contracts/repaired_mutaco_sbi_salvage/README.md) |
| NLE/NRE SBI-zoo extension | [Comparison manifest](paper/sbi_zoo_comparison_v1/manifest.json), [summary](paper/sbi_zoo_comparison_v1/summary.json), [rows](paper/sbi_zoo_comparison_v1/rows.csv); [study contract](../scientific_contracts/sbi_zoo_extension/README.md) |
| Controlled B40 terminal-domain B'/C' study | [Refit rows/manifest](paper/terminal_domain_ablation_b40_refit_v1), [authoritative v2 held-out report](paper/terminal_domain_heldout_assessment_b40_v2/analysis_report.json), [v2 manifest](paper/terminal_domain_heldout_assessment_b40_v2/manifest.json) |
| Response-standardization support | [Evidence](paper/graph_paper120/b40_standardization_ablation_v1/manifest.json), [report](../docs/paper/results/graph_paper120_b40_standardization_ablation_v1.md); arm-specific learning streams, not a shared-initial-data causal ablation |
| Cross-experiment semantics | [B40 analysis](paper/cross_experiment_analysis/b40_v1/analysis_report.json), [manifest](paper/cross_experiment_analysis/b40_v1/manifest.json); assessment families are not paired across studies |

### Historical and diagnostic evidence

- Historical MUTACO/NPE B40: [paired records](paper/graph_paper120/paper120/paired_records.jsonl),
  [summary](paper/graph_paper120/paper120/population_summary.json), [report](../docs/results/PAPER120_B40.md).
- Historical three-method SBI: [B40 manifest](paper/graph_paper120/paper120_mutaco_npe_snpe_v1/manifest.json),
  [B100 manifest](paper/graph_paper120/b100_v1/manifest.json),
  [B40 report](../docs/paper/results/graph_paper120_b40_v1.md) and
  [B100 report](../docs/paper/results/graph_paper120_b100_v1.md).
  Their superseded assessment endpoints do not replace repaired SBI evidence.
- The [three-target B100 qualification](paper/graph_paper120/paper120_b100_three_method_qualification_v1/manifest.json) supplied original cases 0, 49 and 99. The [recovery ledger](paper/repaired_mutaco_sbi_recovery_qualification/v1/b100_raw_recovery.json) identifies them separately from the remaining 117 cases. Its three-target summary remains engineering evidence.
- [Terminal-lineage forensics](paper/historical_terminal_lineage_forensics)
  and [qLogNEI fallback forensics](paper/graph_scalar_qlognei_fallback_forensics_v1/manifest.json)
  retain diagnostic claim boundaries.
- [Terminal held-out v1](paper/terminal_domain_heldout_assessment_b40_v1/manifest.json)
  is incomplete, with an attempted outcome unknown; use v2 for the accepted endpoint.
- The [paper packages](paper/) include preparation, source validation and scalar-primary repair receipts that the current population registry cites. Each receipt keeps its original status. The [superseded runs](../scientific_archive/superseded_results/README.md) document the earlier composite-NEI attempts and historical engineering runs.
- [Original compact registry](../scientific_archive/project_records/registry.json) and [content manifest](../scientific_archive/project_records/benchmark_content_manifest.original.json)
  are historical Paper-100 compact-export inventories. Some referenced paths
  belong to the archived research tree and are absent from this curated checkout.
  They are preserved provenance, not a complete live index or a regeneration recipe.

## Figures, source arrays and experiment outputs

| Surface | Existing location / interpretation |
| --- | --- |
| Paper figure registry and persisted-input workflow | [Figure guide](../experiments/paper/figures/README.md), [figure_plan.toml](../experiments/paper/figures/figure_plan.toml) |
| Study-specific figure specifications | [Paper-120 figure specs](../experiments/paper/studies/graph_paper120/figures); the B40/B100 scaffold documents retain their own rendering-authorization status |
| Accepted SBI-zoo comparison figures | [PDF/PNG outputs](../docs/paper/figures/sbi_zoo_comparison_v1), [source](../experiments/paper/studies/graph_paper120/analysis/sbi_zoo_comparison.py), [input specification](../experiments/paper/studies/graph_paper120/analysis/sbi_zoo_comparison_v1.json) |
| Primary and repaired-SBI review plots | [Offline renderer implementations](../experiments/paper/figures/README.md#persisted-input-renderers); physical/mixed reconstruction, source-recorded producers and the failing scatter remain at their [separate boundaries](../experiments/paper/figures/README.md#deliberately-separate-producers) |
| Target-98 archive-only physics analysis | [Response trajectories, gauge contrast and manifest](paper/oscillator_physics_post_analysis/target098_v1) |
| Target-98 sensitivity | [Derived sensitivity and provenance](paper/oscillator_physics_post_analysis/target098_sensitivity_job6352257) |
| Twelve-target sensitivity and cohort figure source | [Worker outputs, plotting data, figures and provenance](paper/oscillator_physics_multi_target_sensitivity/paper120_maximin12_job6353452); the colocated producer is frozen source evidence |
| Convergence and source-validation records | Scalar-primary and repaired-SBI receipts (private historical archive); target-98 best-observed search traces are not held-out convergence curves |

Only claim the data actually present. Some renderers require a separate frozen
array archive or TeX installation; an existing source file does not establish
fresh-machine rendering. Do not overwrite figures to test navigation.

## Code-regression evidence

This is engineering protection for behavior-preserving refactoring, separate
from the 120-target scientific performance comparisons above.

- [B40/B100 observation archives](paper/scalar_primary_trajectory_archives_v3r2/README.md):
  B40 target 16, `/records/48`; B100 target 0, `/records/0`.
- [Original B40 portable reference](reproducibility/shareable_milestone_v1/README.md)
  and [B100 baseline/refactor gate](reproducibility/b100_regression_v1/README.md).
- Accepted runner extractions: [frozen response](reproducibility/shareable_milestone_v1/manifest.json),
  [pre-run options](reproducibility/pre_run_validation_v1/README.md),
  [initial collection](reproducibility/initial_observation_collection_v1/README.md),
  [GP data](reproducibility/gp_training_data_preparation_v1/README.md),
  [sequential commitment](reproducibility/sequential_observation_commit_v1/README.md),
  [fit reporting](reproducibility/gp_fit_result_v1/README.md).
- Acquisition and GP extractions: [raw screening](reproducibility/raw_screening_v1/README.md),
  [refined selection](reproducibility/refined_selection_v1/README.md),
  [GP optimizer diagnostics](reproducibility/gp_optimizer_diagnostics_v1/README.md).
- Test-only history: [NEI fixture diagnosis/correction](reproducibility/test_quality_v1/manifest.json)
  and [v2 zero-failure gate closure](reproducibility/test_quality_closure_v2/README.md).
- [Qualified documented scientific core](reproducibility/documented_core_qualification_v1/README.md):
  source `35ef82d`, 523 passed/8 unchanged skips, exact B40/B100 and fit-event
  comparisons; the eight documentation hashes are qualified without rewriting old evidence.
- [Final qualified validation of the reorganized source](reproducibility/final_qualified_validation_v1/README.md):
  source `4f56343`, the exact 531-case selection, both full historical replays
  and independent raw-reference comparisons; no production or reference changes.

The [final architecture qualification](reproducibility/repository_architecture_qualification_v2/README.md) covers the scientific workflow and identifies the historical and figure comparisons that remain incomplete. The [original TRUBA report and job register](reproducibility/repository_architecture_qualification_v1/README.md) preserve the full run and case outcomes.

The latest saved 531-case suite is 523 passed, 8 skipped, zero failures. Historical baseline
reports correctly retain the old NEI fixture failure. Strict replay matched all
40/100 observations before returning archived responses, plus candidate/context,
GP, RNG, terminal and accounting checkpoints. Reconstructed GP states are not
original historical optimizer snapshots. These gates used zero physical calls.

## Bounded real-simulator integration evidence

The [physical qualification record](reproducibility/physical_integration_qualification_v1/README.md)
reports analytic/graph reference comparisons and a real 16-initial + 1-adaptive
MUTACO run. All scientific checks passed within existing parity thresholds;
Slurm job 6925432 failed afterward during binary evidence export. Original JSON
responses, GP diagnostics and call/event records were independently verified,
with the incomplete tensor capture explicitly retained as a limitation.
This used 29 real simulator API evaluations (62 forward computations including
independent references). It is separate from zero-call historical replay and
paper performance evidence, and does not establish complete paper reproduction.

The [offline export repair and partial recovery](reproducibility/physical_export_recovery_v1/README.md)
restored 14 tensor leaves from retained records. The subsequently approved
[qualified supplementary capture](reproducibility/physical_export_closure_v1/README.md)
completed the declared 236-leaf contract with 222 newly computed leaves, clearly
separated from original values. Job 6925576 completed 33 kernel runs and passed
21 exporter tests, with zero GP fits, simulator API calls or end-to-end repeats.
Original job 6925432 remains FAILED and its sealed evidence is unchanged; the
lost original dump was not recovered. This closes the bounded export task,
without claiming complete paper reproducibility.

## Verification without new science

The [data and software availability statement](../DATA_AVAILABILITY.md) lists the public datasets, licenses, and public trajectory checks. Original scheduler source-validation receipts are held in the private historical archive.

Start with [read-only reproducibility commands](../reproducibility/README.md)
for portable B40/B100 packages and saved checkpoint/history/event/test reports.
These checks reuse the checksum-recorded implementations and do not fit models,
replay the algorithm or submit jobs. Private-original verification is a separate,
explicit input requirement; a successful portable check does not claim it.

Use the [local integrity commands](../infrastructure/local/README.md#integrity-only-no-gp-fitting-or-replay)
for portable evidence and archive member/sidecar checks. Every reproducibility
milestone has its own `SHA256SUMS`; on macOS, run `shasum -a 256 -c SHA256SUMS`
inside that directory (Linux: `sha256sum -c SHA256SUMS`). A checksum verifies
bytes, not scientific claims or fresh historical reproduction. No archive is
regenerated by an integrity check. Exact environment comparison also requires
privately preserved raw reference packets, as the
[validation contract](../scientific_archive/validation_history/refactor_validation/README.md) explains.

## Reading records across the migration

Start with the [experiment registry](experiment_registry.toml) for the declared
population, loss, comparison family and evidence status. Use the scientific
index above or [resolve_recorded_path](../reproducibility/recorded_paths.py) to
open a sealed bundle whose registry entry retains an older pathname. Within a bundle, keep sidecars alongside their data
files. A held-out MAE averages the absolute component differences between the
fresh simulator response at the recommended parameters and the stored physical
target response. Acquisition scores and the terminal GP score answer
different questions. A qualification receipt describes the source, inputs and
operational outcome of one gate. It does not supply population measurements.

The historical three-method SBI record bundles are stored directly in
[the B40 result directory](paper/graph_paper120/paper120_mutaco_npe_snpe_v1)
and [the B100 result directory](paper/graph_paper120/b100_v1). They retain
their original file bytes. Historical repaired-SBI inventory identities
include the native paths of the compact row files; reproduce those identities
in a separate checkout of the recorded source. The [offline verification
guide](../reproducibility/README.md#reading-historical-command-lines) shows the
checkout boundary. The twelve-target oscillator package retains its physical
location because its sealed figure producer derives its repository root from
its own filename.

Existing declarations, Git revisions and SHA-256 manifests retain their
original path strings. For example, the historical SNPE qualification identifies
its manifest under `benchmarks/graph/`. Its unchanged manifest is available
under `results/paper/graph_paper120/` today. Use the [checkout inventory](../docs/decisions/legacy_checkout_paths.md)
and the [revision-specific ledger](../docs/decisions/repository_migration_manifest.json)
to look up retired addresses. A Git lookup at a recorded historical revision
uses that revision's path. Enable Git symlinks before checkout on Windows when
working with an older layout. The source-validation receipts have individual
compatibility links so undeclared raw outputs from a research run continue to
go to the experiment workspace. New accepted results require their own
versioned directory and manifest.
