# MUTACO repository migration map

The isolated branch `refactor/repository-architecture` began at `9f3ce1893112bb37c23e3aad1508564f55d2c81b`. The local and fetched canonical development branches matched. The base tree contained all 2,555 paths listed in the preparation manifest. The original checkout retains its untracked audit.

The [per-file ledger](repository_migration_manifest.json) records base paths, actual destinations, original SHA-256 values, reviewed content hashes and individual exceptions. The [validation record](repository_validation_plan.md) distinguishes tests run here from scientific qualification tied to earlier revisions. The tables below preserve approved ownership and constraints. An approved destination alone is not evidence of a physical relocation.

## Public release candidate

The pinned release candidate retains the scientific implementation. It adds release licensing and citation metadata, public derivatives of the two trajectory archives, and saved-evidence instructions. Original scheduler receipts, local account paths, and manuscript source bundles are handled separately. The historical paths, identities, and byte checks below apply to the verified private archive. See [Data availability](../../DATA_AVAILABILITY.md).

The licensing documentation records the contributors' BSD-3-Clause and CC BY 4.0 agreements and credits the software authors listed in [CITATION.cff](../../CITATION.cff). This documentation update preserves the validated source, configuration, datasets, and evidence.

## Approved ownership

| Canonical area | Responsibility |
| --- | --- |
| `src/mutaco/` | The reusable sequential inference algorithm and its public API. The implemented filenames are listed below. |
| `simulators/graph-simulator/` | The physical graph oscillator forward model, its scientific and retained profiles, and its independent reference implementation. Installed import name stays `mutaco_graph`. |
| `experiments/analytic_target_matching/` | Analytic target-matching studies, their configs, actual analytic forward functions, noisy-cohort programs and SBI benchmarks. Installed analytic import name stays `mutaco_analytic`. |
| `experiments/benchmarking/` | Shared experiment comparison machinery, methods, simulator adapters, stored-result utilities and accounting. |
| `experiments/graph_forward/` | Forward evaluation, parity checks, device checks, profiling and forward artifacts. The name remains `graph_forward`. |
| `experiments/graph_inversion/` | Graph-specific inversion bindings, callbacks, workers, declared targets and frozen inputs. |
| `experiments/paper/protocol/` | Shared paper campaign, target-freezing, population, method-isolation, validation and append protocols. |
| `experiments/paper/studies/` | Individual paper studies and their local configs, tests and scheduler profiles. |
| `experiments/paper/analysis/` | Analysis of stored scientific evidence. |
| `experiments/paper/figures/` | Figure preparation and rendering, with the five common helpers directly in this directory. |
| `experiments/paper/response_reconstruction/` | Response reconstruction and inspection, with explicit distinction between saved-array rendering and commands that call physics or replay. |
| `experiments/paper/historical_workflows/` | Source-sensitive reconstruction, replay and repairs. This describes provenance; it does not mean all code or evidence here is obsolete. |
| `experiments/paper/workflow/` | Experiment record preparation, ingestion, consolidation, reporting, transport and promotion. |
| `experiments/contracts/` | Existing shared Python identities, including `PhysicalCallCounts` and `PaperContractError`. |
| `results/` | Scientific results and supporting evidence still used in current research, whether produced recently or long ago. |
| `scientific_archive/` | Superseded or unused results, historical exploration, validation history and old project coordination records. |
| `scientific_contracts/` | Scientific definitions, normative and candidate contracts, support tables, interpretation rules and the scientific registry. |
| `reproducibility/` | Explicit saved-evidence verification. Verification does not silently fit a GP, run physics, launch tests, transfer files or submit jobs. |
| `infrastructure/` | Local runtime setup, environment captures, TRUBA execution, scheduling, transport and the retained provenance-bound verifier sources. |
| `tests/` | Existing scientific and engineering acceptance cases, independent fixtures and new bounded migration checks where needed. |
| `docs/` | Manuscript, explanatory reports, current navigation, decisions and historical documentation. TeX assets remain adjacent to their manuscript. |

The final checkout removes all seven old roots: `benchmarks/`, `configs/`, `data/`, `packages/`, `scientific/`, `tools/` and `project/`. The [checkout guide](legacy_checkout_paths.md) supersedes the earlier physical-path exceptions in this historical decision ledger. Exact old source paths belong to revision-specific, detached checkouts. On TRUBA, job `7012390` qualified this layout at `7a1652d92432` against the 531 scientific tests and both archived trajectories. [Qualification details](legacy_checkout_paths.md#qualified-truba-check).

The root keeps `.gitattributes`, `.gitignore`, `AGENTS.md`, `README.md`, `conftest.py` and `pyproject.toml`. `CHANGELOG.md` is archived as explicitly requested.

## Decision ledger

| User decision | Effective approved instruction |
| --- | --- |
| 01–18 | See the verified implementation ledger below for recovered decisions, implemented paths and any remaining gaps. |
| 19 | Use the name `scientific_archive`. The later decision 45 replaces the initial idea of sending all existing benchmark evidence there. |
| 20 | Distribute root configs to their actual owners: analytic configs to `experiments/analytic_target_matching/configs/`, analytic benchmark campaigns to its `benchmark_campaigns/`, environment captures to `infrastructure/environments/`, plotting style with noisy-cohort plotting. Investigate legacy smoke declarations before treating them as unused. |
| 21 | Rename scientific authority area to `scientific_contracts/`; provide a detailed explanatory English README linking math, code, experiments, evidence and manuscript. |
| 22 | Move historical validation/lineage documents to `scientific_archive/validation_history/`, retaining active acceptance guidance and current evidence links in current navigation. |
| 23 | Move physical graph package under `simulators/graph-simulator/`; put analytic functions in `experiments/analytic_target_matching/forward_models/`. Preserve separate package installs and the existing public imports. |
| 24 | Remove `tools/` as an ordinary owner; verification belongs to `reproducibility/`, physical fixture capture to the graph simulator owner. Source-pinned tools may remain as explicit exceptions. |
| 25 | Remove root `project/` without creating `research_registry/`. Decision 46 supplies the final disposition of active registries and historical coordination. |
| 26 | Keep manuscript, figures and explanatory reports in `docs/`. Use existing `docs/decisions/`; move old status/strategy documents to `docs/history/`; update documentation authority. |
| 27 | Move the frozen 242-file paper120 target input to `experiments/graph_inversion/frozen_targets/paper120_v3/` if its complete reference contract can be preserved. No target, digest or declared identity changes. |
| 28 | Keep `infrastructure/`, with `local/`, `truba/` and the environment owner; explain execution boundaries. |
| 29 | Keep TRUBA responsibilities intact rather than mechanically subdividing its 106 audited files. Preserve the six exact-source verifier implementations. |
| 30 | Remove ordinary historical output storage from experiment-code owners. The 260 tracked output files follow the revised results/archive rule in 45. |
| 31 | Keep `analytic_target_matching` as the single main analytic family. Retain old `experiments/analytic` paths only as required compatibility bindings. |
| 32 | Keep shared benchmarking; rename its `simulators/` to `simulator_adapters/`; separate exactly the stored-result API into `results.py`; preserve public bindings, row order and method identities. |
| 33 | Keep the name and location `experiments/graph_forward/`; explain its actual work. Do not use `qualification` as its replacement name. |
| 34 | Separate graph-specific inversion machinery from paper protocols. Paper campaign configs target `experiments/paper/configs/graph_inversion/`; inspect physical target/cohort definitions separately. |
| 35 | Move both analytic paper campaigns into `experiments/paper/studies/`, keeping their configs and run entrypoints together without duplicating the forward functions. |
| 36 | Move the twelve named shared paper protocol modules to `experiments/paper/protocol/`; preserve their distinct accounting, freezing and append responsibilities. |
| 37 | Give source-sensitive reconstruction/repair/replay code the `historical_workflows/` owner, with original physical paths retained where exact-source contracts require them. |
| 38 | Replace mixed exploration ownership with response reconstruction, historical workflows, figures and workflow owners. Preserve the unresolved Fig. 6 exact-loader issue. |
| 39 | Put both oscillator studies under `experiments/paper/studies/`, with their study-local configs, tests, schemas and TRUBA profiles. Route their outputs using 45. |
| 40 | Keep paper analysis, figures and workflow distinct. Flatten the five figure helpers. New renders target versioned result locations, leaving established manuscript figures intact. |
| 41 | Keep `experiments/contracts/` and original Python type/exception identities. This is not a replacement for scientific prose contracts. |
| 42 | Preserve the existing test tree, scientific expectations, independent fixtures and historical identities. Adapt only current import/location expectations when justified by a real move. |
| 43 | Keep six necessary root files and package boundaries. Archive `CHANGELOG.md` to `scientific_archive/project_records/CHANGELOG.md`. |
| 44 | Define `reproducibility/` narrowly as saved-evidence verification. Keep source-bound TRUBA implementations and the pinned tensor writer in their original locations where required. |
| 45 | Current-use scientific results, including past results, belong in `results/`. Unused/superseded history belongs in `scientific_archive/`. Preserve supporting failed/repair records needed to interpret current results. |
| 46 | Move scientific registry to `scientific_contracts/scientific_registry.toml` and experiment registry to `results/experiment_registry.toml`. Review active state fields, then archive original state/schema and old coordination records. |
| 47 | Maintain this migration map with old path, actual target, operation, scientific risk, compatibility and verification evidence. Use the three relocation modes below; preserve science and historical provenance. |

## 4. Later decisions that take precedence

**Results classification:** decision 45 replaces every earlier age-based or blanket archive move in decisions 19, 22, 30 and 39. A completed B40 or B100 result used in the paper is a result. A failed precursor used to explain an accepted package may remain part of its supporting evidence. A retired experiment with no current role belongs in the archive. The date and folder name do not decide this.

**Registries and coordination:** decision 46 replaces the earlier proposal to archive all project TOMLs or put the named handoff notes in `docs/history/`. Scientific and experiment registries remain active under their owning areas. `HANDOFF.md`, `FRESH_CHAT_PROMPT.md` and `UBUNTU_CONTINUATION.md` go to `scientific_archive/project_records/`. Other historical status/strategy documents can still belong in `docs/history/`.

**Forward-model naming:** the analytic benchmark forward functions do not go under `simulators/`. The physical graph model does. The workflow that exercises it remains `experiments/graph_forward/`.

**Verification and writers:** `tensor_export.py` writes new tensor evidence and has retained source bindings. Its bytes now reside in `reproducibility/tensor_export.py`. The original path is used only in the recorded historical checkout.

The [legacy checkout inventory](legacy_checkout_paths.md) records the removal of the final seven legacy roots and the independent historical-checkout procedure. The original revision-specific hashes in the per-file ledger still apply to their recorded revisions.

## 5. Three physical migration modes

| Mode | What changes | Required evidence |
| --- | --- | --- |
| Relocation | The implementation or record moves to its approved owner. Mutable consumers follow it. | Byte/content verification as appropriate, reference resolution, matching API and scientific behavior. |
| Relocation with compatibility | One canonical implementation moves; old supported import/CLI addresses delegate to the same implementation. | Fresh imports in both orders, shared object/global identity, required serialization lookup, CLI output/errors and optional dependency isolation. |
| Exact-source exception | A specific implementation or artifact remains at its original physical address; its logical owner documents the exception. | Named binding that prevents a safe move, original commit/path/hash and an explicit record of which proof would permit future relocation. |

An import alias can preserve shared globals; it cannot by itself preserve `inspect.getsourcefile`, every reload/discovery behavior, a historical Git blob or a literal filesystem location. There must not be two authoritative implementations or two silently divergent datasets.

The per-file ledger records both the canonical address and the physical blob path for native exceptions. Its hashes are tied to the two stated revisions.

## 6. Concrete migration map for the main areas

| Current path or group | Canonical destination | Key acceptance condition |
| --- | --- | --- |
| `packages/graph-simulator/` | `simulators/graph-simulator/` | `mutaco_graph` standalone wheel and original scientific/retained distinction remain. |
| `packages/analytic-simulator/src/mutaco_analytic/` | `experiments/analytic_target_matching/forward_models/` | Explicit package mapping keeps `mutaco_analytic` independently installable; no second module identity. |
| `configs/analytic/` and `configs/experiments/analytic_target_matching/` | `experiments/analytic_target_matching/configs/` | Config identities, version variants and supported CLI defaults remain. |
| `configs/benchmarks/analytic/` | `experiments/analytic_target_matching/configs/benchmark_campaigns/` | Campaign variants are not merged. |
| `configs/environments/` | `infrastructure/environments/` | Exact package versions and native qualification meanings remain. |
| `data/cohorts/paper120_v3/` | `experiments/graph_inversion/frozen_targets/paper120_v3/` | All 242 members and recorded references remain resolvable, without re-hashing changed inputs. |
| `scientific/`, except historical validation groups | `scientific_contracts/` | Contract classifications and normative reference fixtures remain intact. |
| `scientific/refactor_validation/` | `scientific_archive/validation_history/refactor_validation/` | Retain original history; preserve current same-environment acceptance guidance in current navigation. |
| `scientific/historical_terminal_lineage_forensics/` | `scientific_archive/validation_history/historical_terminal_lineage_forensics/` | Old/corrected terminal lineage and current evidence links remain distinguishable. |
| `tools/reproducibility/{verify,verification}.py` | `reproducibility/` | Current API and command under reproducibility; the original checkout supplies the historic tooling addresses. |
| `paper/common/hash_pair_inventory_cli.py` | `reproducibility/hash_pair_inventory.py` | Stdlib-only standalone use outside checkout, ordered JSON, stderr and exit semantics remain. |
| `experiments/benchmarking/api.py` result-view definitions | `experiments/benchmarking/results.py` | Exact six-name extraction described below, old exports retained. |
| `experiments/benchmarking/simulators/` | `experiments/benchmarking/simulator_adapters/` | Same adapters and method IDs; diagnostic `linear.py` model is documented, not duplicated. |
| Twelve selected `paper/common/` modules | `experiments/paper/protocol/` | Freeze order, method isolation, append semantics, budgets and seed coordinates remain. |
| Source-sensitive historical paper programs | `experiments/paper/historical_workflows/` logical owner | Native source paths remain wherever required; current dependencies remain connected. |
| Actual response reconstruction programs | `experiments/paper/response_reconstruction/` | Saved rendering and actual physical/replay operations are described separately. |
| Two analytic and two oscillator paper study folders | `experiments/paper/studies/` | Local configs, tests and TRUBA stay with the study; sealed output packages remain intact. |
| Five `paper/figures/common/` helpers | `experiments/paper/figures/` | Correct root-depth lookup, same selected arrays, style and data order. |
| Current-use evidence under `benchmarks/` and experiment `outputs/` | `results/` by family and version | Preserve native members, data, manifest status and all dependent historical lookup. |
| Unused or superseded evidence | `scientific_archive/superseded_results/` or `historical_exploration/` | Classify by present scientific use; original bytes and context remain. |
| `project/scientific_registry.toml` | `scientific_contracts/scientific_registry.toml` | Preserve complete schema/status and historical declaration lookups. |
| `project/experiment_registry.toml` | `results/experiment_registry.toml` | Preserve full history, including failed and pending records; update active workflow consumers. |
| `project/state.toml` and schema | `scientific_archive/project_records/` after active-field review | Current authority reaches the appropriate owner before originals are archived. |
| Named handoff notes and root changelog | `scientific_archive/project_records/` | Archive original text; current instructions point to current owners. |
| `docs/status/CURRENT.md`, old paper strategy | `docs/history/` | Historical snapshots keep their dates and meaning. |

`paper/` in shortened rows above is relative to `experiments/`. The linked per-file ledger gives full paths.

### Exact benchmark API split

Move `ROW_FIELDS`, `BenchmarkResultSet`, `load_results`, `summarize_results`, `export_tables` and `plot_results` into `experiments/benchmarking/results.py`. Keep `load_campaign`, `run_campaign`, `_compact_public_row`, `_markdown_report`, `_field_dictionary`, `_environment_provenance` and `_recover_method_record` with execution. Re-export the moved definitions without changing field order, `prompt5d` bindings, class identity, failure behavior or trusted legacy serialization paths. Stored-result utilities still use their actual Torch-backed and plotting dependencies; they are not promised to be dependency-free.

### Exact twelve protocol modules

`campaign.py`, `cli.py`, `cohort_v2.py`, `target_freeze_v2.py`, `supplied_targets.py`, `method_specs.py`, `protocol_v3.py`, `population_v3.py`, `validation.py`, `integrity.py`, `method_append.py`, `compact_append.py`.

The 45 existing `paper/common` files also include 24 historical workflow modules, five existing analysis aliases, the hash inventory, a lazy initializer, and two easily missed pinned sources: `cross_experiment_contract.py` and `evidence_analysis.py`. The latter two are documented under protocol and analysis ownership respectively while source constraints are resolved. None is silently discarded.

## 7. Results and archive rules in practice

Useful result families include `results/paper/graph_paper120/`, the two oscillator study families, `results/analytic_target_matching/`, `results/graph_forward/` and `results/reproducibility/`. Create actual directories when there is content; empty placeholders are unnecessary. Versioned completed results are immutable. A new producer writes a new declared version rather than overwriting accepted evidence.

Whole-package classification is essential. Repaired-SBI assessment, preparation, corrected-terminal and recovery records support current figure provenance. The failed terminal-domain held-out v1 is referenced by accepted v2. Export-recovery and qualification records can support current physical-export closure. These dependencies prevent a simplistic “successful means results, failed means archive” rule.

Before moving any sealed package, enumerate all members and its cross-package dependencies, resolve every recorded native path from each relevant consumer, preserve applicable Git attributes and verify the original digests. Keep recorded strings inside old manifests unchanged. If a versioned external resolver cannot satisfy a literal or Git-source contract, retain that original physical path and expose the result through current navigation. Do not split a historical producer out of the sealed package that records its source identity; this applies to `build_cohort_figure.py` inside the maximin output bundle.

The results guide documents current-use, supporting and superseded groups with links to their manifests.

## 8. README requirements

Every important owner needs an English guide that answers: what scientific or engineering task this area performs; what its inputs and outputs mean; how it relates to the algorithm and manuscript; which subfolders and programs are entrypoints; which operations merely inspect records and which generate, fit, train, transfer or submit; which imports and historical paths remain supported; and where independent validation evidence lives.

Do not replace this with a filename inventory. Use a short concrete example where naming is otherwise confusing. For instance, the graph model evaluates physical parameters to a response; `graph_forward` exercises that mapping; `graph_inversion` uses repeated evaluations to find parameters matching a target. Analytic forward models provide controlled target-matching test functions; an analytic GP landscape is a frozen forward function, distinct from MUTACO's learned response surrogate.

The [scientific guide](../../scientific_contracts/README.md) incorporates the supplied draft. It explains equations, tensor axes, frozen standardization, GP fitting, acquisition, terminal selection, experiment roles, all retained contract subfolders, scientific status and manuscript links. Verify its links and statements against the actual checkout before applying it. Other guides should meet the same standard at a proportionate length.

## 9. Implementation batches and completion evidence

| Batch | Deliverable | Gate before proceeding |
| --- | --- | --- |
| 0 | Verified current repository, isolated new branch, baseline path/hash/identity capture | Correct repository and base, clean isolated worktree, recovered source constraints. |
| 1 | Migration map, authority/navigation updates, portable verification entrypoints | Old/new CLI and import identity; no hidden scientific execution; pinned source bytes unchanged. |
| 2 | Stored benchmark-result split, shared protocol ownership and figure helpers | Independent saved-output behavior, public bindings, error semantics and path-root checks. |
| 3 | Physical/analytic package locations and experiment/config/study ownership | Three independent installed package interfaces, correct config discovery and historical lookup. |
| 4 | Frozen inputs, results/archive classification and active registry relocation | Complete manifest member/digest/reference verification and no change in claim classification. |
| 5 | Separately approved core moves after exact 01–18 mapping recovery | Computational AST/signature/policy identity review and appropriate qualified numerical gates within authorized scope. |
| 6 | Final coherent navigation, validation report and reviewable commits | Every source path accounted for, every exception explicit, truthful tested/untested status and no unintended source/data changes. |

Batches are implementation order, not a request for seven further design approvals. Independent work should continue if a particular pinned file must stay. Core and remaining-repository changes should remain distinguishable in commit ranges.

The [validation record](repository_validation_plan.md) gives the actual checks and bounds. The [documented-core qualification](../../results/reproducibility/documented_core_qualification_v1/README.md) identifies source `35ef82d`. The [final architecture qualification](../../results/reproducibility/repository_architecture_qualification_v2/README.md) identifies tested source `7f4ef91`, the same-environment comparison and remaining historical exceptions. The [original job record](../../results/reproducibility/repository_architecture_qualification_v1/README.md) preserves every test and scheduler outcome.


## Implemented core filenames

The original review discussion confirmed the following destinations. An old dotted import can resolve through a module alias even when the old filename is absent. Use the per-file ledger for the physical location.

| Old family | Implemented destination and scope |
|---|---|
| Response, surrogate, posterior and simulator | `response/response_standardization.py`, `surrogate/exact_multitask_gp.py`, `posterior/posterior_summary.py`, `simulator/simulator_adapter_contracts.py`. Old textual source views serve pinned readers and the held-out declaration generator. |
| Orchestration | `sequential_search/sequential_runner.py` remains one runner. Its guide follows the manuscript stages. |
| Acquisition | `confidence_geometry.py`, `acquisition_objectives.py`, `candidate_proposal.py`. Composite NEI and diagnostic aliases have installed comparator addresses. Scalar qLogNEI remains a separate experiment method. |
| Identity and randomness | `identity/canonical_identity_hashing.py`, `identity/strict_metadata_validation.py`, `identity/sequential_scientific_identity.py` and `randomness/deterministic_seed_derivation.py`. |
| Specifications, documents and writes | `parameter_and_response/parameter_response_specifications.py`, `file_management/artifact_documents_and_manifests.py`, `file_management/atomic_file_writes.py`. |
| Recommendation | `recommendation/recommendation_contracts.py` owns evaluated designs, the mean-only posterior protocol and the existing failure type. `comparators/observed_physical_mae_recommendation.py` owns observed-point MAE. Old public names and pickle bytes still resolve. |

The final architecture decision supplies these ten implementation paths. All paths in the table are relative to `src/mutaco/`. Legacy imports remain available. The checkout inventory identifies the physical addresses that still have independent file readers.

| Previous implementation | Current implementation |
| --- | --- |
| `design/initial.py` | `design/initial_design.py` |
| `policies/registry.py` | `policies/scientific_policy_registry.py` |
| `policies/surface.py` | `policies/policy_surface.py` |
| `runtime/resolution.py` | `device_and_resources/device_and_precision_resolution.py` |
| `runtime/parallelism.py` | `device_and_resources/cpu_parallelism_provenance.py` |
| `observability/monitoring.py` | `device_and_resources/resource_monitoring.py` |
| `recommendation/physical_mae.py` | `recommendation/physical_mae_terminal_recommendation.py` |
| `recommendation/_continuous.py` | `recommendation/_continuous_terminal_optimizer.py` |
| `recommendation/physical_l2.py` | `recommendation/comparators/physical_squared_l2_recommendation.py` |
| `recommendation/policies.py` | `recommendation/evaluated_design_recommendation.py` |

Decisions 16–18 on execution, results and reproducibility are implemented by the repository owners above. The root package, public API and typing marker retain their installed addresses.

## Physical locations before final root elimination

The following observations describe the earlier qualified layout. For the active layout use the [final checkout guide](legacy_checkout_paths.md).

- Current-use data under `results/` includes earlier and failed supporting packages cited by current evidence. The experiment registry retains failed and pending records. Historical project snapshots keep their original bytes. The small current Paper Lab promotion bindings keep the path strings used by the two independent bundle identities.
- Of the 1,553 reviewed evidence, catalog and registry files, 1,324 have their original Git blobs at their new physical destinations. The original compact B40 and B100 SBI source sets (2 and 10 files) include resolved native names in the repair inventory. The 217-file maximin bundle includes a producer whose repository-root calculation depends on its original depth. These 229 files retain their native locations and have current result addresses. Enumerate individual packages from the results index. A recursive glob starting at the old parent need not follow directory links.
- The graph adapter source file, declared target-98 producers, original terminal bootstrap and reconstruction modules, pure verification function owners and tensor exporter have specific source identities. Each retained source is identified in the ledger. Historical `revision:path` queries still use their original paths. The validation record identifies the five original private-input failures. With the verified compact archive restored in a disposable TRUBA checkout, both held-out inventory tests passed. The three remaining original tests still require private worker records.
- Git checkouts must materialize symbolic links. The macOS and Windows launchers check essential legacy paths before using the numerical packages. A symlinks-disabled checkout was tested as a negative case. The three independently installed distributions work outside the research checkout; the wheel build can materialize textual compatibility views while old and new imports share a single runtime module.
- Public API, legacy pickle resolution and patched globals were exercised. Python 3.11 class-source introspection through legacy `__module__` metadata cannot locate `EvaluatedDesigns`, `MeanOnlyPosterior` or `PhysicalL2RecommendationError` after extraction. Their definitions are in `recommendation_contracts.py`. The tracked historical bootstrap inspects the original optimizer function. Downstream third-party inspection and arbitrary plugin discovery remain unqualified.
- Four persisted-input figures support a new, non-overwriting versioned output command. Fig. 7 exposes a fixed manuscript destination in its source-pinned producer. Its historical surface and the archived figure remain intact.
