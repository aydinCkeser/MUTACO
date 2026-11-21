# Repository migration validation

This document retains the acceptance criteria and ends with the results obtained on the local migration branch.

## Baseline and scope

The audit source is `aydinCkeser/MUTACO`, `ege/repro-refactor`, `9f3ce1893112bb37c23e3aad1508564f55d2c81b`. Verify the actual source before choosing the new branch base. Capture each moved file's current bytes, applicable Git text attributes, public references and source-bound artifact identities. Compare the full live tracked inventory with the 2,555 audited paths.

Tests must respect the repository's external temporary-output policy. Inspect test bodies and existing launchers before selecting tests; a filename containing “test” or “offline” alone does not prove absence of a fit, physical evaluation or transfer.

## Acceptance by change

| Change | Relevant existing anchors | Required comparison |
| --- | --- | --- |
| Navigation and ownership | `tests/contracts/test_repository_navigation.py`, `test_clean_break_handoff.py` | Active links point to correct owners. Historical notes retain original status. Adapt only mutable current-source path assertions. |
| Public packages | `tests/smoke/test_boundaries.py`, `tests/experiments/test_graph_entrypoint_usability.py`, `test_shared_contract_imports.py` | Core, graph and analytic wheels install separately outside checkout; optional dependencies remain optional; old/new objects and safe help behave correctly. |
| Saved verification | `tests/reproducibility/test_verification_entrypoints.py`, `test_verification_characterization.py`, `test_shareable_evidence.py` | Exact saved summaries, corruption/missing-file failures, stdout/stderr/exit behavior, and original underlying functions. Six pinned verifier sources retain hashes. |
| Tensor export source exception | `tests/infrastructure/test_tensor_evidence_export.py` | Source remains pinned; no export during ordinary verification. If explicitly testing the writer, only declared external temporary outputs are used. |
| Benchmark stored results | `tests/experiments/test_benchmarking.py` and independent saved records | Same six moved public definitions, row/field/group ordering, summaries, exports, data selected for plots and malformed-input behavior. No method dispatcher initialization via view imports. |
| Shared paper protocol | `tests/experiments/test_paper_protocol_v3.py`, shared-contract import checks, workflow tests | Freeze-before-assessment, method isolation, append-only receipt semantics, seed coordinates, identity vocabulary and exact call caps. |
| Saved analysis and figures | `test_paper_analysis_characterization.py`, `test_paper_analysis_compatibility.py`, `test_offline_figure_compatibility.py`, `test_offline_figure_renderers.py` | Independent archived input summaries and selected plot arrays match. Render only to external temporary outputs; do not overwrite manuscript or archived figures. |
| Historical source loaders | `test_terminal_lineage_forensics.py`, `test_repaired_sbi_terminal_preparation.py`, `test_repaired_sbi_terminal_archive.py` | Native revision:path and source-origin checks stay meaningful. A zero-recommendation loader probe does not certify terminal generation. |
| Infrastructure bindings | `tests/infrastructure/test_truba_lifecycle.py`, `test_readability_regression.py`, `test_refactor_reference.py` | Mocked transport, clean-source and unsafe-path refusal, profiles and source guard behavior; no actual scheduler submission. |
| Core relocation | Existing acquisition, mathematics, orchestration and surrogate tests chosen by actual changed definitions | Computational AST and signatures; class/exception identities; mutable settings/rebinding; policy/seed identities and ambient RNG. Numerical checks use their qualified scope. |
| Optional SBI boundaries | Existing `test_sbi_zoo_p1_execution.py`, `test_sbi_zoo_p2_execution.py`, `test_sbi_zoo_p3_execution.py` | Use declared fake/mocked cases first. P1/P3 physical execution and P2 real training remain separately scoped. |
| Oscillator study relocation | Existing local tests under both studies | Same config/schema discovery and local paths; no change to target-98 vs maximin-12 scientific meanings. |

Use the [migration ledger](repository_migration_manifest.json) for current addresses. Anchors above are a selection guide, not evidence that every case is safe or required for every move.

## Compatibility checks that file moves often miss

Exercise fresh old-first and new-first import orders, package attributes, lazy facade behavior and shared module globals. Check trusted legacy pickle references without loading unknown/untrusted artifacts. Verify old CLI `-m`, direct-file invocation, `runpy`, positional arguments, safe help and failure codes where supported. A standalone stdlib hash-inventory script must continue to work from outside checkout.

Inspect root discovery through `Path(__file__).parents[n]`, dynamic import strings, `spec_from_file_location`, `sys.modules` overlays, `inspect.getsourcefile`, shell working-directory assumptions, scheduler `-m` tokens, relative sidecar paths and TeX inputs. Do not treat an import shim as proof of exact historical filesystem identity.

For package builds, ensure the root distribution includes only the installed core. Build graph and analytic distributions independently. In a fresh environment and outside repository/PYTHONPATH, verify `mutaco`, `mutaco_graph` and `mutaco_analytic` using their supported installation boundaries. Do not install arbitrary latest dependencies as a substitute for the recorded qualified environment.

## Evidence integrity checks

Retain original native Git paths when querying original commits. Preserve archive member order, checksums, source-hash pins, scientific metadata and path-bearing declared identities. A moved current registry must not rewrite the sealed declarations that cite its old path. Verify every identified consumer through a versioned resolution strategy or document an exact-path exception.

Preserve the 242 frozen target members, the 978-file paired terminal-domain v2 ledger, and all 260 experiment-side output members while classifying their current use. These counts describe the audited source; reconcile drift before applying them to a newer checkout. Keep the source producer inside a sealed output package when its hash forms part of that package.

Check `.gitattributes` at both sides of a move. The benchmark store has a more specific text-conversion rule; changing the owning folder must not normalize native evidence bytes.

## Independent fixtures and scientific gates

Do not modify these fixed characterization fixtures to fit new behavior:

- `offline_figure_rendering_before_2b.json`
- `offline_verification_before_2c.json`
- `shared_contracts_before_2d.json`

Existing test fixtures materialize archived members outside checkout. Preserve their external origin and member hashes rather than expanding or regenerating new expected records in the repository.

Where the separately approved scientific scope permits it, the strict archived-response reference uses B40 target 16 and B100 target 0, with full sequential trajectories and checkpoints in the same qualified environment. Compare actual proposed parameters and effective contexts before returning any stored response; never substitute archived proposed candidates to make replay match. Compare identities, ordered observations, GP state, duplicate recovery, terminal policy, events and physical-call accounting. Stop at the first difference without loosening tolerance. This replay performs real GP computation and is not a saved-file-only check.

Graph physics and CUDA parity require their own environment and authorization. SBI-specific execution cannot be certified by an unrelated core replay. Preserve the recorded same-environment acceptance rule instead of inventing a universal cross-platform bitwise-equality demand.

## Report outcomes without reusing historical qualification

The audit refers to 531 historical accepted node identities and a prior result of 523 passed, 8 skipped at `35ef82d`. Those are earlier evidence. They are not a result for `9f3ce189`, this plan or any new branch.

Record each real validation command, actual source SHA, environment, selected cases, outcome, output location and computation boundary. Distinguish static checks, import/build checks, saved-data checks, real numerical replay and physical/optional-SBI qualification. Report concrete unresolved failures and unrun checks. Stop optional testing after the relevant risk is resolved; do not broaden into scientific execution simply to produce a larger passing count.

## Local implementation results

These checks used the macOS CPU environment and existing pinned dependencies. They did not install new numerical dependencies or submit work to a cluster. Counts in different rows can overlap.

| Check | Result | Scope |
|---|---|---|
| Source and result inventory | All 2,555 prepared paths found in the audited Git tree. All 1,553 evidence/catalog/registry blobs, executable modes and applicable text/eol/diff/merge attributes verified. | Hashes bind the base to the actual physical file at the reviewed revision, including native exceptions. |
| Scientific records | 810 repository checksum entries, 255 embedded path/digest declarations, both compressed trajectory archives and all eight members verified. All 978 terminal ledger members preserved. | Checks saved bytes, order and member sizes. The recorded 47 qualified source digests verified at `35ef82d`. |
| Frozen cohort and declarations | 242 cohort files, 121 sidecars and 240 original revision/path lookups; 88 existing graph declaration addresses preserved. | The original target-parameter ledger is present at its recorded revision. |
| Installed packages | Three wheels built offline and imported outside the checkout. Core aliases also tested under both old/new first-import orders after relocation. | An isolated installation reused existing macOS numerical dependencies. Native Windows and qualified Linux installations were not run. |
| Experiment contracts, stored results and reports | Same 56 selected tests passed on the base and migrated tree. 79 offline analysis/rendering cases passed. Stored result selection, schemas, pickle data and exports were checked against pre-migration captures. | New and old CLI/help and mutable bindings checked. Original characterization fixtures retained. |
| Saved-data verification | Same 34 checks passed before and after. Four additional compatibility cases passed. B40/B100 verifier stdout matched baseline bytes. | Six original verifier sources plus the pinned tensor exporter retain source identity. No replay. |
| Acquisition | 214 passed, one CUDA skip, two real GP-fitting cases excluded in each run. Seven fresh compatibility cases passed. 151 deterministic fixed-fixture captures matched. | Protects ranking, comparator and patched import bindings on local fixtures. |
| Core stages, identities and utilities | Nine stage/identity/seed source bodies byte-identical; 82 original definition pickles matched across the five stages. Fifteen identity/seed pickle bindings and three fixed seed coordinates matched. Three storage/specification moves retained 13 signatures and 12 captured pickle hashes. | Fresh process and installed wheel, including old module patch interception and ambient RNG. |
| Recommendation | 13 definition ASTs, eight public signature/pickle captures and nine selected pre/post tests matched. The new identity/ownership selection recorded 31 passes. | Controlled posterior/observed-data cases; no GP fit. |
| Versioned figure output | 65 relevant passes including 14 new cases. Eight temporary PDF/PNG hashes matched pre-migration fixtures using the recorded test backend/fonts. | Historical TeX rendering remains unqualified. Nothing was rendered into the tracked result tree. |
| Scientific test discovery | Collected all 531 original test identities in the same order. Each complete node ID matched its independently recorded SHA-256. Four display labels in the portable record intentionally redact path parameters. | Collection only. The historical 523-pass/8-skip result still belongs to its original source. |

For a local integration check use the guarded launcher from the [local guide](../../infrastructure/local/README.md). The reviewed local selection included the repository navigation, handoff, curated data, authority, result ownership, core import paths, comparator ownership, experiment owner/import compatibility, verifier entrypoints and characterization, TRUBA lifecycle (mocked), refactor-reference/readability infrastructure and import-boundary tests. The results ledger and the frozen target manifest can also be inspected independently using only Git and SHA-256.

### Inherited missing inputs

The same failures were reproduced at the audited base. Keep these checks and their expected values intact.

| Test | Required saved input |
|---|---|
| `test_repaired_sbi_terminal_preparation.py::test_terminal_preparation_package_and_declaration_are_reproducible` | Complete raw B40 worker archive. |
| `test_repaired_sbi_terminal_archive.py::test_corrected_terminal_archive_reproduces_from_append_only_attempts` | Historical attempted corrected-terminal generations. |
| `test_terminal_domain_heldout_assessment.py::test_zero_call_preparation_has_complete_paired_inventory` and the corresponding `_v2.py` test | Expanded B40 compact export at its declared workspace address. A separate verified compact fixture reproduced all 240 original task rows for each version. |
| Oscillator `test_analysis.py::test_zero_call_analysis_uses_only_frozen_response` | Hash-bound target-98 worker-18 method record outside the curated checkout. |

The historical terminal selection reported 26 passed, one skip and four missing-input failures. The extended oscillator/ledger selection reported 50 passed and one missing-input failure. No fixture, tolerance or expected quantity was updated to change these outcomes. Scientific replay, physical model qualification, optional SBI training, Windows/Linux installation and historical TeX figure rendering remain separate checks.

### Integrated checkout

After the recommendation and figure batches, a single guarded macOS run passed **193 tests** with two upstream Torch deprecation warnings. The selection described above also included the recommendation ownership and versioned figure-output tests. All 506 tracked Python source addresses compiled in memory. The 24 entries under `docs/paper/` and all three independent characterization fixtures matched the original Git trees, including modes and blob identities. The 531 scientific test identities were collected separately and checked against their ordered SHA-256 list.

The tests use fresh processes, external temporary directories and independently recorded inputs where applicable. With Git symbolic links disabled, a legacy evidence address fails to resolve. The checkout launchers detect missing required compatibility paths before importing scientific packages. No scientific reference was regenerated.

## Qualification of the implemented architecture

The [TRUBA architecture qualification](../../results/reproducibility/repository_architecture_qualification_v1/README.md) records the subsequent execution on scientific source `7f4ef91e34fefb00fd5f1f8779f1c69641afb424`. The [per-file ledger](repository_migration_manifest.json) now verifies 2,555 paths at their actual destinations, including all ten final filename decisions. The original macOS and historical outcomes above retain their original scopes. On TRUBA the two B40 held-out inventory tests passed after the original compact archive was expanded. The report gives the remaining missing inputs and experimental differences.

The [final qualification decision](../../results/reproducibility/repository_architecture_qualification_v2/README.md) applies the revised private-branch publication criterion. It records the original-versus-refactored check under identical TRUBA settings, the bounded historical search and the effects of the remaining figure differences on the manuscript. The original test and scheduler outcomes remain available in the first report.
