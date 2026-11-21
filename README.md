# MUTACO

MUTACO solves a budget-limited inverse problem: given a desired vector of
responses, choose physical parameters whose simulator responses match it.
It models the response vector with a correlated multitask Gaussian process,
then selects experiments through compatibility screening and sequential
acquisition. The reusable core is Torch-native and simulator-independent;
analytic and graph-oscillator forward models are separate packages.

The release contains source code, frozen inputs, compact results, and verification evidence.
Read the [MUTACO preprint v2](https://arxiv.org/abs/2610.04619v2) for the scientific study.

## Install and run

Use Python 3.11 or 3.12. The documented macOS ARM64 scientific environment uses
Python 3.11. On Windows, enable symbolic-link creation (for example, through
Developer Mode) **before** cloning from Git Bash or PowerShell:

```sh
git -c core.symlinks=true clone https://github.com/aydinCkeser/MUTACO.git
cd MUTACO
```

In an activated Python environment, install the core and simulator packages
from the checkout root:

```sh
python -m pip install . ./experiments/analytic_target_matching ./simulators/graph-simulator
```

Use the [local environment guide](infrastructure/local/README.md) for the
qualified scientific stack, SBI workflows, and plotting dependencies. The
handwritten analytic example runs a small simulation-based experiment:

```sh
python -m experiments.analytic_target_matching.run_handwritten --config experiments/analytic_target_matching/configs/handwritten.json --output runs/analytic_target_matching/handwritten --device cpu
```

For read-only checks, follow [Data availability](DATA_AVAILABILITY.md)
for the public trajectory and portable B40/B100 commands. The contributors have
agreed to release the original software under [BSD-3-Clause](LICENSE) and the
designated datasets under [CC BY 4.0](DATA_LICENSE.md). [CITATION.cff](CITATION.cff)
credits Ege Karadeniz and Aydın Cem Keser for the software and gives the preprint
citation.

## Start here

| You want to… | Read |
| --- | --- |
| Follow the algorithm alongside the paper | [Core navigation and authoritative paper-to-code map](src/mutaco/README.md) |
| Find a research program, configuration and execution boundary | [Experiment catalog](experiments/README.md) |
| Inspect results, data, figures and verification records | [Scientific evidence index](results/README.md) |
| Check the current paper's permitted claims | [Paper scope](scientific_contracts/paper_scope/README.md) |
| Use the qualified local environment and guarded tests | [Local guide](infrastructure/local/README.md) |
| Understand exact scientific regression | [Validation contract](scientific_archive/validation_history/refactor_validation/README.md) |

## Scientific execution

The accepted graph profiles select options over one implementation. In
[run_sequential](src/mutaco/sequential_search/sequential_runner.py):

1. Validate capabilities/options and establish named seeds.
2. Generate scrambled Sobol points and collect initial observations (16 in
   the accepted B40/B100 profiles).
3. Freeze response means/scales from the initial observations and standardize.
4. Assemble ordered training tensors and fit the correlated exact multitask GP.
5. Score and screen raw candidates, choose the compatibility-barrier or
   uncertainty-exploration branch, refine and select a non-worsening candidate.
6. Recover duplicates where necessary; evaluate, standardize and commit the
   observation; update the GP until the budget is exhausted.
7. Recommend a continuous minimizer of posterior-mean physical MAE for these
   profiles. Fresh held-out physical assessment is a separate experiment.

The [source map](src/mutaco/README.md) identifies each function. Production GP
fitting includes registered priors (MAP-type fitting); the historical reference
profiles retain their 12-iteration limit. No prior-free MML ablation or numerical
policy correction is included in the readability work.

## Repository navigation

| Location | Responsibility |
| --- | --- |
| [src/mutaco](src/mutaco/README.md) | Scientific algorithm, GP, acquisition, transforms, randomness and typed interfaces |
| [simulators](simulators/README.md) | Physical graph-oscillator model and independent reference. The [analytic forward models](experiments/analytic_target_matching/forward_models/) live with their experiments. |
| [experiments](experiments/README.md) | Research protocols, workers, assessment, analysis and rendering |
| [frozen graph targets](experiments/graph_inversion/frozen_targets/paper120_v3) | Frozen ordered 120-target inputs |
| [results](results/README.md) | Compact results, trajectory archives and engineering regression evidence |
| [scientific_contracts](scientific_contracts/README.md) | Mathematical/protocol contracts and claim boundaries |
| [documentation authority](docs/DOCUMENTATION_AUTHORITY.md) | Active registries, paper claim authority and dated project records |
| [infrastructure](infrastructure/README.md) | Local launchers and explicit TRUBA source/validation lifecycle |
| [reproducibility](reproducibility/README.md) | Read-only checks of saved hashes, numerical packets, events and test evidence; no scientific execution |
| [docs](docs/README.md) | Manuscript, result reports and figure bundles |
| [scientific_archive](scientific_archive/README.md) | Superseded comparisons, dated validation reports and original project coordination |
| [migration map](docs/decisions/repository_migration_map.md) | Current owners and scientific validation. The [checkout inventory](docs/decisions/legacy_checkout_paths.md) lists the seven retired roots and how to verify source-bound programs in a separate checkout. |
| [tests](tests) | Mathematical, orchestration, protocol, import and infrastructure checks |

## Existing scientific results

Start at the [evidence index](results/README.md). The current primary
comparison is MUTACO versus scalar qLogNEI and Sobol at independently executed
B40/B100 budgets. Repaired SBI assessments and the controlled B40 terminal-domain
study have their own claim boundaries. Older MUTACO/NPE endpoints remain
historical provenance; the [paper scope](scientific_contracts/paper_scope/README.md)
explains which results may support the paper.

Nine readability extractions passed strict B40 target-16 and B100 target-0
archived-response replay and exact saved scientific comparisons on TRUBA.
These two trajectories protect code behavior; they are not a population
performance comparison. The documented-core qualification recorded **523 passed,
8 skipped, 0 failed** in the [documented-core record](results/reproducibility/documented_core_qualification_v1/README.md).

**The repository architecture and core scientific workflow are qualified on TRUBA at `7f4ef91`.** The 531-case selection had 523 passes and eight expected skips. B40 target 16 and B100 target 0 matched their complete historical trajectories. Pre-migration and refactored sources also agreed on three B100 targets under the same 20-thread profile. Some original worker records are unavailable, and several paper figures still differ at the byte or pixel level. The [final qualification report](results/reproducibility/repository_architecture_qualification_v2/README.md) gives the scientific effect of those differences and the scope of GPU, simulator and SBI coverage. The [migration checks](docs/decisions/repository_validation_plan.md) cover import, packaging and data compatibility.

The checkout with all seven legacy roots removed was qualified at `7a1652d92432` in TRUBA job `7012390`. The same 523 scientific tests passed with the same eight skips. Both complete trajectories and their GP checkpoints matched the independent references exactly. The [checkout and qualification guide](docs/decisions/legacy_checkout_paths.md#qualified-truba-check) records the package, path, simulator and historical checks for this revision.

## Safe local checks

Use an existing qualified environment through the approved launchers. From
repository root, on macOS:

```sh
infrastructure/local/run_pytest.sh -q tests/smoke/test_boundaries.py tests/experiments/test_exploration_entrypoint_safety.py::test_documented_cli_help_has_no_physical_calls tests/contracts/test_repository_navigation.py
```

On Windows, use the pinned `mutaco-sbi` launcher:

```powershell
.\infrastructure\local\run_pytest.ps1 -q tests/smoke/test_boundaries.py tests/experiments/test_exploration_entrypoint_safety.py::test_documented_cli_help_has_no_physical_calls tests/contracts/test_repository_navigation.py
```

These checks use isolated physical-call tripwires where relevant. The larger
[guarded scientific/import selection](infrastructure/local/README.md#guarded-current-scientificimport-suite)
uses a session-wide physical guard. Plain repository-wide pytest is a different
collection and includes real forward-model workflows. Test temporary output must
stay outside the checkout.

**Plotting is not necessarily offline.** Read the
[entry-point safety table](experiments/paper/exploration/README.md#execution-boundaries).
Importing the legacy notebook-style plot no longer executes its body; explicitly
running it still requests physical reconstruction. Do not import every research
module indiscriminately or run a worker to learn its interface. Generic benchmark
execution and some functions named `replay` also invoke simulators.

## Environment and reproduction limits

The scientific reference is qualified TRUBA Linux x86-64 CPU: Python 3.11.17,
Torch 2.12.1+cpu, BoTorch 0.18.1 and GPyTorch 1.15.2. Full package, backend and
thread metadata is in the [documented-core manifest](results/reproducibility/documented_core_qualification_v1/manifest.json).
That core-only runtime lacks SBI. The separate macOS ARM64 Python 3.11
`mutaco-sbi` environment supports local development through shell launchers.
A separate TRUBA environment exercised the specified SBI workflows. CUDA qualification covers the tested graph and scientific placement cases on two Tesla V100s. Windows and MPS numerical equivalence, arbitrary dependency combinations, independent fresh-machine installation and cross-platform bitwise reproduction remain unverified.

Use the [existing installation recipe and limitations](infrastructure/local/README.md#macos-arm64-installation).
Dependency snapshots are not all portable installers. Exact-environment
regression additionally requires privately preserved original reference packets;
portable derived summaries do not replace them. Numerical replay belongs in an
authorized qualified compute allocation, not a cluster login node. TRUBA
[prepare/validate](infrastructure/truba/README.md) does not authorize submission.
Historical scheduler runs and full-packet comparisons require the original private records and a qualified environment.
