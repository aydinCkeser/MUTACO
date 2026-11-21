# Local scientific environment and tests

The checkout launchers need Git symbolic links for the source import roots and historical evidence addresses. On Windows, enable permission to create symbolic links (for example through Developer Mode) before cloning with `git -c core.symlinks=true clone`. A checkout that materializes links as plain text cannot resolve sealed input paths. The launcher checks representative paths before importing Python packages. Installed wheels contain ordinary Python files.

## Current scope (28 October 2026)

The accepted reference environment is TRUBA Linux x86-64 CPU. Nine readability
extractions passed full B40 target-16 and B100 target-0 strict replay and exact
saved scientific checkpoints. The latest selected scientific/import suite has
523 passes, 8 unchanged skips and zero failures, locally and on TRUBA. The NEI
precision-promotion fixture is corrected in current development tests; original
baseline failure records remain unchanged. See the [source map](../../src/mutaco/README.md)
and [v2 software gate](../../results/reproducibility/test_quality_closure_v2/README.md).

macOS is a qualified local development/test environment, not a requirement for
bitwise reproduction of the historical trajectories. Its old candidate-17
diagnosis is retained below as historical evidence. Qualified TRUBA is core-only
and lacks SBI; the local launcher requires SBI. These are different environments.

The Windows entry points remain `run_mutaco.ps1` and `run_pytest.ps1`.
The macOS entry points are `run_mutaco.sh` and `run_pytest.sh`. They require a
separate environment named `mutaco-sbi`, import MUTACO from this checkout, keep
pytest temporary data outside the repository, and preserve command arguments
and nonzero exit statuses. The macOS launcher intentionally exposes only the
scientific environment, without a development-environment bypass.

## macOS ARM64 installation

From the repository root, using an ARM64 Conda installation:

```sh
conda create --prefix "$PWD/.venv/mutaco-sbi" python=3.11.15 pip --yes --no-default-packages
".venv/mutaco-sbi/bin/python" -m pip install -r infrastructure/environments/mutaco-sbi-macos-arm64-requirements.txt
".venv/mutaco-sbi/bin/python" -m pip check
infrastructure/local/run_mutaco.sh -c 'import torch, sbi; print(torch.__version__, sbi.__version__)'
infrastructure/local/run_pytest.sh -q tests/mathematics tests/surrogate tests/orchestration
```

`.venv/` is ignored by Git. `MUTACO_SBI_PYTHON` can select an existing Python
3.11 interpreter whose environment directory is named `mutaco-sbi`.
`MUTACO_PYTEST_TEMP_ROOT` can select an external pytest temporary directory.
Repository-local temporary directories and `--basetemp` paths are rejected.

This recipe uses the existing top-level version pins, which do not freeze every
transitive dependency. The `resolved.txt` file is an original environment snapshot containing
Conda-build `file://` references; it is not a portable fresh-install requirements
file. Do not edit that historical snapshot or assume the recipe establishes a
new exact scientific environment.

### Independent installation checks (29 October 2026)

A disposable source copy and new Python 3.11.15 virtual environments were
checked on macOS ARM64. The initial offline dependency attempt stopped at a
missing BoTorch wheel, and an empty isolated Conda cache could not resolve
Python/pip. An announced public-PyPI attempt then installed all 13 pinned
scientific requirements successfully into a new venv with system site-packages
disabled. The three local packages (`mutaco`, `mutaco-analytic`, `mutaco-graph`)
also installed from source with isolated build dependencies; `pip check` passed.
Existing environments were not changed. The available CPython 3.11.15 binary
was reused; online Conda environment creation and a separate machine were not
tested.

Isolated imports resolved the core, both simulator packages and scientific
dependencies from the new environment's `site-packages`, not checkout sources
or the existing environment. The new environment passed 96 guarded safe tests,
five CLI-help routes and both portable B40/B100 verification commands from the
source copy. Scientific/scheduler/transfer execution was blocked; optional SBI
imports were blocked for help/verification. A preliminary environment reusing
old dependencies passed the same walkthrough, and the portable verification
commands also passed before Torch was installed. Research modules and evidence
remain repository-local; the three wheels do not install `experiments/`,
`reproducibility/` or benchmark datasets.
Cold Matplotlib imports may enumerate local fonts with `fc-list` or the macOS
font inventory command; that read-only discovery is allowed and is not rendering.

With `PYTHON311` set to an available CPython 3.11.15 interpreter and
`MUTACO_INSTALL_ROOT` set to a new external directory, the tested venv route is:

```sh
"$PYTHON311" -m venv "$MUTACO_INSTALL_ROOT/mutaco-sbi"
"$MUTACO_INSTALL_ROOT/mutaco-sbi/bin/python" -m pip install -r infrastructure/environments/mutaco-sbi-macos-arm64-requirements.txt
"$MUTACO_INSTALL_ROOT/mutaco-sbi/bin/python" -m pip install --no-deps . experiments/analytic_target_matching simulators/graph-simulator
"$MUTACO_INSTALL_ROOT/mutaco-sbi/bin/python" -m pip check
"$MUTACO_INSTALL_ROOT/mutaco-sbi/bin/python" -I -c 'import mutaco, mutaco_analytic, mutaco_graph'
```

Run the install commands from a disposable source copy; downloads contact the
public package index, and local builds create temporary build metadata. Set
`MUTACO_SBI_PYTHON` to the new interpreter when using checkout launchers. The
source-only research entry points require that checkout as their working root.

The following offline wheel/install route was also checked from a disposable
copy with dependencies and `setuptools>=77,<82`/wheel already available:

```sh
MUTACO_SBI_PYTHON="${MUTACO_SBI_PYTHON:-$PWD/.venv/mutaco-sbi/bin/python}"
"$MUTACO_SBI_PYTHON" -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir /tmp/mutaco-wheels . experiments/analytic_target_matching simulators/graph-simulator
"$MUTACO_SBI_PYTHON" -m pip install --no-index --no-deps --find-links /tmp/mutaco-wheels mutaco==0.1.0.dev0 mutaco-analytic==0.1.0.dev0 mutaco-graph==0.1.0.dev0
"$MUTACO_SBI_PYTHON" -I -c 'import mutaco, mutaco_analytic, mutaco_graph'
```

`--no-deps` is appropriate only after dependency qualification; it is not a way
to repair an incomplete installation. The primary pins matched, but fresh
transitive/build-tool versions differed from the historical Mac snapshot:
fonttools 4.68.0, grpcio 1.85.0, narwhals 2.28.0, tzdata 2026.6, pip 24.0 and
setuptools 79.0.1. The historical lock was not rewritten. Optional `pypdf`
(the `plot-test` extra) was not installed by the primary requirements.
These checks qualify same-Mac clean-environment installation and safe inspection;
they do not qualify a numerical reference environment, a separate fresh machine,
Windows/Linux installation, Mac historical equality or original TeX rendering.

The next acceptance step is a separately authorized final TRUBA core test run
and exact B40/B100 archived-response replay against the immutable references,
with the physical backend blocked. Include the new safe import/navigation
checks where their dependencies are supported. The qualified TRUBA runtime is
core-only: SBI-required lifecycle/training tests need a separately qualified SBI
environment; B40/B100 does not validate SBI-specific behavior. No job was
submitted during this stabilization milestone.

The primary version requirements are in
`infrastructure/environments/mutaco-sbi-macos-arm64-requirements.txt`. The resolved
requirements include all installed dependencies; the adjacent macOS JSON lock
records the platform, numerical backend, versions and reference differences.
The original `mutaco-sbi-lock.json` remains the reference and is unchanged.

All reference package versions installed successfully on 24 October 2026 with
Python 3.11.15. macOS uses Torch `2.12.1` instead of the Linux `2.12.1+cpu`
build, and Apple Accelerate instead of the original execution platform's
numerical libraries. CPU is the initial execution device; MPS is not used.
The launcher does not change RNG seeds or numerical thread counts. Historical
replay explicitly sets 20 intra-op and one inter-op thread in its own process.
Matching versions does not establish bitwise equivalence across platforms.

## Guarded current scientific/import suite

From repository root, in the existing qualified local environment, the following
uses the original test engine's selection plus the six infrastructure modules
and import-boundary check used by the latest accepted 523-pass run. It blocks
the graph physical backend for the whole session and checks attempted calls.
Keep opt-in historical-replay/diagnostic variables unset. This runs tests, not a
new physical campaign; default repository-wide pytest is a different collection.
The same selection can be checked with `--collect-only` added to pytest arguments.
It includes real GP fits. For a zero-fit inspection, collect only; the physical
guard alone does not turn those numerical tests into metadata checks.

```sh
infrastructure/local/run_mutaco.sh -c '
from contextlib import ExitStack
import pytest
from infrastructure.truba.refactor_reference import TEST_PATHS, simulator_guard
paths = TEST_PATHS + ["tests/smoke/test_boundaries.py"] + [
    "tests/infrastructure/" + name for name in (
        "test_truba_lifecycle.py", "test_readability_regression.py",
        "test_refactor_reference.py", "test_b100_regression.py",
        "test_b100_evidence.py", "test_fit_event_evidence.py")
]
with ExitStack() as stack:
    attempts = simulator_guard(stack)
    result = pytest.main(["-q", *paths])
    if attempts:
        raise SystemExit("physical backend attempt blocked")
raise SystemExit(result)
'
```

The accepted 8 skips are part of that versioned collection, not permission to
skip new failures. The original NEI failure is required only by the historical
v1 gate; corrected development tests must satisfy v2 with zero failures.

## Integrity only: no GP fitting or replay

```sh
infrastructure/local/run_mutaco.sh -m infrastructure.truba.shareable_evidence verify --directory results/reproducibility/shareable_milestone_v1
infrastructure/local/run_mutaco.sh -c 'from experiments.paper.exploration.scalar_primary_replay import verify_archive; print(verify_archive(40)); print(verify_archive(100))'
```

The first command verifies the original portable B40 payload, not every later
milestone. Each later evidence directory has its own `SHA256SUMS`; run
`shasum -a 256 -c SHA256SUMS` inside it. Exact-environment acceptance additionally
requires private original packets. Integrity verification is not a fresh
scientific replay. Cluster numerical work requires an authorized Slurm job;
the tracked lifecycle performs preparation/validation only, never submission.

## Historical 24 October qualification notes

Everything below records the pre-TRUBA investigation and its then-current
limitations. Its “blocked”, “fails” and “unchanged fixture” statements refer to
that historical revision. Current acceptance is described above; do not use
these old outcomes as the current development test gate or refactor decision.
The optional replay/diagnostic commands below execute numerical calculations;
they are not integrity-only checks and do not authorize cluster execution.

### Original numerical portability failure

`tests/acquisition/test_nei.py::test_baseline_factorization_promotes_without_jitter`
fails on the qualified ARM64 environment. Its dyadic-rational 5-by-5 matrix has
exact determinant zero and rank four. Accelerate-backed float64 Cholesky
returns failure information 5. The existing test expects successful promoted
factorization, which is sensitive to roundoff for this singular fixture.
Neither that test nor the production factorization policy has been changed.
This is a separate NEI portability issue, not evidence about MUTACO performance.
The follow-up investigation reproduced this failure on the unmodified canonical
baseline under the same environment and verified its exact singularity; see the
[diagnosis](../../experiments/paper/exploration/replay_divergence.md).

### Original historical regression attempt

The committed archives in
`results/paper/scalar_primary_trajectory_archives_v3r2/` are read without
extracting or changing their members. Both compressed hashes, both manifest
hashes, all eight member hashes and sizes, and the internal JSON sidecars pass.
The references are B40 target 16 at `/records/48` and B100 target 0 at
`/records/0`. These are independent historical campaigns.

The adapter `experiments/paper/exploration/scalar_primary_replay.py` invokes the
existing production `run_mutaco_target` with the existing graph configuration,
seed binding, Sobol generator and callback adapter. It regenerates the initial
Sobol design. It never supplies a historical candidate to the optimizer.
The production core, graph package, primary benchmark modules, selected
bindings and effective MUTACO configurations match the respective historical
source checkpoints as well as the canonical baseline.
Before returning an archived observation, it checks the candidate exactly,
the evaluation order, the effective physical simulator context, and available
acquisition decisions. A mismatch latches failure; subsequent calls cannot
consume another response. Physical simulator entry points are replaced with
tripwires. GP and transform diagnostics are explicitly reconstructed, not
historical parameter snapshots.

Run the exact gates explicitly, in their declared order:

```sh
MUTACO_RUN_GOLDEN_REPLAY=1 infrastructure/local/run_pytest.sh -q tests/reproducibility/test_scalar_primary_replay.py -k exact_historical
```

For a detailed, append-only local report from one reference:

```sh
infrastructure/local/run_mutaco.sh -m experiments.paper.exploration.scalar_primary_replay --budget 40 --target 16 --output /tmp/mutaco-b40-target16.json
```

On 24 October 2026 both references matched their 16 initial observations and
effective physical contexts, then rejected evaluation 17, the first adaptive
candidate. Maximum coordinate differences were `0.0026917457580566406` for
B40 and `0.000010311603546142578` for B100. Both first proposals used the
historical barrier/refined branch. Their raw-pool identities and proposal
configurations and raw index order match; numerical ranking scores and fitted
model identities differ.

An earlier identity discrepancy is observable after initial standardization:
the reconstructed frozen-transform hashes differ from the archived hashes.
Transform identities include fitted observation identifiers and numerical
mean/scale values. Follow-up calculations recover both complete historical
transform hashes by changing only the float32 means to an alternative reduction
order, with identical scales and other semantic fields. Standardized numerical
data therefore differ, rather than just checksum representation. The historical
means are hash-verified reconstructions, not directly stored arrays. A separate
fit using those recovered means still differs at candidate 17. No complete
numerical root cause is established.

Both exact pytest gates fail; neither is marked xfail and no tolerance is
relaxed. Both stop after consuming 16 archived observations, preserve ambient
Python/NumPy/Torch RNG state, and make zero physical simulator calls. Duplicate
recovery later in the trajectories and terminal recommendation have therefore
not been reached by a successful replay. The core refactor remains gated.

The relevant suite (mathematics, surrogate, acquisition, orchestration, unit,
policy surface, local launchers, reference integrity, scalar-primary protocol,
and B100 archival/preparation checks) reports **183 passed, 10 skipped, 1
failed**. Skips include seven unavailable-CUDA checks, the two
explicit exact-replay gates, and the legacy expanded-fetch archive check.
The new archive/guard and launcher subset reports **14 passed, 2 skipped**;
the exact gates were additionally executed and failed as described above.
`infrastructure/environments/mutaco-sbi-macos-arm64-qualification.json` records the
compact outcomes and first divergences.

The [follow-up diagnosis](../../experiments/paper/exploration/replay_divergence.md)
records historical-source comparisons, eight deterministic local replays,
first-fit/proposal captures, a mean-only sensitivity experiment, and the
smallest useful Linux x86-64 reference packet. The compact evidence is
`results/reproducibility/shareable_milestone_v1/macos-diagnostic-summary.json`. Decision D
keeps the refactor blocked pending that reference. Production numerics, exact
equality and candidate selection remain unchanged.

Capture a bounded first-fit packet outside the checkout:

```sh
infrastructure/local/run_mutaco.sh -m experiments.paper.exploration.replay_divergence --budget 40 --target 16 --output /tmp/mutaco-b40-diagnostic.json
MUTACO_RUN_REPLAY_DIAGNOSTICS=1 infrastructure/local/run_pytest.sh -q tests/reproducibility/test_replay_divergence.py
```

The capture command's successful exit means the diagnostic packet was written;
it does not certify historical equality. Use the original exact gates for that.

The same Python diagnostic supports Linux x86-64. Its `--format reference`
option writes compact deterministic `.json.gz` with exact little-endian tensor
bytes. Follow the [Linux reference-capture procedure](../../experiments/paper/exploration/linux_reference_capture.md)
for source transfer, the pinned environment preflight, exact B40 command and
artifact checksums. This preparation does not establish cross-platform equality.
