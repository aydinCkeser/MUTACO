# Checkout paths and historical verification

Current programs read [results](../../results/README.md), [frozen graph targets](../../experiments/graph_inversion/frozen_targets/paper120_v3), [scientific contracts](../../scientific_contracts/README.md) and [environment inputs](../../infrastructure/environments/). Install the three distributions from the repository root, `experiments/analytic_target_matching` and `simulators/graph-simulator` using the [environment guide](../../infrastructure/local/README.md). Their Python imports remain `mutaco`, `mutaco_analytic` and `mutaco_graph`.

## Removed roots

The starting commit was `16d5e5c4787c29c5daf176881f57ba8d39468530`. Current callers open canonical paths directly. When an immutable declaration supplies an old filename, the [recorded-path reader](../../reproducibility/recorded_paths.py) maps it to its reviewed owner for I/O. Identity calculations retain the string from the declaration. A directory whose former members have different owners must be addressed by its specific package or member.

| Former root | Regular files + links before | Where its contents are used now |
|---|---:|---|
| `benchmarks/` | 12 + 26 | Twelve original compact SBI files are regular files in [the graph result collection](../../results/paper/graph_paper120/). Experiment and verification readers use the versioned packages under `results/`. |
| `configs/` | 0 + 3 | Scheduler and bootstrap inputs live in `infrastructure/environments/`; the plotting style lives with its experiment. Stored figure provenance keeps the original style identifier. |
| `data/` | 0 + 1 | The graph experiment owns the 120 frozen targets, manifest and sidecars. A historical input can still identify the same cohort by its original string. |
| `packages/` | 0 + 5 | Bootstrap and deployment scripts use the two simulator distribution roots listed above. |
| `scientific/` | 0 + 1 | Graph parity reads the retained oracle from [its contract](../../scientific_contracts/graph_oscillator/reference_fixtures/retained_cpu_small_v1.json). |
| `tools/` | 5 + 0 | Use `python -m reproducibility.verify`. The saved-file [API](../../reproducibility/verification.py) and [tensor exporter](../../reproducibility/tensor_export.py) live with the reproducibility tools. The configured research launcher also resolves the old dotted API and CLI. The physical script paths are available at their original commits. |
| `project/` | 0 + 0 | The [current authorities](../DOCUMENTATION_AUTHORITY.md) and archival coordination records have separate owners. |

In `experiments/paper/common/`, 22 forwarding files and the old hash-inventory link were removed. The 21 remaining physical entries include the sources used by historical producers and their callers. The retired forwarding files share module objects with the active analysis, protocol and historical-workflow implementations. Package-scoped aliases preserve imports, pickle globals and command dispatch. Source-bound producers remain identifiable at their own paths and revisions.

The [previous inventory](legacy_checkout_paths.tsv) applies to the cleanup tested at `67eda626`. The [per-file migration ledger](repository_migration_manifest.json) records earlier source identities. Those snapshots retain their original bytes.

## Verify a source-bound program

Use a separate clone for original producer paths, artifact identity strings, or historical CLI commands. Choose the commit in the scientific record. The following example exercises the original saved-evidence CLI at the last pre-migration scientific-documentation revision. It reads tracked evidence and performs no simulator calls. Run it from the current repository with a Python 3.11 interpreter.

```sh
scratch=$(mktemp -d)
git clone --quiet --local --no-checkout . "$scratch/historical"
git -C "$scratch/historical" checkout --quiet --detach 9f3ce1893112bb37c23e3aad1508564f55d2c81b
(
  cd "$scratch/historical" || exit 1
  git rev-parse HEAD
  python3 -B -m tools.reproducibility.verify b40 --directory benchmarks/reproducibility/shareable_milestone_v1
  git status --short
)
```

At `9f3ce189` the two original terminal preparation tests verify the stored paired inventory. For comparison with all six original producer-file hashes, use the detached snapshot `3c74fba15e5d6a22ba8e202df032ba3180510f0a`. The recorded jobs used a dirty source tree; the manifest preserves its original commit and status. The later snapshot supplies verifiable source bytes and does not recreate that dirty worktree identity.

For the historical B40 held-out v1/v2 preparations, the original tests also need the decompressed compact inventory. In the `9f3ce189` checkout, first verify the SHA-256 of `scalar-primary-b40-v3r2.tar.gz` against `2f86646f7c21914552d10e9e1fcb209d29880cee194a3e79f8f8fae211b6659c` and of its manifest against `afd4df364ba56e5fa357193661d68fa03249d29bdf4d004008f016ef002553a2`. Both are under `benchmarks/paper/scalar_primary_trajectory_archives_v3r2/b40-v3r2/`. Restore the four manifest members in the ignored directory `experiments/paper/exploration/outputs/scalar_primary_smoke/population_b40_v3r2_compact_fetch/population/` inside the disposable clone. Check each size and SHA-256 against the manifest, then run `tests/experiments/test_terminal_domain_heldout_assessment.py` and `test_terminal_domain_heldout_assessment_v2.py` through that checkout's pytest launcher. These checks use saved inputs and retain the original raw-file requirements. The [archive guide](../../results/paper/scalar_primary_trajectory_archives_v3r2/README.md) lists both budgets and their limits.

## Qualified TRUBA check

Qualified source `7a1652d924328d19fd7346439cc39ef56160885e` ran on TRUBA Linux x86-64 in Slurm job `7012390`. The CPU job completed with exit `0:0` in 00:09:51 using a 20-CPU allocation, a 20 GiB memory request and a 20-minute limit. Commands ran in a clean exact-source checkout. The independent verdict used the six original reference inputs, original scientific commit `aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3`, and ordered test identities from the accepted selection.

| Check | Result |
| --- | --- |
| Accepted scientific suite | 523 passed, 8 expected skips, zero failures and zero deselections. All 531 identities, their order, outcomes and skip reasons matched the reference. |
| B40 target 16 | 40/40 archived observations and 50 GP fit-progress events. History, checkpoint and event comparisons were exact. |
| B100 target 0 | 100/100 archived observations and 170 GP fit-progress events. History, checkpoint and event comparisons were exact. |
| Public packages and paths | Built and independently installed all three wheels. Old-first and new-first imports agreed for ten module pairs and ten pickle globals per order. Path, CLI, scheduler and provenance selection: 346 passed, 6 platform skips (macOS launchers and real CUDA allocation). |
| Historical programs | All 52 tests in the separate `9f3ce189` checkout passed, including the saved-evidence CLI and paired B40 preparation. |
| CPU forward models | 14 analytic tests passed, one CUDA-only test skipped. The 16 current and 16 retained graph cases passed NumPy–Torch parity, as did the direct frozen-fixture check (65 graph kernel evaluations). Physical backend attempts and calls during the guarded scientific selection and replays were zero. |
| Preserved material | 810 evidence checksums, eight archive members, 242 frozen cohort files, 24 manuscript entries and both sets of 47 source hashes verified. The 12 relocated SBI files retain their original Git blob IDs. |

The first candidate job (`7012233`) was cancelled before scientific replay. A local preparation test found that hashing the relocated scheduler profile changed both the B100 historical profile hash and the derived package identity. Historical verification now reads the original profile at `649235bcb7e8773f5f4b0425ad111314d7d3b1f9`, verifies it against its receipt, and checks the one reviewed path change against the current profile. Fresh preparations record their actual profile hash. The sealed manifests and scientific references retain their original bytes.

The [previous path qualification](https://github.com/aydinCkeser/MUTACO/blob/16d5e5c4787c29c5daf176881f57ba8d39468530/docs/decisions/legacy_checkout_paths.md#qualified-truba-check) records job `7012034` at `67eda626`. The [architecture qualification](../../results/reproducibility/repository_architecture_qualification_v2/README.md) records the earlier private-input and figure reproduction limits.
