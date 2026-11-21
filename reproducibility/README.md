# Verifying saved scientific evidence

Use this directory to answer whether **two already-recorded results satisfy
their original comparison contract**. A stored test verdict, a trajectory
replay, and a physical assessment have different meanings. Reading these
records can establish their hashes, equality and declared provenance. It
does not qualify a new machine, rerun a fit or reevaluate a simulator.

From the repository root, the local launcher can inspect the canonical
result packages without private inputs:

```sh
infrastructure/local/run_mutaco.sh -m reproducibility.verify --help
infrastructure/local/run_mutaco.sh -m reproducibility.verify b40 --directory results/reproducibility/shareable_milestone_v1
infrastructure/local/run_mutaco.sh -m reproducibility.verify b100 --directory results/reproducibility/b100_regression_v1
```

On Windows, use `infrastructure/local/run_mutaco.ps1 -- -m` with the same module
and arguments. The API and CLI use the standard library. They accept explicit
input locations and write their result to stdout.

| Command | Inputs | Interpretation |
| --- | --- | --- |
| `b40` or `b100 --directory DIR` | A portable evidence package | Verifies the mathematical digests, checksum entries and sidecars. Private original bytes require separate access. |
| `history --budget 40\|100 --first A --second B` | Two saved histories from the specified profile | Compares scientific history, accounting and RNG under the original exact contract. |
| `checkpoints --first A --second B` | Two saved checkpoint packets | Checks internal digests, scientific values, environment and backend-attempt records. Complete packets may be private. |
| `fit-events --first HISTORY --second EVENTS` | Saved fits and an observed event trace | Checks order and events against expectations reconstructed from the fit records. |
| `tests --source-root SRC --references REFS --input-manifest SPEC --output-root REPORTS` | Checksum-bound source/input revision and previously written JUnit/status files | Validates the test expectations appropriate to that revision, including the historical NEI fixture and the later zero-failure gate. Despite its historical name, `--output-root` is read here. |
| `python -m reproducibility.hash_pair_inventory DIR` | A tree of JSON records and corresponding `.sha256` sidecars | Prints sorted inventory, category counts, digest, mismatches, missing and orphan sidecars. Read the mismatch lists: reporting an inconsistency does not itself make this command exit nonzero. |

[verification.py](verification.py) imports the original verified functions.
Their definitions remain in [infrastructure/truba](../infrastructure/truba/README.md)
because the archived sources fingerprint six specific files: `refactor_reference.py`,
`b100_regression.py`, `readability_regression.py`, `fit_event_evidence.py`,
`shareable_evidence.py` and `b100_evidence.py`. [verify.py](verify.py)
exposes only the saved-file operations. Historical hashes apply at their original
commits. The current bootstrap paths in the first three modules point to their
new owners; the tests compare their source against the pinned bytes with only
those path literals substituted. The full-packet comparison retains
its original stdout ordering. Failed scientific/test comparisons exit 2;
missing/malformed inputs retain the underlying error behavior.

Use [verification.py](verification.py) for the saved-file Python API,
`python -m reproducibility.verify` for the CLI, and
[hash_pair_inventory.py](hash_pair_inventory.py) for the standalone
standard-library inventory. The [tensor exporter](tensor_export.py) preserves
the original exporter bytes and writes only when explicitly called.

Both the former `tools.reproducibility` API and the current
`reproducibility` API are checkout tools. The installed `mutaco`,
`mutaco_graph` and `mutaco_analytic` packages have separate public APIs.
Current scripts should import `reproducibility.verification`. Research
launchers include the repository and its `src/` on the Python path. In that
environment, the three legacy `tools.reproducibility` modules still provide
the same verifier and exporter objects, trusted pickle names and verifier CLI.
The standalone wheels contain no top-level `tools` namespace.

### Public trajectory archives

Run the read-only checks from the repository root:

    python -m reproducibility.public_trajectories 40
    python -m reproducibility.public_trajectories 100

These commands check the public derivative manifests, archive hashes, and four member hashes. The original-byte validators and previously sealed figure fixtures use the original packets, which are held separately in the private historical archive. See [Data availability](../DATA_AVAILABILITY.md).

### Reading historical command lines

Historical receipts identify their producing commit and use paths from its
checkout. To repeat the old verifier CLI or unpickle a trusted historical
callable, use a separate checkout of that commit. For the last complete legacy
layout, the following example verifies the saved B40 package from an empty
temporary directory. Run it from a clone that contains the recorded commit.

```sh
repo=$(git rev-parse --show-toplevel)
scratch=$(mktemp -d)
git clone --quiet --local --no-checkout "$repo" "$scratch/source"
git -C "$scratch/source" checkout --quiet --detach 16d5e5c4787c29c5daf176881f57ba8d39468530
(cd "$scratch/source" && python3 -B -m tools.reproducibility.verify b40 \
  --directory benchmarks/reproducibility/shareable_milestone_v1)
```

A historical producer with an embedded source hash must use its **own recorded
commit**. For example, the [physical recapture plan](../results/reproducibility/physical_export_recovery_v1/recapture-plan.json)
identifies the fixture path and digest, and the [closure manifest](../results/reproducibility/physical_export_closure_v1/manifest.json) records the exporter digest and recapture commit. The recipe above runs a
saved-file check only. See [saved results](../results/) and [TRUBA
tooling](../infrastructure/truba/README.md) for the boundaries between archived
verification, scientific replay and physical execution.
