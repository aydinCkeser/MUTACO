# TRUBA execution

The original account-bound submission profiles and source-validation receipts are held in the private archive. Path examples identify the original directory layout using placeholder account names. Choose your own paths and confirm the execution policy before using any scheduler entry point.

Exact-source shell validation requires absolute source, interpreter, output and
log paths containing only ASCII letters/digits, `.`, `_`, `-` and `/`, without
parent-directory traversal. Spaces, shell metacharacters and newlines are
rejected intentionally. Keep these paths outside synchronized/personal working
directories when their names violate this restriction. Local analysis and
package imports can work from a checkout with spaces; that does not make it a
supported TRUBA deployment path. Lifecycle tests use a separate supported path
and independently exercise rejection; the production guard is unchanged.

## Execution versus saved-evidence verification

Use [reproducibility](../../reproducibility/README.md) to check saved
hashes, checkpoint packets, history/accounting, fit events and JUnit reports.
Its restricted command writes no files, executes no tests or GP fits, contacts
no scheduler and never searches for private reference archives.

This directory keeps deployment/lifecycle transport, Slurm launchers, environment
profiles, qualified numerical capture/replay, and collection/packaging. The
following Python modules own the comparison functions that the offline verifier
imports. Three modules now point to the canonical input data and package
locations. Their original source fingerprints can be verified at their
recorded revisions.

| Legacy module | Boundary |
| --- | --- |
| [refactor_reference.py](refactor_reference.py) | Stored-packet comparison is offline. `capture`, `historical` and `tests` run numerical work and require qualified Slurm compute; `tests` can fit GPs |
| [readability_regression.py](readability_regression.py) | Metadata/history/event/JUnit/input checks are offline; `finish` writes a sealed job-status bundle. Its `tests` action checks saved outcomes rather than executing pytest |
| [b100_regression.py](b100_regression.py) | Historical B100 metadata/environment/test gates; `finish` writes status/checksums. Preserve the original v1 test policy and resource declarations |
| [fit_event_evidence.py](fit_event_evidence.py) | Pure comparison of saved fit/event records; no GP calculation, but source-hash recorded |
| [shareable_evidence.py](shareable_evidence.py) | Portable verification is offline. Derivation/sealing writes new summaries from explicit private originals; original-byte verification requires those originals |
| [b100_evidence.py](b100_evidence.py) | Portable verification is offline. Export/packaging and persisted scheduler-accounting parsing are separate responsibilities; no automatic private transfer |

To verify the present checkout, use the current launchers and the
[read-only verification guide](../../reproducibility/README.md).
The `readability_regression_v1.env` lifecycle verifies both canonical trajectory
archives during zero-call preflight. Input manifests bind the original archive
bytes and qualified test expectations.

For a historical audit, place a detached checkout at a new absolute path
outside the architecture repository. To verify the six comparison sources at
the last pre-migration revision, set `LEGACY` to that external path and run
the following from the modern repository root.

```sh
git worktree add --detach "$LEGACY" 9f3ce1893112bb37c23e3aad1508564f55d2c81b
git -C "$LEGACY" rev-parse HEAD
(cd "$LEGACY" && python3 -B - <<'PY'
import hashlib, json
from pathlib import Path

pins = json.loads(Path("tests/fixtures/offline_verification_before_2c.json").read_text())
for module, expected in pins["pinned_implementation_sha256"].items():
    source = Path("infrastructure/truba", module + ".py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"historical verifier checksum differs: {source}")
print("Verified six original comparison source files.")
PY
)
```

The archived `validate_refactor_b40.sbatch` job takes both the exact B40
capture checkout and the candidate identified by its original comparator
hash. Run that protocol only with those original revisions and private inputs.
The `refactor_reference.sbatch` and `b100_regression.sbatch` reference-creation
jobs materialize the 9f3ce189 qualification harness next to the historical
baseline tree. Captures record the scientific and harness commits separately.
The candidate-only strict replay uses the current worker and existing sealed
references.

## Retained execution and historical workflow instructions

The zero-physical-call refactor qualification uses `refactor_reference_v1.env`,
`refactor_reference.sbatch`, and `refactor_reference.py`. Its accepted scope,
exact same-environment comparison rule, and first proposed helper extraction
are in [the validation contract](../../scientific_archive/validation_history/refactor_validation/README.md).
The [B40 reference manifest](../../results/reproducibility/shareable_milestone_v1/manifest.json)
retains the original baseline test failure as well as successful mathematical
references. This qualification does not authorize further submissions.

This directory retains the generic Paper-v3 sharding/call-cap machinery and
the Paper-120 CPU submission surface. The completed MUTACO/NPE campaign used
40 workers over eight Barbun nodes and added exactly 9,840 calls.

Exact-commit deployment and validation are owned by
`python -m infrastructure.truba.lifecycle prepare-validate`. The command
requires an explicit transfer confirmation, a clean checked-out commit, a
known clean TRUBA base source, a new remote root, the pinned remote Python, and
one tracked environment profile. It creates its incremental Git bundle under
the operating-system temp directory, invokes SSH/SCP only with argument lists,
streams the tracked LF-only remote scripts, refuses reused roots, runs only
`--validate-only`, and writes an append-only machine-readable receipt with
`scientific_calls = 0` and `job_submitted = false`. It deliberately exposes no
submission subcommand; scientific submission remains a separate approval.

The repaired MUTACO--SBI assessment surface is
`repaired_sbi_assessment_v1.env`. It validates a 120-worker, 30-node layout
with exactly six assessment calls per worker. Each rank owns one target at
both B40 and B100 and evaluates MUTACO, NPE, and SNPE sequentially under the
shared target--budget context. The batch template deliberately runs the
collector after `srun` reaches a terminal state even if one or more workers
failed; all attempts are append-only and automatic retry is disabled.

On the Windows workstation, invoke it through the pinned runtime launcher:

```powershell
.\infrastructure\local\run_mutaco.ps1 -- -m infrastructure.truba.lifecycle prepare-validate `
  --host truba `
  --base-source /arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/previous/source-exact `
  --base-commit FULL_40_CHARACTER_BASE_COMMIT `
  --remote-root /arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/STUDY/FRESH_ROOT `
  --remote-python /arf/home/REPLACE_WITH_TRUBA_USER/mutaco/envs/mutaco-sbi-0261/bin/python `
  --profile infrastructure/truba/TRACKED_PROFILE.env `
  --receipt runs/truba-receipts/STUDY/validation.json `
  --confirm-transfer
```

The remote root and receipt are append-only. A failed root is never reused.

The SBI-zoo qualification execution package uses
`sbi_zoo_qualification_v1.env`, `submit_sbi_zoo_qualification.sh`, and
`sbi_zoo_qualification.sbatch`. Lifecycle validation runs a non-publishable
zero-call preflight that checks the frozen target and 140-slot/four-recommendation
plan. It neither creates the reserved worker root nor submits a job. A later Q0
authorization must use the tracked self-contained declaration for accepted
historical Paper-100 target position 0; its sidecar binds the declaration to the
archived source record and source SHA-256. After lifecycle preparation issues
its append-only validation receipt, the immutable external declaration must be
created with `python -m experiments.graph_inversion.sbi_zoo_qualification
declare --lifecycle-receipt RECEIPT.json`. The declaration embeds and hashes the
fresh receipt while preserving its internal identity separately from its file
SHA-256. Physical
submission additionally requires `MUTACO_SBI_ZOO_EXECUTION_AUTHORIZED=1` and
is not authorized by the tracked profile.

`paper120_v3.py` validates the frozen cohort, method identities, 40-way shard
coverage, and historical 82-call per-target cap. The existing global collector
and submission surface now derive target/method coverage and call caps from a
pinned v3 declaration and configured worker count.

`paper120_b40_snpe_remaining117_v1.env` pins the relative scientific inputs,
39-worker geometry, and 123-call per-worker cap for the missing-slot SNPE
campaign. Site paths, the exact source commit, output root, and scheduler-log
root remain explicit environment inputs. `submit_paper120_v3_population.sh`
validates by default; `--submit` is a separate scientific authorization gate.

`paper120_b40_all_method_smoke_v1.env` pins a one-target, one-worker
MUTACO/NPE/SNPE qualification with a 123-call hard cap. Optional scheduler
geometry variables reduce that smoke allocation to one node and one task. The
task reserves the site-required 20 cores while the worker remains pinned to
eight computation threads. Population defaults remain eight nodes and forty
tasks. The smoke-specific exclusion also avoids `barbun70`, which failed before
batch launch during the first accepted one-node submission.

The batch template requests 64 GB per node explicitly. This matches the
completed eight-node Paper-120 launch and prevents Barbun's 8.5-GB-per-CPU
default from inflating a 320-CPU allocation to 2.72 TB.

The template also excludes `barbun112-119`, matching the retained successful
eight-node diagnostic and completed Paper-120 population launch. Slurm remains
free to select any available nodes outside that known-problematic block.

After a successful global collection, the batch epilogue now creates a
deterministic `compact-export/paper-v3-$SLURM_JOB_ID` package. It contains only
the collection, method records, assessment-randomness coordinates, and hash
sidecars; model tensors and posterior samples are excluded. The Paper Lab
`ingest fetch` command performs the separately authorized, verified transfer.

`paper120_b40_standardization_ablation_qualification_v1.env` pins the prepared
three-target, two-arm response-standardization qualification at 82 calls per
worker and 246 calls globally. It reuses the accepted Paper-120 assessment
campaign and is validation-only preparation; the environment file is not
scientific submission authority.

`paper120_b40_standardization_ablation_remaining117_v1.env` pins the exact
117-target complement after the completed qualification. Its 39 scientific
workers each own three targets and a 246-call cap. The 40-task, ten-node Barbun
allocation leaves one scheduler task scientifically idle while satisfying the
site's 20-core-per-node rule. This profile is also validation-only and grants
no population execution authority.

`primary_nei_q1_v1.env` selects the primary-Q1 validation surface. It binds the
five constructed off-cohort responses and post-construction declaration, five
one-target workers, and a 140-call per-worker / 700-call global qualification
ceiling. Validation routes through `primary_q1.py` and adds zero calls.
`submit_primary_nei_q1.sh` still requires an explicit authorization flag. The
first authorized qualification attempt, job `6244497`, preserved five failed
worker terminals after 80 successful initial-design calls. Zero-call diagnosis
reproduced the failure at the position-16 snapshot-state identity: legitimate
positive-infinity GP constraint bounds lacked an explicit canonical evidence
encoding. Signed infinities are now encoded explicitly, NaNs remain rejected,
and failed trajectory results are persisted before budget validation. No retry
is authorized. A new append-only retry declaration and exact source validate at
zero calls. The focused encoding regression passes in that TRUBA source. The
tracked real-GP compute smoke then completed as job `6244651` on `barbun13`:
one position-16 snapshot, three tier observations, four explicitly encoded
positive infinities, and zero simulator calls. The scientific retry remains a
separate authorization gate. The subsequently authorized retry, job `6244676`,
failed after 436 successful calls with zero simulator failures. One worker
completed both trajectories; four target-dependent full or partial trajectory
results reached a broader attempt-hash boundary containing non-finite evidence.
Zero-call diagnosis traced this to expected `-inf` values in full-trajectory
NEI raw-pool diagnostics, outside the snapshot-specific encoding path. Complete
Q1 attempts now cross one path-aware evidence canonicalizer before semantic
hashing and persistence; signed infinities use a reserved versioned tag, NaN
reports its exact path, and pre-hash failure envelopes contain no raw result.
Production-shaped B40/B100 lifecycle regressions pass without simulator calls.
No qualification result or production tier was created, and no further retry
is authorized.

The post-repair declaration is
`primary_nei_q1_evidence_repair_v1.json`; its companion run declaration names
repair ancestor `ae8dfdf`, excludes jobs `6244497` and `6244676` from tier
selection, and requires a complete matrix from the new declaration alone. The
disabled validation profile is `primary_nei_q1_evidence_repair_v1.env`.
The tracked lifecycle materialized exact commit `00c81dd` under the fresh
`qualification-evidence-repair-v1` root and passed validation-only preflight:
five target identities/order, declaration identity, graph configuration hash,
worker cap, scheduler geometry, and path disjointness all passed. The receipt
records zero scientific calls, no submitted job, and no submission authority.

The graph-specific driver is now implemented by `primary_q1_driver.py` and
`run_primary_q1_worker.py`. It saves exact GP state at every declared snapshot,
optimizes all declared tiers plus the high-effort reference candidate, and
scores their union with four independent common-base 4,096-sample batches.
`primary_nei_q1_target_construction_v1.env` is the historical validation-first
profile used for the completed exact five-call target construction. Both
profiles retain authorization flags set to zero.

`primary_nei_q1_compute_smoke_v1.env` selects the one-node, one-task
zero-simulator smoke surface. Validation emits the standard disjoint scheduler
layout and the canonical zero-call preview. Submission requires its own
authorization flag and runs only deterministic synthetic responses through the
real Q1 GP and qualification snapshot path; it deliberately stops before a
seventeenth response can be evaluated.

`primary_prequalification_b40_integration_smoke_v1.env` selects a distinct
one-node, one-task, 123-call-cap validation surface for the MUTACO/NEI/Sobol
integration smoke. It reuses exactly one frozen Q1 off-cohort target, declares
the high Q1 controls as provisional engineering settings, and is ineligible
for tier selection or primary performance reporting. Validation checks the
hash-paired declaration and plan, the selected target and graph configuration,
diagnostic persistence contracts, exact scheduler geometry, and append-only
path separation. Its execution authorization remains zero; the lifecycle has
no submission command.

Job `6247524` completed all 120 search calls and froze all three recommendations
before an incomplete assessment context failed at the production simulator
bridge. `primary_prequalification_b40_assessment_rescue_v1.env` selects the
fresh assessment-only recovery surface. It verifies the original freeze and
target ledger, previews zero calls, has no search runner, and hard-caps any
separately authorized execution at three paired held-out calls. Its tracked
authorization flag is zero and its fresh output/log paths are disjoint from the
failed attempt.

After that rescue was archived, `primary_nei_q1_post_smoke_v1.json` declared a
fresh five-target B40/B100 numerical-qualification attempt. It changes only the
append-only attempt/output identity relative to the prior Q1 scientific
declaration, names all three historical Q1 attempts as ineligible, and binds
the repaired evidence, covariance-factorization, and terminal paths through
its exact source. `primary_nei_q1_post_smoke_v1.env` remains validation-only
with a 140-call worker cap, a 700-call global ceiling, and execution authority
set to zero.

The tracked lifecycle materialized exact commit `f39df98` under
`qualification-post-smoke-v1` and validated the five-target ledger, frozen
configuration, 140-call worker cap, 700-call global ceiling, scheduler layout,
and pairwise-disjoint paths. Receipt
`q1_post_smoke_source_validation_receipt.json` records zero scientific calls,
no submitted job, and no submission authority. Scientific execution remains a
separate gate.

## SBI-zoo Paper-120 P1

`sbi_zoo_p1_v1.env` selects the zero-call lifecycle surface for the qualified
single-round NLE/NRE extension's shared-dataset stage. The canonical Python
surface is `experiments.graph_inversion.sbi_zoo_p1`; it freezes the ordered
Paper-120 cohort, 16,800 row slots, 240 expected datasets, lifecycle receipt,
and exact source before any physical call.

`submit_sbi_zoo_p1.sh` validates by default and requires a separate execution
flag for submission. Its physical array is `0-59%20`; task `i` runs logical
targets `i` and `i+60` sequentially in separate Python processes, preserving a
maximum of 20 simultaneously active target workers. A nonzero first target does
not suppress the second, and the physical task reports failure after both have
terminated if either failed. Each target freezes B40 before beginning
independently addressed B100. Each worker runs with 20 application threads and
requests 20 Slurm CPUs, 64 GB, and 30 minutes without exclusivity. Successful
historical Barbun jobs establish that this `20 CPU / 64G / non-exclusive`
representation receives `AllocCPUS=20`. The expected Slurm allocation is
therefore 20 CPUs per worker before execution; the authoritative observed value
must later come from `sacct`. Node colocation remains scheduler-controlled. The
64 GB request is a Barbun submission-policy compatibility choice, not an estimate
of application memory demand (qualification peak RSS was about 599 MiB). The
dependent non-exclusive collector requests 20 CPUs, 64 GB, and 15 minutes. Both
surfaces disable requeue. The collector opens the P2 gate only after
all 240 datasets and all 16,800 successful slots validate. No P1 execution is
authorized by the tracked profile.
