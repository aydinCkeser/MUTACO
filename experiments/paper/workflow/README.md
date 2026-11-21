# Paper Lab workflow foundation

This package is an orchestration layer around accepted paper primitives. It
does not implement a new scientific runner. Study and execution documents bind
the existing unified runner; ingestion documents bind immutable result
fragments; analysis, report, and figure documents are separately versioned.

The foundation exposes read-only validation and planning commands through:

```text
python -m experiments.paper.workflow.cli
```

Scientific submission and figure rendering remain unavailable until separately
approved. Every evidence-stage command executes zero simulator calls.

The generic `ingest build` path materializes a new append-only canonical bundle
from hash-paired Paper-v3 worker populations. It requires the explicit runtime
flag `--confirm-authorized`, validates cohort coverage, method identities, call
accounting, and paired assessment coordinates, and adds zero simulator calls.
`verify BUNDLE` independently reloads the resulting compact bundle.

The standard post-run path is:

1. `ingest export WORKER_ROOT --output-prefix PREFIX --confirm-authorized`
   creates a deterministic package containing only the global collection,
   method records, assessment-randomness coordinates, and their hash sidecars.
2. `ingest fetch REMOTE_PREFIX --destination NEW_ROOT --confirm-authorized`
   retrieves the exact package (local paths are also supported for testing),
   verifies both package files and every member, rejects unsafe archive paths,
   and extracts into a new append-only staging root.
3. `ingest build` and `verify` construct and independently audit the canonical
   bundle.
4. `report build SPEC --confirm-authorized` deterministically materializes the
   Markdown report; `report build SPEC --check` verifies it without writing.
5. `promote verify SPEC` requires the canonical evidence, deterministic report,
   [reviewed promotion bindings](../../../results/paper/promotion_state.toml), and the [experiment registry](../../../results/experiment_registry.toml) to agree.

The TRUBA population batch epilogue runs step 1 only after successful worker
collection. Transfer and repository promotion remain explicit authorization
gates.
