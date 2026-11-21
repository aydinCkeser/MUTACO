# Historical Paper-120 operational checkpoint

This retained report describes an earlier campaign checkpoint. Statements below
about running or missing jobs are historical, not a live scheduler query.
Use [project state](../../project/state.toml), the
[current paper scope](../../scientific/paper_scope/README.md) and the
[28 October engineering audit](../../scientific/refactor_validation/submission_readiness_audit.md)
for current review context. No campaign authorization follows from this page.

Paper-120 B40 is complete: 120 targets, 240 completed method slots, 9,840
successful calls, no failures/retries/exclusions. Curated `main` retains the
core, simulators, active profiles, frozen cohort, compact evidence, current
strategy, and figure/caption framework. Full research history is preserved at
`archive/full-research-state-20260809` and tag
`paper120-b40-complete-20260809`.

The three-target B40 SNPE qualification is complete: targets 0, 49, and 99
produced 3/3 completed records with 123/123 successful calls. The immutable
records passed computation-free paired-context validation after correction of
a post-run mapping-hashability defect; no scientific retry occurred. Compact
SNPE evidence is consolidated with the matching Paper-120 MUTACO/NPE rows.

The missing-slot SNPE campaign is prepared for TRUBA: it excludes the three
completed qualification targets and contains the remaining 117 frozen targets
in 39 equal three-target shards. Each worker is capped at 123 calls and the
global cap is 4,797 calls. The zero-call production-worker preview resolved the
frozen ledger and method specification successfully.

TRUBA validation-only passed at pushed commit `0ad911166b09716efe0fd0e491d8130d03d75b54`.
It confirmed disjoint source/log/output paths, the pinned SBI environment,
SNPE-only dispatch, the 39-worker geometry, and a 123-call cap for rank 0.
The campaign output root remains absent and no matching scheduler job exists.

TRUBA accepted submission attempts `6214084` and `6214096` from the exact
validated scratch checkout, then held both with
`launch_failed_requeued_held` before allocating nodes or starting the batch
script. Both attempts had zero runtime, produced no campaign output, and were
cancelled to prevent a later unintended start. No matching job remains and no
scientific call ran. The next gate is resolution of the TRUBA scheduler launch
failure before any separately authorized resubmission. Machine authority:
`project/state.toml`.

An explicit 64-GB-per-node correction was pushed at `528fffa6`; zero-call
eight-node diagnostic job `6214173` nevertheless reached the same pre-script
launch hold and was cancelled. Retained successful jobs `6213306` and
`6213308` used 64 GB per node plus exclusion of `barbun112-119`, completing on
`barbun98-105`. At that diagnostic checkpoint no scientific output or active
MUTACO job existed.

The proven node exclusion was pushed at `46a2a45`. Zero-call diagnostic job
`6214179` then completed successfully across eight nodes on `barbun98-105`.
The authorized 117-target SNPE campaign is now running once as Slurm job
`6214181` from that exact commit and node set; its scientific-call ledger will
remain unreported until terminal collection.

The shipment license remains unresolved pending written ownership and release
authority confirmation from Bilkent TTO against the project agreements.
