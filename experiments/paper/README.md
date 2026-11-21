# Paper studies and experiment protocols

Start with [studies](studies/README.md) to find a scientific comparison, its
frozen inputs and declared outputs. The [protocol](protocol/README.md) owns
how decisions are frozen before a held-out assessment. The graph experiment
binds that protocol to the [physical callbacks](../graph_inversion/README.md).
Shared counting and validation types belong to [experiments/contracts](../contracts/README.md).

Choose [offline analysis](analysis/README.md) to interpret saved method records,
[figures](figures/README.md) to render stored series, or
[response reconstruction](response_reconstruction/README.md) to track a program
that can evaluate a physical model. [Historical workflows](historical_workflows/README.md)
cover source-specific replay, assessment and forensic analysis. The [workflow
guide](workflow/README.md) explains artifact ingestion, reporting and promotion.
Produced evidence is indexed under [results](../../results/). Existing
`experiments.paper.common` imports remain available, along with the original
source files that immutable declarations identify by path.

## Offline analysis

[analysis/](analysis) is the shallow home for result analysis. It consumes
persisted observations and reports, rather than running scientific experiments
or fitting GPs. Statistical reductions, bootstrap seeds, ordering, schemas,
formatting and archived outputs are unchanged.

| Responsibility | Implementation |
| --- | --- |
| Validated record loading, paired/population summaries and bootstraps | [aggregate.py](analysis/aggregate.py) |
| Persisted timing summaries, including unavailable clocks | [timing_report.py](analysis/timing_report.py) |
| Descriptive B40/B100 synthesis of independent campaigns | [cross_budget_synthesis.py](analysis/cross_budget_synthesis.py) |
| Check/materialize the frozen cross-budget package | [cross_budget_synthesis_cli.py](analysis/cross_budget_synthesis_cli.py) |
| Check/materialize the frozen B40 cross-experiment package | [evidence_analysis_cli.py](analysis/evidence_analysis_cli.py) |

Both `experiments.paper.analysis.evidence_analysis_cli` and
`experiments.paper.analysis.cross_budget_synthesis_cli` support `python -m`
through the [approved local launchers](../../infrastructure/local/README.md).
Their existing `--repository-root` and `--check` arguments are unchanged.
Use `--check` to compare against archived files without writing; omitting it
retains the original materialization behavior. The corresponding
`experiments.paper.common.*` commands and imports remain compatibility paths
to the same implementations, including private functions, monkeypatch globals
and historical pickle references. There is no second analysis engine.

The B40 checker needs the expanded compact export named in its source constants.
That ignored fetch directory may be absent from a checkout. Tests restore the
exact export from the committed B40 archive into external temporary storage,
verify its frozen SHA-256, then compare complete output bytes with independent
archived reports. They do not generate replacement expected results or write
an expanded archive into the repository.

Saved analysis also depends on two original paths in `common/`:

- [cross_experiment_contract.py](common/cross_experiment_contract.py) owns the
  shared scientific vocabulary, identity and declaration validators used by
  execution/preparation code; the declaration validators preserve the stored scientific identities.
- [evidence_analysis.py](common/evidence_analysis.py) is offline, but the frozen
  B100 preparation manifest records its exact path and source hash. Relocating
  it needs an explicitly versioned provenance plan; a compatibility shim would
  change those bytes. Existing references and verifier checks remain intact.

## Choose the correct surface

- **Run science:** [graph workers/configuration](../graph_inversion/README.md)
  and [primary benchmark execution](primary_benchmark/execution.py). Physical
  execution requires separate authorization.
- **Analyze results:** [analysis/](analysis), using persisted inputs and frozen
  specifications in [studies/](studies).
- **Render stored data:** [write a fresh version under results](figures/README.md#new-figure-renders) or consult the
  [exploration execution table](exploration/README.md#execution-boundaries).
  Five persisted-input renderers live directly in `figures/`; legacy imports
  remain compatible. Physical/mixed reconstruction and source-recorded
  producers stay separate. Some reconstruction producers perform physical
  calls despite their names.
- **Verify evidence:** [benchmark index](../../results/README.md), hash
  sidecars and offline `--check` commands. These do not rerun historical search.


Import-only names and module commands formerly represented by individual
`paper/common` files resolve to their canonical owners on demand. The
standalone exact-source bootstrap still has a file entry point. Use
`python -m reproducibility.hash_pair_inventory` for the current hash-pair
inventory. The [native terminal verification](historical_workflows/README.md#verify-the-original-terminal-domain-declarations)
runs the two source-bound assessments' zero-call preparation in a detached
historical checkout.
