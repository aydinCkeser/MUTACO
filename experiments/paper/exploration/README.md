# Paper exploration workspace

This directory is the controlled entry point for custom tests, analyses, and
plots. Exploratory work must declare classification
`exploratory_non_paper`, starts with `promotion_status = not_promoted`, and
cannot be consumed by accepted paper figures or reports.

Tracked reusable specifications belong in `specs/`. Temporary work and output
belong in ignored `work/` and `outputs/` directories. Exploration should reuse
the Paper Lab loaders and analysis primitives; it must not introduce a second
scientific runner.

Promotion is a separate future gate requiring frozen inputs, complete
identities, call-accounting validation, immutable outputs, and registry review.

See [physical response reconstruction](../response_reconstruction/README.md), [historical workflows](../historical_workflows/README.md), and the [workflow template](../workflow/templates/exploration_v1.json) for their current owners. Original source paths remain available for historical commands and checksums.

## Execution boundaries

A `plot` or `replay` filename does not guarantee offline execution. Import the
specific documented module only; no repository-wide import sweep is safe.
These boundaries were characterized with isolated subprocesses and tripwires
on graph `evaluate`, `evaluate_batch` and `evaluate_condition_grid`.

The five eligible population/mechanism renderers now live under
[figures/](../figures/README.md#persisted-input-renderers). Their old module
paths here are compatibility entry points, not additional implementations.
Physical and mixed reconstruction programs below stay separate. The offline
SBI response/parameter scatter is also retained here while its exact B100
row-check failure is unresolved; the figure guide records that boundary.

| Entry point | Import / explicit execution |
| --- | --- |
| [render_paper120_target098_archive_v1.py](render_paper120_target098_archive_v1.py) | Import-safe; explicit rendering reads a hash-validated response archive, with no simulator import in the renderer. Requires existing archive/manifest and external output directory; does not create a missing archive |
| [plot_paper120_target.py](plot_paper120_target.py) | Import and CLI help make no physical calls. Default `main`/`reconstruct` evaluates two physical recommendations; its injected evaluator is used only by tests |
| [plot_paper120_target098_mutaco_only_v1.py](plot_paper120_target098_mutaco_only_v1.py) | Import-safe; explicit `main` performs three condition-grid physical evaluations. Archive output flags do not turn it into an offline renderer |
| [plot_paper120_target_v2.py](plot_paper120_target_v2.py) | Import-safe main guard around the unchanged legacy notebook body. Explicit execution retains its original machine-specific root, physical evaluations, writes and later incomplete diagnostic code; not a portable supported tutorial. No `--help` interface |
| [plot_paper120_b40_convergence.py](plot_paper120_b40_convergence.py) | Import-safe; default execution loads saved diagnostics for plotting. Physical replay needs both existing explicit controls; it is not strict archived-response replay |
| [scalar_primary_replay.py](scalar_primary_replay.py), [replay_divergence.py](replay_divergence.py) | Archived-response numerical reconstruction with physical tripwires; GP fits still occur. Qualified-environment and authorization requirements remain |

The legacy notebook change only adds an import boundary: every original body
statement and module-global operation order is retained under direct script
execution. Tests stop its first physical request before entering any kernel.
They do not certify the later notebook cells or regenerate figures. The two
explicit physical reconstruction producers are unchanged. Their historical
thread/context/tolerance behavior is not a new execution authorization.

Use the frozen archive renderer for offline target-98 panels. Its interface
can be inspected safely from repository root:

```sh
infrastructure/local/run_mutaco.sh -m experiments.paper.exploration.render_paper120_target098_archive_v1 --help
```

Rendering requires `--archive-directory` and `--output-directory`; obtain the
approved frozen files first. TeX/plotting dependencies and figure source hashes
must also be respected. Do not run a physical producer to manufacture missing
render inputs. The [evidence index](../../../results/README.md#figures-source-arrays-and-experiment-outputs)
links existing arrays, figure sources and provenance; it does not promise that
every separate visualization archive is distributed in this checkout.
