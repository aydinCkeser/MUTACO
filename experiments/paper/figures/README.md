# Paper figures and captions

## New figure renders

Use the versioned command when rendering a new review copy from stored data.
For example, from the repository root:

```sh
infrastructure/local/run_mutaco.sh -m experiments.paper.figures.render_versioned --help
infrastructure/local/run_mutaco.sh -m experiments.paper.figures.render_versioned primary-response --version v1
```

The command uses the existing loaders and deterministic PDF/PNG producers and
writes both original filenames to `results/paper/figures/<figure>/<version>/`.
Choose `primary-response`, `primary-parameters`, `sbi-response` or
`terminal-domain` and a fresh version such as `v1` or `v2`. It refuses an
existing version or a destination through a symbolic link. Input checks and
rendering complete before a new version is published. The production TeX and
font requirements still apply. Some producers also require the recorded compact
exports to be available at their expected input locations.

The Fig. 7 local-sensitivity script has only a fixed manuscript-PDF output in
its recorded implementation. It has no destination-aware rendering function;
use its existing archived figure for review. Frozen archive producers and
physical reconstruction are listed separately below.

## Persisted-input renderers

The following implementations now live directly in this package. They read
frozen results and render offline; they never run a physical simulator, fit a
GP or generate a new recommendation. Scientific/rendering expressions and
default output names, formats and locations are preserved.

| Implementation | Stored inputs / output |
| --- | --- |
| [plot_paper120_primary_boxplot_v1.py](plot_paper120_primary_boxplot_v1.py) | Primary cross-budget synthesis → B40/B100 response-MAE boxplots |
| [plot_paper120_primary_parameter_boxplot_v1.py](plot_paper120_primary_parameter_boxplot_v1.py) | Frozen targets and archived primary recommendations → parameter-error boxplots |
| [plot_paper120_sbi_response_boxplot_v1.py](plot_paper120_sbi_response_boxplot_v1.py) | Repaired SBI and SBI-zoo results → response-MAE boxplots |
| [plot_paper120_terminal_domain_ablation_v1.py](plot_paper120_terminal_domain_ablation_v1.py) | Frozen B40 terminal-domain v2 results → paired B'/C' comparison |
| [plot_paper120_local_sensitivity_v6.py](plot_paper120_local_sensitivity_v6.py) | Hash-pinned twelve-target plotting data → presentation-only local-sensitivity panel |

Use the [approved local launcher](../../../infrastructure/local/README.md).
For example, this command shows help without rendering or writing outputs:

```sh
infrastructure/local/run_mutaco.sh -m experiments.paper.figures.plot_paper120_primary_boxplot_v1 --help
```

All five support safe `--help`, their original no-argument rendering behavior,
and standalone-file execution. Corresponding imports and commands under
`experiments.paper.exploration` remain compatibility paths to the same code.
Direct-file help needs only the plotting dependencies, without loading the
scientific paper facade. Unknown arguments are rejected before rendering.

**Use the versioned command above for new renders.** The historical no-argument
commands keep their original output destinations. The local-sensitivity
producer writes a manuscript PDF, and the other producers use their legacy
exploration-output directories.
This relocation rendered only into external temporary test directories.
Original production plots require their existing TeX/fonts. This migration's
before/after captures used explicitly labelled test-only DejaVu fallback fonts;
they do not establish byte reproduction of the historical TeX PDFs.

## Shared figure inputs

[validated_inputs.py](validated_inputs.py) reads hash-verified Paper-100 data.
[target_selection.py](target_selection.py) resolves the plotted cohort,
[style_registry.py](style_registry.py) resolves semantic styles,
[plot_validation.py](plot_validation.py) checks saved SVG, PNG and PDF artifacts and
[figure_artifacts.py](figure_artifacts.py) writes and inventories outputs.
The former `figures.common` names refer to the same modules.

## Deliberately separate producers

- Physical reconstruction is cataloged in [response reconstruction](../response_reconstruction/README.md) and [historical workflows](../historical_workflows/README.md):
  `plot_paper120_target.py`, `plot_paper120_target098_mutaco_only_v1.py` and
  `plot_paper120_target_v2.py` can perform real evaluations. The convergence
  utility mixes offline plotting with separately gated physical replay.
- The [target-98 archive helper](../exploration/target098_response_archive.py)
  and [archive renderer](../exploration/render_paper120_target098_archive_v1.py)
  remain at their recorded source paths; archive provenance hashes those files.
- The frozen cohort-figure producer stays beside its
  [archived outputs/provenance](../oscillator_physics_multi_target_sensitivity/outputs/paper120_maximin12_job6353452).
  The [SBI-zoo comparison producer](../studies/graph_paper120/analysis/sbi_zoo_comparison.py)
  stays with its source-specific study/specification and analysis lifecycle.
- The [SBI response/parameter scatter](../exploration/plot_paper120_sbi_response_parameter_scatter_v1.py)
  is offline but remains in place: its unchanged pre-move loader rejects B100
  MUTACO target 6 because persisted/recomputed parameter L2 values differ by
  `5.551115123125783e-17` under exact comparison. No tolerance or data was changed.

Moving a renderer does not promote its output into accepted paper evidence.
See the [evidence index](../../../results/README.md) for result authority.

## Figure registry and caption framework

The Paper-120 figure registry is `figure_plan.toml`. Figure preparation must
consume persisted, hash-validated inputs and add zero simulator or method
calls. Shared loaders, validation, style, and artifact helpers live under
`common/` and `styles/`.

Each completed figure bundle should contain PDF, SVG, and PNG outputs; the
plot-ready data; `caption.md`, `caption.tex`, and `caption.json`; and a manifest
that binds every numerical claim to source hashes. Human prose belongs in the
Markdown/TeX captions; machine facts and validation belong in JSON. Draft
registry order is not final manuscript numbering.

Prompt 21A establishes this surface but intentionally generates no final
figures.

The B40 and B100 Paper-120 scaffolds under
`experiments/paper/studies/graph_paper120/figures/` deliberately contain no
panels. Their `rendering_authorized` values remain false until figure content
is discussed and separately approved.
