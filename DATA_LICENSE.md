# Dataset license

The contributors have agreed to release the datasets listed below under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
All six dataset entries credit **MUTACO Research Team, Bilkent University** as
their collective creator. When reusing a dataset, give this creator credit and
include the dataset title and version from the table, the release URL or DOI,
a license link, and an indication of any changes.

| Dataset | Version | Covered files |
| --- | --- | --- |
| Paper-120 frozen synthetic targets | `paper120_v3` | `experiments/graph_inversion/frozen_targets/paper120_v3/target_*.json` |
| Scalar-primary B40/B100 trajectories and endpoints | `v3r2-public` | The two `*-public.tar.gz` archives listed in [Data availability](DATA_AVAILABILITY.md), including their numerical JSON payloads |
| Scalar-primary B100 and B40/B100 synthesis rows | `graph_scalar_primary_b100_v1`, `scalar_primary_b40_b100_synthesis_v1` | `results/paper/graph_scalar_primary_b100_v1/analysis_rows.jsonl` and `results/paper/scalar_primary_b40_b100_synthesis_v1/cross_budget_rows.jsonl` |
| SBI comparison tables | `sbi_zoo_comparison_v1` | `results/paper/sbi_zoo_comparison_v1/{budget_change,outliers,paired_vs_mutaco,rows,summary}.csv` |
| Terminal-domain ablation rows | `terminal_domain_ablation_b40_refit_v1` | `results/paper/terminal_domain_ablation_b40_refit_v1/rows.jsonl` |
| Portable mathematical verification datasets | `shareable_milestone_v1`, `b100_regression_v1` | The `*.json.gz` mathematical snapshots in those two `results/reproducibility/` directories |

## Shared scientific provenance

The six datasets arose from the same MUTACO research program under the
scientific leadership and supervision of **Aydın Cem Keser** throughout the
project. The work progressed through these stages.

1. **Aydın Cem Keser** formulated the inverse problem and designed the stochastic
   oscillator forward model. **Cem Tekin** formulated the initial acquisition
   algorithm. They developed the original MUTACO research proposal together.
2. **İlter Onat Korkmaz** developed the first numerical implementation of the
   proposed methodology.
3. **Umut Can Turhan** and **Ege Karadeniz** contributed to further algorithm
   development and implementation and reproduced numerical results.
4. **Aydın Cem Keser** substantially redesigned the algorithm and methodology.
   **Cem Tekin** reviewed and endorsed the overhaul, then worked with Keser to
   redesign the workflow and settle implementation choices.
5. **Ege Karadeniz** contributed to implementing the revised methodology.
   **Aydın Cem Keser** ran the final numerical experiments and analyzed the
   results. **Ege Karadeniz** reproduced and independently verified the results
   and prepared the datasets for release.

The creator credit recognizes scientific contributions. Copyright ownership and
licensing authority are separate legal matters. [CITATION.cff](CITATION.cff)
records the software and preprint author credits.

[BSD-3-Clause](LICENSE) covers the original MUTACO software. Manuscripts,
standalone figures, third-party material, and private originals are outside the
dataset license. Third-party notices and licenses apply to their respective material.
