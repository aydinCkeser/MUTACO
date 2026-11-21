# Data and software availability

The intended public source is [aydinCkeser/MUTACO](https://github.com/aydinCkeser/MUTACO).
See [CITATION.cff](CITATION.cff) for the software and preprint citation and
[DATA_LICENSE.md](DATA_LICENSE.md) for the dataset inventory, versions,
and attribution. The manuscript is available as
[arXiv:2610.04619v2](https://arxiv.org/abs/2610.04619v2).

| Evidence | Location | Integrity check |
| --- | --- | --- |
| Paper-120 frozen synthetic inputs | [Cohort v3](experiments/graph_inversion/frozen_targets/paper120_v3/) | Per-target SHA-256 sidecars and cohort manifest |
| Compact B40/B100 trajectories | [v3r2-public exports](results/paper/scalar_primary_trajectory_archives_v3r2/README.md), with one archive per budget | Manifest, compressed archive, and all four members verified by the command below |
| Paper result tables | [Results index](results/README.md) and the dataset inventory | Use each available sidecar against its corresponding file |
| Portable B40/B100 verification packets | [B40](results/reproducibility/shareable_milestone_v1/) and [B100](results/reproducibility/b100_regression_v1/) | Saved SHA-256 inventories and read-only comparators |

Run from the repository root with Python 3.11 or 3.12.

```sh
python -m reproducibility.public_trajectories 40
python -m reproducibility.public_trajectories 100
python -m reproducibility.verify b40 --directory results/reproducibility/shareable_milestone_v1
python -m reproducibility.verify b100 --directory results/reproducibility/b100_regression_v1
```

These commands verify saved artifacts. The public compact exports replace only
the recorded absolute Python interpreter path. The new checksums describe the
public files. Original packet hashes remain in the provenance, and the
byte-exact historical validators still require the original private archives.
Account-specific paths in retained operational records use placeholders. Scheduler IDs, resource settings, and scientific provenance are retained. Historical hashes can identify private originals.
The original source revisions, private packets, and any required resource
allocation remain subject to the research team's access policy. Public access
to an original TRUBA environment or byte-identical rendering is outside this
release's reproducibility claims.

For an APS submission, cite the versioned software and datasets in the
references and in the submitted data-availability statement. After approval, deposit the versioned release in Zenodo or another persistent archive and cite the version DOI. The preprint DOI identifies the article.
