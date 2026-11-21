# graph paper120 b100 v1 report

**Status:** accepted canonical repository evidence

**Classification:** primary paper comparison

**Figure status:** discussion pending; no figures generated

## Experiment parameters

| Parameter | Value |
|---|---|
| Frozen targets | 120 |
| Methods | MUTACO, NPE, SNPE |
| Inference budget | 100 calls per method and target |
| Assessment | one paired held-out call per method and target |
| Assessment randomness | shared target-addressed disorder, initial state, and thermal noise |
| Successful full-cohort calls | 36,360 |

## Method results

All errors are lower-is-better. Values are means over all targets; medians are shown in parentheses.

| Method | Response MAE | Response L2 | Parameter L2 |
|---|---:|---:|---:|
| MUTACO | 0.0389974 (0.0326168) | 0.2650662 (0.2224479) | 0.3917221 (0.3453756) |
| NPE | 0.0671341 (0.0539570) | 0.4780793 (0.3645253) | 0.3699192 (0.3583801) |
| SNPE | 0.0755838 (0.0438847) | 0.5402168 (0.3015828) | 0.4283973 (0.4153994) |

## Paired response-MAE results

- MUTACO minus NPE: mean `-0.0281367`; 78 wins, 0 ties, 42 losses; seeded 95% percentile bootstrap interval `[-0.0408373, -0.0169419]`.
- MUTACO minus SNPE: mean `-0.0365864`; 77 wins, 0 ties, 43 losses; seeded 95% percentile bootstrap interval `[-0.0533359, -0.0213925]`.
- NPE minus SNPE: mean `-0.0084496`; 59 wins, 0 ties, 61 losses; seeded 95% percentile bootstrap interval `[-0.0270566, 0.0107940]`.

## Call accounting

The comparison contains 36,360 successful physical calls: 36,000 inference/training calls and 360 paired assessments. There were 0 failed calls and 0 retries. Export, transfer, consolidation, analysis, verification, and reporting add zero scientific calls.

## Provenance

The accepted evidence manifest is `benchmarks/paper/graph_paper120/b100_v1/manifest.json` at SHA-256 `14de31cc009fffc8003bb1bb3ef67d43d349eef2982d9a513740c76618eca936`.
The paired-record SHA-256 is `29f6af90cdec9581dd3b850636c05c6dd740dac2aa22164c7f4bc68b3ab61f67`.
The canonical bundle identity is `paper-canonical-population-evidence-v1-37f198a4c844356de1e1c538f81e036afcd26ea3ef638393e3e9fca9d3b82fa8`.

## Figures pending

No panels, plot types, captions, or rendering choices have been approved.
