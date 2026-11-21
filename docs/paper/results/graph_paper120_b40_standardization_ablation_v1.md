# graph paper120 b40 standardization ablation v1 report

**Status:** accepted canonical repository evidence

**Classification:** paired response-standardization ablation

**Figure status:** discussion pending; no figures generated

## Experiment parameters

| Parameter | Value |
|---|---|
| Frozen targets | 120 |
| Methods | MUTACO standardized, MUTACO unstandardized |
| Inference budget | 40 calls per method and target |
| Assessment | one paired held-out call per method and target |
| Assessment randomness | shared target-addressed disorder, initial state, and thermal noise |
| Successful full-cohort calls | 9,840 |

## Method results

All errors are lower-is-better. Values are means over all targets; medians are shown in parentheses.

| Method | Response MAE | Response L2 | Parameter L2 |
|---|---:|---:|---:|
| MUTACO standardized | 0.0452620 (0.0420632) | 0.3107794 (0.2862718) | 0.4370030 (0.3864594) |
| MUTACO unstandardized | 0.0423348 (0.0367570) | 0.2977823 (0.2588835) | 0.3914015 (0.3775397) |

## Paired response-MAE results

- MUTACO standardized minus MUTACO unstandardized: mean `0.0029272`; 52 wins, 0 ties, 68 losses; seeded 95% percentile bootstrap interval `[-0.0035445, 0.0090982]`.

## Call accounting

The comparison contains 9,840 successful physical calls: 9,600 inference/training calls and 240 paired assessments. There were 0 failed calls and 0 retries. Export, transfer, consolidation, analysis, verification, and reporting add zero scientific calls.

## Provenance

The accepted evidence manifest is `benchmarks/paper/graph_paper120/b40_standardization_ablation_v1/manifest.json` at SHA-256 `5b30c721cd4cec591bf1b62eb8cda6e55f1802f1f3132f6b56ff230959106b00`.
The paired-record SHA-256 is `e8eb2b513d60318ee54228f132cfbddb1bda741fc888f7794ae06131f3d282eb`.
The canonical bundle identity is `paper-canonical-population-evidence-v1-300e2d86714c653f961d0860cf35634e5367d6a966dcbbacee1265405e7b06e1`.

## Figures pending

No panels, plot types, captions, or rendering choices have been approved.
