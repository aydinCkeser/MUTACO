# Graph Paper-120 B40 three-method report

**Status:** accepted tracked compact evidence  
**Classification:** primary paper comparison  
**Figure status:** discussion pending; no figures generated

## Experiment parameters

| Parameter | Value |
|---|---|
| Frozen targets | 120 |
| Methods | MUTACO, cold-start NPE, SNPE |
| Inference budget | 40 calls per method and target |
| MUTACO allocation | 16 initial + 24 sequential |
| NPE allocation | one round of 40 |
| SNPE allocation | 14 + 13 + 13 |
| Terminal policy, MUTACO | continuous posterior-mean physical L1/MAE |
| Terminal policy, NPE/SNPE | posterior-sample MAP in physical coordinates |
| Assessment | one paired held-out call per method and target |
| Assessment randomness | shared target-addressed disorder, initial state, and thermal noise |

## Method results

All errors are lower-is-better. Values are means over 120 targets; medians are
shown in parentheses.

| Method | Response MAE | Response L2 | Parameter L2 |
|---|---:|---:|---:|
| MUTACO | 0.0401100 (0.0396044) | 0.2718813 (0.2716197) | 0.4142873 (0.3677705) |
| NPE | 0.1342731 (0.0975258) | 0.9510522 (0.7076265) | 0.5099710 (0.4975238) |
| SNPE | 0.1429250 (0.1057520) | 1.0319610 (0.7741502) | 0.4959962 (0.4584642) |

For response MAE, MUTACO beat NPE on 102 of 120 targets and beat SNPE on
102 of 120 targets. NPE beat SNPE on 52 targets and lost on 68.

## Call accounting

The accepted three-method comparison contains 14,760 successful physical
calls: 14,400 method inference/training calls and 360 paired assessments.
There were zero failed calls and zero retries. Consolidation added zero
scientific calls.

## Provenance

This report is bound to
`benchmarks/graph/paper120_mutaco_npe_snpe_v1/manifest.json` at SHA-256
`ce08ebecad386a488539b3d9fe2a827580f173499c6977311b64d5b3649523cc`.
The paired-record file is bound by that manifest at SHA-256
`cd0352310c8e570bdc56fe6e4ef587bd58e1c26f1f0865d44878ffc7dd2c0c72`.
