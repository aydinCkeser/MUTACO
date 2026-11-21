# Paper-120 B40 result

The completed graph experiment used 120 frozen inverse problems. MUTACO and
cold-start target-specific NPE each received 40 inference/training simulator
calls per target, followed by one paired held-out assessment call per method.
Target construction and terminal decisions added zero calls.

| Method | Response MAE | Response L1 | Response L2 | Parameter L2 |
|---|---:|---:|---:|---:|
| MUTACO | 0.0401100 | 0.8022000 | 0.2718813 | 0.4142873 |
| NPE | 0.1342731 | 2.6854616 | 0.9510522 | 0.5099710 |

MUTACO had lower response MAE on 102 targets and higher MAE on 18. The mean
paired MUTACO-minus-NPE difference was `-0.0941631`; the predeclared 10,000
resample descriptive 95% percentile interval was
`[-0.117183, -0.0735712]`.

All 240 method slots completed. The ledger contains 4,800 MUTACO inference
calls, 4,800 NPE training simulations, and 120 assessment calls per method:
9,840 attempted and successful calls, with no failures, retries, or exclusions.

Authoritative compact inputs are under `benchmarks/graph/paper120/`; their
hashes are registered in `manifest.json`. The frozen cohort is under
`data/cohorts/paper120_v3/`. Full worker evidence and scheduler logs remain in
the preserved archive/local/remote locations recorded by the archive tag.
