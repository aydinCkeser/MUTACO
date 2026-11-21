# Analytic MUTACO/SBI benchmark

Run the bounded Prompt 5B smoke from PyCharm or as a module:

```powershell
python -m experiments.analytic_target_matching.sbi_benchmark.run
```

Reuse completed data without rerunning science:

```powershell
python -m experiments.analytic_target_matching.sbi_benchmark.summarize
python -m experiments.analytic_target_matching.sbi_benchmark.plot --method mutaco
```

See `docs/benchmarks/PROMPT5B_GENERIC_BENCHMARK.md` for configuration,
interfaces, artifacts, environment, precision, device, budget, and timing
contracts.

## Prompt 5C calibration

Select the `mutaco-sbi` interpreter in PyCharm and run
`run_calibration.py` directly with no arguments. It uses the same
`device="auto"` configuration on CPU and CUDA, writes to
`runs/benchmarks/analytic/prompt5c_calibration/`, and is capped at 30 minutes
by parent-enforced deadlines.

The calibration uses hidden-target indices `0`, `49`, and `99`, 14 inference
calls per method and target, `[5, 5, 4]` for sequential SBI, and exactly
10,000 posterior samples. It is a runtime/failure calibration, not a
scientific method comparison. Its results must not be used to tune or select
methods.

The output directory had to be absent or empty for its approved fresh run.
Inspect `CALIBRATION_REPORT.md`, `calibration_summary.json`, `records.csv`, and
`manifest.json`. Prompt 5D remained separately approval-gated at calibration
completion and was later executed under its own approval.

The complete contract and artifact guide is
`docs/benchmarks/PROMPT5C_CALIBRATION.md`.

## Completed Prompt 5D campaign

Prompt 5D is complete. The first bundle at
`runs/benchmarks/analytic/prompt5d_campaign/` is immutable failed evidence.
The accepted corrected bundle is
`runs/benchmarks/analytic/prompt5d_campaign_corrected/`; do not rerun,
overwrite, or merge either bundle. Current status, limitations, and canonical
artifacts are in the living
[`RESULT_INDEX.md`](../../../docs/benchmarks/RESULT_INDEX.md) and
[`PROMPT5D_RESULTS.md`](../../../docs/benchmarks/PROMPT5D_RESULTS.md).

The master-target ledger constructed 24 noisy observations once. The 14, 40,
and 105-call regimes consumed those same records. The retained entry point and
its recovery controls are reproduction interfaces, not current execution
authorization; `--smoke` remains mechanics-only and must never be reported as
scientific evidence.
