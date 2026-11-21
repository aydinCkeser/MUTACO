# Graph inversion

This package binds the shared benchmark machinery to the scientific graph
oscillator. `paper_policies.py` loads typed option profiles from
`configs/profiles/`; canonical and Paper-120 choices dispatch through the same
MUTACO runner and continuous recommendation implementation.

The [paper declarations](../paper/configs/README.md) specify the compared methods and execution budgets. The [frozen response cohort](frozen_targets/README.md) supplies targets for those comparisons. Historical declarations continue to resolve the original paths. The compact completed result and its provenance are indexed in the [results catalog](../../results/README.md).

`paper_v3_callbacks.py` and `run_paper_v3_worker.py` implement the capped,
paired-assessment population boundary. Preparing or inspecting configurations
does not authorize a run.

## Supported entry points

- [run_paper_v3_worker.py](run_paper_v3_worker.py) is the Paper-v3 population
  CLI. `python -m experiments.graph_inversion.run_paper_v3_worker --help` is
  safe through the approved local launcher; actual execution can invoke physics.
- Scalar-primary execution uses the existing
  [TRUBA worker](../../infrastructure/truba/run_scalar_primary_population_worker.sh)
  and [submission boundary](../../infrastructure/truba/submit_scalar_primary_population.sh).
  [primary_benchmark/worker.py](../paper/primary_benchmark/worker.py) supplies
  Python orchestration functions, not a `python -m` command.
- [Offline analysis](../paper/analysis) reads persisted results. The
  [read-only verification tools](../../reproducibility/README.md) check
  saved hashes/history/checkpoints without fitting, physics or submission.

[run_experiment.py](run_experiment.py) is an unsupported historical interface,
not the current campaign or offline-reporting command. Its import and `--help`
are safe; `execute` raises `ModuleNotFoundError` for the absent historical
`plot_results` module before loading a configuration or calling `run_campaign`.
The accompanying `summarize_results` module is absent too. Restoring that API
would require a separate review of its original reporting contract; current
benchmark APIs must not silently substitute a new interpretation of old results.
