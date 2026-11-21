# Comparing target-matching methods

A benchmark campaign fixes a simulator adapter, a target bank, a selection of
targets, the methods and the resources available to them. Its method records
attribute calls to target construction, inference and held-out assessment.
Reports must retain failed rows as well as completed rows to preserve the
denominator of a comparison.

| Task | Where to start | Inputs and outputs |
| --- | --- | --- |
| Load or execute a campaign | [api.py](api.py) | `load_campaign` reads the declared configuration. `run_campaign` may evaluate a simulator, fit models, train optional SBI methods and write an output tree. |
| Interpret stored rows | [results.py](results.py) | `load_results(directory)` reads `manifest.json` and `metrics.json`. Select targets, methods or statuses before summarizing. `export_tables` writes CSV and JSON to the supplied directory. `plot_results` consumes one selected summary group and an explicit style. |
| Find the physics binding | [simulator adapters](simulator_adapters/README.md) | The adapter exposes the declared parameter and response axes, evaluation context and optional latent errors to the benchmark engine. |
| Find a method or accounting policy | [methods](methods), [engine](engine), [statistics](statistics) | Method implementations supply decisions. The engine owns reference execution, replay and target records. Statistics reduce the recorded metrics. |

The names exported by `experiments.benchmarking` and the historical
`experiments.benchmarking.api` remain supported. Importing a stored-result
view through the package reads no campaign, trains no method and does not
initialize the method dispatcher. The metrics and plotting dependencies of
those views still apply. The paper's scalar qLogNEI baseline is implemented in
[scalar_qlognei.py](methods/scalar_qlognei.py); the optional composite
multitask-response NEI comparison has a different scientific identity.
