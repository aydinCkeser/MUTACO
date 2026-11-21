# Sequential target matching

[The sequential runner](sequential_runner.py) owns the state for one target.
Start at `run_sequential`. Its inputs are a simulator adapter, a
`SequentialRunnerConfig` describing the scientific intent and execution
options, and an optional progress callback. The result records committed
observations, simulation attempts, fit and proposal histories, the terminal
recommendation where available, and the status that ended the search.

| Manuscript stage | What to follow in the runner |
| --- | --- |
| 1. Design and physical responses | Validate capability and option identities, resolve seeds, generate or accept the ordered initial design, evaluate it, and freeze the initial response transform. Committed responses and failed attempts are recorded separately. |
| 2. Fit the correlated GP | Prepare ordered observations and their standardized task values, fit the exact multitask model, and record the outcome. The response axes and frozen normalization remain consistent throughout the search. |
| 3. Propose, evaluate and update | Use the current GP to screen and refine acquisition candidates. Follow the explicit duplicate-recovery and failure branches before the simulator call. Commit each accepted new response using the frozen transform and refit on the enlarged dataset. |
| 4. Recommend | Pass the final available model to the selected terminal policy. The paper profile optimizes physical posterior-mean component MAE over continuous parameters. Terminal selection adds no simulator evaluation or GP refit. |

The [scientific package guide](../README.md) links the underlying GP,
acquisition and terminal computations. The runner records scientific and
operational failure separately so a partial run can be inspected without
treating every attempt as a successful observation. Experiments supply
adapters and profiles; the runner contains no concrete physical kernel.

Use `mutaco.run_sequential` for the existing public interface or
`mutaco.sequential_search.sequential_runner` to inspect this implementation.
The earlier `mutaco.orchestration.runner` name remains available to existing
experiments, stored Python references and users of the module's patch points.
