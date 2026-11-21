# Analytic forward models

These functions supply known test problems for target matching. An experiment evaluates a hidden parameter to construct a target response, gives the response to the inverse search, and compares its recommendation with the hidden solution. The forward function and MUTACO's fitted inverse-search surrogate have separate state.

- [Handwritten response](handwritten.py) maps two coordinates to three ordered channels. The adapter can append a constant channel or use task-diagonal observation noise according to its configuration.
- [Frozen GP landscape](gp_landscape.py) constructs a reproducible random-feature function with a specified number of parameters and response components. Its exported state and fingerprint determine the function for subsequent evaluations.

[Build metadata](../pyproject.toml) installs these sources under the public name `mutaco_analytic`. The installed package depends on `mutaco` and Torch. The experiment entry points, configs and SBI workflows stay outside the wheel. Use the [analytic experiment guide](../README.md) for inputs and output formats. Named model imports and configuration inspection do not evaluate these functions. Running the experiments does.
