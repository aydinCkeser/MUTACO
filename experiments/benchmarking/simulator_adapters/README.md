# Benchmark simulator adapters

The registry maps the simulator identifier in a campaign declaration to the
operations required by the benchmark. Keep the identifier, parameter axes,
target realization and assessment context together when interpreting a result.

- [Handwritten analytic functions](handwritten.py) and the [frozen GP landscape](gp_landscape.py)
  connect controlled analytic target-matching problems to the installed
  `mutaco_analytic` model family. The landscape is the forward function in
  that experiment, distinct from the response GP learned during inversion.
- The [graph adapter](../simulators/graph_scientific_d4.py) binds the four
  paper parameters to the graph simulator's twenty response components.
  The B40 held-out terminal-domain v1 and v2 declarations fingerprint this
  exact file in its original location. Both registry paths load this one class.
- [linear.py](linear.py) contains the diagnostic linear forward map and its
  adapter. Its smaller response dimension is useful for mechanical tests.

Reading the registry is distinct from running an adapter. Calling its
evaluation methods can produce a new response. Existing
`experiments.benchmarking.simulators` imports and patch points resolve to the
same registry and classes as `experiments.benchmarking.simulator_adapters`.
