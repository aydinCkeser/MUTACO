# Graph-oscillator scientific and retained evidence

This directory contains immutable characterization evidence captured from the
read-only historical repository at commit
`9e81949fc8be16bf967f5b1c27b39f082f096e6a`.

Decision 0018 selects `graph-oscillator.overdamped-scientific.v1` as the
scientific model and preserves the incompatible historical implementation as
`graph-oscillator.retained-oracle.v1`.

The fixture under `reference_fixtures/` is a numerical oracle, not scientific
endorsement. It freezes retained CPU NumPy/Numba behavior and is never
relabelled as scientific-model output.

`retained_cpu_small_v1.json` uses:

- canonical theta `(pairwise_coupling_mean=1.25,
  pairwise_coupling_CV=0.30)`;
- retained expansion `(1.0, 1.25, 1.0, 0.30)`;
- `H=1`, `N=3`, `B=2`, `R=2`;
- forces `(0, 5)`, gauges `(0, pi)`, hence `V=4` and `M=8`;
- `dt=0.1`, three integration steps, temperature `0.15`, seed `73007`;
- exact retained CPU NumPy/Numba coupling execution.

Each tensor record includes named axes, shape, dtype, raw-byte SHA-256, raw
base64 data, and readable values. Captured tensors include realized node and
edge disorder, initial state, overdamped mask, every thermal-noise draw,
first-step intermediates, first-step state and accumulators, final state and
accumulators, raw observables, and the final flattened response.

Regenerate only to a new path:

```powershell
python simulators/graph-simulator/audits/capture_retained_graph_fixture.py `
  --historical-repository ..\mutaco-historical `
  --output <new-output.json>
```

The script refuses to overwrite an existing fixture.
