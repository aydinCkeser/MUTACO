# MUTACO graph simulator

`mutaco-graph` contains the audited Prompt 7 graph-oscillator forward models.
It depends on public MUTACO simulator contracts and keeps graph physics out of
`mutaco` core.

## Versioned identities

- `graph-oscillator.retained-oracle.v1` preserves the historical equations,
  one-generator NumPy/SciPy sampling order, mixed dtype behavior, and M=12
  adapter profile. It exists for interpretation and numerical regression.
- `graph-oscillator.overdamped-scientific.v1` is the approved pure
  overdamped, Euler-Maruyama, native-Torch-float32 model with named streams and
  canonical M=20 response.

The identities are intentionally incompatible. Their differences are never
described as implementation-parity errors.

The package exposes typed P=2..6 profiles, strict named axes and device
resolution, frozen retained-fixture loading, Torch and transparent NumPy
references, predeclared parity cohorts, and SimulatorProtocol implementations.

The scientific evolution loop performs no NumPy/CuPy conversion or
host/device transfer. Explicit `device="cuda"` fails when CUDA is unavailable;
`device="auto"` records visible CPU fallback.

The [historical fixture producer](audits/README.md) runs an explicitly selected
NumPy/Numba implementation to capture tensor-level reference data. Reading the
stored reference or checking its hashes does not run that producer.
