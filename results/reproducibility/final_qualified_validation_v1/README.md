# Final qualified scientific validation

The reorganized source at `4f563439a6d4445a11d97cd6d81ba784267f729d`
passed the existing qualified TRUBA CPU workflow. This is a validation record;
it does not change the scientific implementation or reference values.

- The full [531-case selection](test-selection.json) passed: 523 passed,
  eight unchanged skips, zero failures/errors/deselections. The skips are seven
  CUDA cases and the existing unavailable expanded-fetch check.
- B40 target 16 (`/records/48`) matched all 40 archived evaluations; B100
  target 0 (`/records/0`) matched all 100. Full histories, checkpoint tensors,
  frozen transforms, seeds/RNG, duplicate recovery, terminal recommendation,
  accounting and 50/170 fit-progress events met the existing exact contract.
- Job 6875156 completed with exit `0:0` in 6m34s: 20 CPUs, 20 GiB requested,
  20-minute limit, no GPUs; reported batch peak RSS was about 10.7 GiB.
- No physical backend was attempted, no original baseline was rerun, and no
  archive, checkpoint reference, source/test policy, tolerance or figure changed.

[manifest.json](manifest.json) contains the exact comparison verdicts/hashes,
resource use and path-sanitized environment summary. The original Linux x86-64
Python 3.11.17 runtime was reused without installing/upgrading dependencies.
The [validation contract](../../../scientific/refactor_validation/README.md)
and [input profile](../../../infrastructure/truba/gp_optimizer_diagnostics_reference_inputs_v2.json)
remain unchanged. Raw captures were independently rechecked after collection. Four display labels
contain private path parameters and are redacted here; individual SHA-256 values
bind all 531 complete raw test identities in order. Acceptance compared unchanged
raw identities, which remain available in the private evidence.

[raw-evidence-index.json](raw-evidence-index.json) records private raw-file hashes
and preserved-input bindings. Raw captures, full environment paths, logs and
source delta remain outside Git. These summaries are derived, not byte-identical
raw acceptance packets; [SHA256SUMS](SHA256SUMS) seals only the portable files.
GP checkpoints are reconstructed original-baseline references, not unavailable
historical campaign GP snapshots. B40/B100 does not validate SBI experiments,
physical kernels, GPU parity, other-platform installation or TeX figure rendering.
This supports private technical review, not external release approval.
