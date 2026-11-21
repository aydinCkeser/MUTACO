# B100 scientific regression reference

B100 target 0, record /records/0, is independent of B40. Both the original
aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3 implementation and accepted frozen-response
refactor 242fac08c1c3dc0041651a731c20ed228969ed6a passed all 100 stored observations.
The archive's historical campaign source is
649235bcb7e8773f5f4b0425ad111314d7d3b1f9. Its SHA-256 remains
e30befedd72b85bd0eda8fe1d7e33ed6ef10acd0c880b1bd6aca60865e4a49c7.

Each run consumed 16 initial and 84 adaptive observations, recording 85 fit/update
events, 84 proposals and one terminal recommendation. Candidate/context equality,
RNG state, duplicate recovery, terminal policy and accounting passed unchanged.
The first-fit/conditional checkpoint mathematical hash is
06338d126708d62fdb3967db3d1603eb1529270ec4c6704322b621cd6b4fa150.
The full replay mathematical hash is
c97b62ec714905ff97037561aad8ac3aef638abf9e189a926d6306844d3854c2.
Both hashes match between baseline and refactor, with no numerical tolerance.

These are derived files, not byte-identical copies of raw packets. The manifest
and raw-evidence index map original raw SHA-256 values to derived-file SHA-256
and mathematical hashes. Tensor bytes and mathematical values are unchanged;
only original predeclared timer/execution exclusions apply to replay payloads.
All new GP states are reconstructed. Conditional final-data cold-fit values are
labelled separately and are not the original final warm-start historical GP.

Raw logs, exact environment fingerprints, source snapshots and original worker
checksums remain in a private evidence collection outside this repository.
Sanitized environment summaries do not replace original acceptance packets.

Job 6853779 ran for 598 seconds under one 20-CPU allocation. Baseline tests:
169 passed, 8 skipped, 1 known NEI fixture failure; refactor tests:
175 passed, 8 skipped, the same failure. Slurm FAILED/exit1 is retained honestly
while the independent scientific gate passed. No physical simulator attempt
or evaluation occurred; no further job was submitted.

Requested memory was 8 GiB. Slurm reported 13927916 KiB peak RSS (about 13.3 GiB),
exceeding the request. No inference about site memory enforcement is made.
Do not reuse 8 GiB as a validated B100 peak bound: review a request of at least
16 GiB before the next B100 job. This assignment authorizes no additional job.

Verify portable file bytes here with:

    shasum -a 256 -c SHA256SUMS

From the repository root, the approved local launcher can verify derived
integrity and, with explicit private inputs, exact original-file bindings:

    infrastructure/local/run_mutaco.sh -m infrastructure.truba.b100_evidence verify \
      --directory benchmarks/reproducibility/b100_regression_v1 \
      --raw-source "$PRIVATE_B100_EVIDENCE"

The source parameters, GP policies, priors, acquisition/terminal logic, seeds,
precision and B40 evidence were not changed. The next proposed code improvement
is the option-validation portion of the pre-run try block; capability capture,
seed-deriver construction, exception/status selection, zero-call accounting and
event boundaries must remain intact. It has not been implemented.
