# Historical replay divergence investigation — 24 October 2026

**Decision D: a reference artifact is still required. The core refactor remains
blocked.** The first confirmed numerical difference is the frozen float32
response mean after the 16 initial observations. Recovering that mean is not
sufficient to recover the historical first adaptive candidate. Source,
configuration and adapter comparisons did not identify a discrepancy. Local
repeatability does not establish that macOS caused the remaining difference.

This is a zero-simulator engineering investigation, not a new benchmark or a
change to an accepted numerical policy. The starting source is
`ccc45090146bdee3f632e7c1084df77daeff9ddb` on `ege/repro-refactor`, with canonical
origin `https://github.com/aydinCkeser/MUTACO.git`. No GitHub operation was made.

## What first differs

The B40 target-16 record `/records/48` and B100 target-0 record `/records/0`
both supply exactly the regenerated initial physical parameters `[16,4]` and
physical response array `[16,20]`. Their JSON numbers are exactly representable
in float32. Shapes, row/component order, observation identities and effective
physical simulation contexts agree. The arrays are bitwise equal after loading.
Both archives and all member/manifest/sidecar hashes pass their existing guards.

The frozen transform identity hashes canonical semantic values: schema,
response names, observation IDs, means, empirical/effective scales, floor flags,
floor, degrees of freedom and dtype. It excludes device, environment paths and
runtime provenance; the standalone standardization-policy label is also omitted
for historical identity compatibility. A different hash alone could mean either
different data or different identifiers. It does not by itself prove roundoff.

Here the entire archived transform hash is recovered by changing **only** the
mean vector to `physical_y.T.contiguous().mean(dim=1)` while retaining the native
`std(dim=0, correction=1)` and every other semantic field. Native production uses
`physical_y.mean(dim=0)`. This is a hash-verified reconstruction of the historical
semantic values, not a directly archived mean array or evidence that historical
source used a transpose. That source has the same reduction expression as today.
Changing the tensor layout changes the float32 reduction order in this local
diagnostic. The native hash is internally self-consistent; this is not a JSON
list/tuple or checksum representation problem.

| Quantity | B40 target 16 | B100 target 0 |
|---|---:|---:|
| Mean components that differ | 9 / 20 | 7 / 20 |
| Standardized training elements that differ | 133 / 320 | 101 / 320 |
| Maximum absolute standardized difference | 6.198883056640625e-6 | 1.4841556549072266e-5 |
| Scale floor | 1e-12 | 1e-12 |
| Floor activations | 0 | 0 |
| Standard deviation degrees of freedom | 1 | 1 |

Example at component 10, `Z_magnitude(F=0.0,gauge=0)` (zero-based indices):

| Value | B40 row 2 | B100 row 7 |
|---|---:|---:|
| Archived and replayed physical response | 0.9500942230224609 | 0.972068190574646 |
| Current mean | 0.9648385643959045 | 0.9652392864227295 |
| Hash-verified historical mean | 0.9648386240005493 | 0.965239405632019 |
| Common effective scale | 0.009714343585073948 | 0.008051114156842232 |
| Current standardized value | -1.5177907943725586 | 0.8481937050819397 |
| Reconstructed historical standardized value | -1.5177969932556152 | 0.8481788635253906 |

The first candidate rejection remains evaluation 17. Maximum coordinate errors
are `0.0026917457580566406` (B40) and `1.0311603546142578e-5` (B100).
The adapter returns no observation for these differing candidates.

## Historical source and production binding

Read-only Git object comparisons used the actual campaign source commits:
B40 `bd83de6ec01f0f7e2241ee498960a4d493d601e6` and B100
`649235bcb7e8773f5f4b0425ad111314d7d3b1f9`. For each, all 23 selected paths/trees
equal the current source: all 47 core files, all nine graph-package files, the
primary benchmark tree, real worker/callback/binding, policies, declaration,
profile, configuration, target, cohort manifest and reference environment lock.
The compact JSON evidence records the object IDs. Changes in adjacent imports
add SBI method support or evidence analysis, without a changed MUTACO execution
path or global numerical setting identified by this review.

The matching path normalizes parameters, uses point-major flattening and task
indices 0–19, constructs rank-one `MultiTaskGP` with no outcome transform and
task-diagonal noise, retains its noise prior and variance floor, derives named
fit seeds inside RNG isolation, and runs the existing SciPy GP optimizer for at
most 12 iterations. Acquisition retains its 64-point scrambled Sobol pool,
joint-chi-square beta `5.604501123581913`, alpha `0.1`, operator-norm confidence
radius, four-start maximum and eight refinement iterations. No MML/prior
comparison or production fitting-policy change was made.

Seven archived declaration/configuration/policy/target checks pass for each
record. A separate offline invocation through the real production
`GraphPrimaryIntegrationSmokeDriver.run_method` agrees exactly with the replay
adapter on both candidates, transform/model/training IDs, fit seed and candidate
ID. All 16 input rows and effective contexts agree. This excludes the replay's
duplicated invocation binding as an explanation in the tested environment.

## Reconstructed fit and acquisition

The new `replay_divergence.py` wraps the executed production operations and
returns their original results unchanged. It stores actual standardized
training/target data, cold initialization, effective seeds/configuration,
fitted parameters/noise/task covariance, actual raw-pool posteriors and all
64 screening scores/margins, restart starts/score calls/results and ranking.
Nonfinite infeasible scores are explicitly tagged in strict JSON. One extra
posterior call over four points occurs after replay and is labelled outside production call
accounting. The float64 covariance mirror is labelled separately from the
float32 fitted model; it is not a second fit or recovery attempt.

| Reconstructed quantity | B40 | B100 |
|---|---:|---:|
| First fit seed | 991989318286734224 | 8084209976490747184 |
| Actual fit attempts | 1 | 1 |
| GP iterations | 12 | 12 |
| Initial objective | 0.9867059588432312 | 1.4625073671340942 |
| Final objective | 0.5109604597091675 | 0.6518608927726746 |
| Fitted noise range | 0.0030407924–2.18165445 | 0.0033400946–0.98442566 |
| Finite strict-interior raw points | 1 | 2 |
| Raw start indices, in rank order | 48 | 15, 51 |

The runner accepts each fit with finite objective and valid posterior; the
optimizer stops at its configured iteration cap, rather than claiming
convergence. No recovery fit is taken. Both acquisitions choose the historical
`barrier / strict_interior_available / refined` branch. Raw index order matches
history; scores and refined candidates differ. For example, B40 raw index 48
scores `3.787945508956909` locally versus archived `3.7827680110931396`.

A separate controlled reconstruction re-fits and proposes with the recovered
historical means, original scales and otherwise identical production routines,
seeds/configurations. It is never inserted into strict replay. Cold GP
parameters remain exactly equal to the original local initialization, but final
parameters change. Candidate errors against history remain
`0.002180755138397217` (B40) and `1.0415911674499512e-5` (B100). Thus the verified
mean difference contributes sensitivity but is not a complete explanation.
Historical cold/fitted parameters and raw posterior tensors are not available
in these trajectory exports; the reconstructed values must not be presented as
historical GP state.

The registries' separate 240 final-GP exact-state replays concern older
MUTACO–NPE–SNPE campaigns and different source commits. They cannot supply these
scalar-primary first fits. Both scalar-primary manifests explicitly exclude
raw posterior artifacts; no applicable committed tensor checkpoint was found.
Original worker roots remain in the private raw-evidence backup; operational
account names and absolute directories are omitted from this shareable report.
Likely record locations are `worker-16/targets/target-016/mutaco.json` (B40) and
`worker-00/targets/target-000/mutaco.json` (B100); no separately named first-GP
checkpoint is indexed. Their existence or additional external contents were
not checked remotely in this local task.

## macOS and independent NEI result

CPU float32 fitting/means and the existing float64 covariance path use Python
3.11.15, Torch 2.12.1, BoTorch 0.18.1, GPyTorch 1.15.2, NumPy 2.4.6 and SciPy
1.17.1 on macOS 26.5.1 ARM64. Torch and NumPy report Apple Accelerate BLAS/LAPACK;
Torch uses OpenMP. Thread environment overrides are unset. Historical proposal
metadata reports Python 3.11.15, Torch 2.12.1+cpu, BoTorch 0.18.1 and GPyTorch
1.15.2, CPU float32, but not the exact CPU ISA or linked BLAS build.

Eight fresh strict replay processes cover both targets, each twice at 20/1
intra/inter-op threads and once at 1/1 and 4/1. Every candidate, transform and
fitted parameter is exactly equal within each target. Other scientific
diagnostics agree after excluding timers and thread metadata. No production
thread default was changed. This rules out observed local nondeterminism and
these thread choices; it does not identify a Mac-versus-x86 cause.

The existing `test_baseline_factorization_promotes_without_jitter` fails both
on the unmodified `aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3` baseline and the
current tree under the same interpreter/backend. Each focused run gives one
failure, one passing indefinite-input control and four deselections. Test and
implementation blobs are identical. The fixture is exactly positive
semidefinite, rank four, determinant zero, with null vector `[1,-4,6,-4,1]` and
exact LDL diagonal `[1,273/4096,101/4368,9/1616,0]`. All entries are exactly
representable before promotion. Cholesky fails at minor five in both float32
and float64; promotion cannot make the mathematical matrix strictly positive
definite. This is a pre-existing singular success fixture with possible
roundoff portability, not a demonstrated implementation defect or explanation
for the MUTACO replay. Neither the test nor the factorization policy changed.

## Next bounded comparison and validation

The smallest useful reference is **B40's first 16 archived observations, one
GP fit and one proposal on the original Linux x86-64 environment**, followed by
the equivalent B100 capture. No physical simulator, scheduler job or full
campaign is needed. Use this same diagnostic file in an isolated source copy
and the reference environment; preserve its source fingerprint and dirty patch.
Through the approved launcher/environment, run these module arguments:

For a complete Linux setup, offline source-transfer procedure, deterministic
reference format, safety checks and return instructions, use
[the Linux reference-capture procedure](linux_reference_capture.md).

```sh
-m experiments.paper.exploration.replay_divergence --budget 40 --target 16 --output /tmp/mutaco-reference-b40.json --torch-intraop-threads 20 --torch-interop-threads 1
-m experiments.paper.exploration.replay_divergence --budget 100 --target 0 --output /tmp/mutaco-reference-b100.json --torch-intraop-threads 20 --torch-interop-threads 1
```

Return each JSON and stdout/stderr log plus the executed source commit/patch
and environment freeze. The capture includes backend/build/thread metadata,
input and tensor hashes/values. It stops before returning response 17 even if
the first proposal exactly matches; a successful diagnostic command therefore
does not certify a passing golden replay. Early production failures preserve a
partial packet. Compare means/scales first, then cold GP parameters, fitted
parameters, raw posteriors/scores and refinement steps. If the x86 execution
also differs from the archive, obtain an original first-fit snapshot/log;
nominal package versions alone cannot reconstruct an absent historical state.

Original diagnostics and raw validation evidence remain in a verified private
backup. The derived record is
benchmarks/reproducibility/shareable_milestone_v1/macos-diagnostic-summary.json.
It retains numerical findings and source identities while omitting operational
locations. Original and derived SHA-256 values are mapped separately.


Validation commands/results are recorded there. Production source, numerical
policies, candidate equality, seeds, dtype/device, budgets, call accounting,
archives, Windows workflows and manuscript remain unchanged. No observation
was returned for a differing candidate, ambient RNG was preserved, and all
physical simulator tripwires remained at zero. Full trajectory, duplicate
recovery and terminal replay remain unqualified. This is a useful local
diagnostic milestone to review before deciding on a GitHub update; it does not
unblock scientific refactoring or establish portable regression tolerances.
