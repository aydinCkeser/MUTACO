# Paper-120 multi-target oscillator local sensitivity

This prospective study asks whether the local anisotropy observed at the
separately illustrative target 98 is characteristic of the frozen Paper-120
oscillator cohort. Target selection uses only the 120 frozen `theta_star`
vectors. Coordinates are normalized by the `p4-paper-baseline` inference
bounds, then 12 targets are selected by deterministic greedy farthest-point
sampling: the first is closest to the unit-cube center, each later target
maximizes its distance to the selected set, and exact ties choose the smaller
target index. Performance, response, SBI, and prior sensitivity evidence are
not selection inputs.

For target `t` and coordinate `a`, with normalized center `z_ta`, the symmetric
central-difference half-step is

`h_ta = min(0.05, 0.9 * min(z_ta, 1-z_ta))`.

The two points are constructed in Torch `float32` at `z_ta +/- h_ta` and mapped
back to physical coordinates. The declaration records the normalized center,
nearest-boundary distance, nominal half-step, persisted minus/plus vectors,
actual persisted physical denominator, and reduction flag. Validation fails
closed if any point is outside the inference bounds. Derivatives must use the
persisted physical plus-minus denominator, never `2*h_ta` or another nominal
width. Each target has four symmetric pairs, no center, no one-sided stencil,
no second step size, and no extra replicate.

Each target has a distinct SHA-256-addressed evaluation seed and scientific run
namespace. All eight points for that target reuse one `SimulationContext`, so
quenched disorder, initial state, and thermal noise are full-stream common
random numbers. Contexts are independent between targets. The target-98
diagnostic seed is not reused.

Execution is append-only and fail-closed: one non-array job processes the 12
targets sequentially, writes an intent before each call, stops on the first
failure, never retries, and enforces both 8 calls per target and 96 calls
globally. Execution requires the exact CLI phrase and
`MUTACO_OSCILLATOR_96_CALLS_AUTHORIZED=1`; submission separately requires
`truba/submit.sh --submit`. Validate-only is the default lifecycle surface and
cannot submit.

The eventual per-target result contains physical `J`, normalized-coordinate
`D`, dimensionless `S` (`s_omega=5`, `s_M=1`), column norms, singular values,
condition number, weakest right-singular vector, column cosines, and
Phi=pi-minus-Phi=0 gauge-contrast sensitivities.

Runtime planning uses completed job 6352257: eight calls took 52 seconds on one
Barbun node with 20 CPUs and 64 GB. Multiplying the observed job elapsed time
by 12 gives 624 seconds (10:24) as a conservative central sequential estimate;
that observation combines worker/process overhead with simulator time and does
not include queue delay. A planning interval of 480--900 seconds (8--15
minutes) allows for amortized process startup at the lower end and
target/node-dependent variation at the upper end; these endpoints are
extrapolations, not additional measurements. The preregistered request is one
node, one task, 20 CPUs, 64 GB, and 30 minutes.

The [accepted twelve-target result](../../../../results/paper/oscillator_physics_multi_target_sensitivity/paper120_maximin12_job6353452)
contains the paired perturbations, response arrays, study-level summaries and
the associated figure producer. The producer uses its original package depth
to locate the study inputs. Its native directory therefore remains intact
behind the result index.
