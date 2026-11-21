# Target-98 oscillator-physics post-analysis

This isolated namespace implements the approved post-analysis of the frozen
Paper-120 target at position 98. The archive-only gauge contrast and response
trajectories consume the byte-hash-verified 20-component target response and
the illustrated B40 MUTACO record's held-out `assessment_response`, without
invoking the simulator. Both series retain the ordering quantity first
(`omega`, then `Z_magnitude`), force-major within each quantity, and gauge
`(0, pi)` within each force.

The prospective sensitivity design is a four-parameter central difference with
normalized half-step `h=0.05`. Perturbation points are constructed in Torch
`float32` by normalizing the audited `theta_star`, adding signed `h` to one
coordinate, and mapping back to physical space. Exactly eight points are
declared: minus then plus for each parameter. Center reruns, extra replicates,
additional step sizes, and retries are forbidden. The response normalization
uses fixed physical scales `s_omega=5` and `s_M=1`.

All eight points use one identical `SimulationContext`, including one fixed
evaluation seed. Because the approved simulator derives quenched disorder,
initial state, and thermal-noise coordinates from the configuration identity
and evaluation seed rather than theta, this is full-stream common random
numbers across all eight perturbations. The worker verifies exact seed-manifest
equality after every successful call.

The worker is append-only and fail-closed. It writes each point intent before
the simulator invocation, permits one invocation per point, stops on the first
failure, never retries, and only emits the Jacobian after eight successful
outcomes. Execution requires both the exact CLI authorization phrase and
`MUTACO_OSCILLATOR_EIGHT_CALLS_AUTHORIZED=1`. The TRUBA wrapper additionally
requires `--submit`; its default and lifecycle-routed behavior is validation
only. No command in this package automatically submits or retries a job.

The JSON schemas define point outcomes and the final 20-by-4 Jacobians with
respect to normalized and physical coordinates, plus the response-scaled
normalized Jacobian. The original
[zero-call response analysis](../../../../results/paper/oscillator_physics_post_analysis/target098_v1)
and the subsequent [eight-point sensitivity result](../../../../results/paper/oscillator_physics_post_analysis/target098_sensitivity_job6352257)
are separately preserved. The latter records the historical execution and its
call accounting; reviewing the saved Jacobian requires no new evaluations.
