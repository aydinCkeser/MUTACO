# MUTACO policy surface and compatibility matrix

This table is generated from `mutaco.policies.surface`; do not edit it independently.
The surface label governs new configuration. `Evidence shim` marks identifiers required by accepted diagnostic or result evidence.

| Identifier | Family | Surface | Ordinary graph | Evidence shim | Owner |
|---|---|---|---:|---:|---|
| `scrambled_sobol` | initial_and_candidate_design | `active_core` | yes | no | `mutaco` |
| `caller_supplied` | initial_design | `active_core` | no | no | `mutaco` |
| `task_diagonal_gaussian` | likelihood | `active_core` | yes | no | `mutaco` |
| `shared_isotropic_gaussian` | likelihood | `compatibility` | no | no | `mutaco` |
| `cold_only` | gp_fit | `active_core` | no | no | `mutaco` |
| `graph_m20_sequential_recovery_v1` | gp_fit | `active_specialized_core` | yes | no | `mutaco` |
| `float32_fit_float64_covariance` | posterior_numerics | `active_core` | yes | no | `mutaco` |
| `operator_norm_radius_v1` | confidence_geometry | `active_core` | yes | no | `mutaco` |
| `operator_norm_uncertainty` | exploration | `active_core` | yes | no | `mutaco` |
| `fixed_beta_v1` | beta | `active_core` | yes | no | `mutaco` |
| `joint_chi_square_response_dimension_v1` | beta | `active_core` | yes | no | `mutaco` |
| `posterior_uncertainty_v1` | infeasible_fallback | `active_core` | yes | no | `mutaco` |
| `reject_exact_duplicate_v1` | duplicate | `active_core` | no | no | `mutaco` |
| `best_ranked_unevaluated_raw_v1` | duplicate | `active_core` | yes | no | `mutaco` |
| `final_posterior_standardized_l1` | recommendation | `active_core` | no | no | `mutaco` |
| `continuous_posterior_mean_physical_l2` | recommendation | `active_core` | yes | no | `mutaco` |
| `continuous_posterior_mean_physical_l1_mae` | recommendation | `active_core` | yes | no | `mutaco` |
| `evaluated_best_observed_physical_mae_v1` | recommendation | `active_core` | yes | no | `mutaco` |
| `composite_mc_nei_physical_mae_v1` | acquisition | `active_core` | yes | no | `mutaco` |
| `best_ranked_finite_unevaluated_raw_v1` | acquisition_recovery | `active_core` | yes | no | `mutaco` |
| `evaluated_posterior_expected_l2` | recommendation | `compatibility` | no | no | `mutaco` |
| `evaluated_posterior_mean_physical_l2` | recommendation | `compatibility` | no | no | `mutaco` |
| `independent` | evaluation_randomness | `active_core` | yes | no | `mutaco` |
| `fresh_held_out` | assessment | `active_core` | yes | no | `mutaco` |
| `high_compute` | proposal_effort_alias | `compatibility` | no | no | `graph_inversion` |
| `iid_uniform` | initial_and_candidate_design | `diagnostic` | no | no | `mutaco` |
| `trace_marginal_variance` | exploration | `diagnostic` | no | no | `mutaco` |
| `native_float32` | posterior_numerics | `diagnostic` | no | yes | `mutaco` |
| `full_float64_diagnostic` | posterior_numerics | `diagnostic` | no | yes | `mutaco` |
| `trace_effective_dimension_radius_diagnostic_v1` | confidence_geometry | `diagnostic` | no | yes | `mutaco` |
| `minimum_normalized_target_miss_diagnostic_v1` | infeasible_fallback | `diagnostic` | no | yes | `mutaco` |
| `paired_coordinate_diagnostic_v1` | evaluation_randomness | `diagnostic` | no | yes | `mutaco` |
| `fixed_full_stream_crn_diagnostic_v1` | evaluation_randomness | `diagnostic` | no | yes | `mutaco` |
| `same_target_realization_assessment` | assessment | `diagnostic` | no | yes | `graph_inversion` |
| `continuous_expected_physical_l2_diagnostic` | recommendation | `diagnostic` | no | yes | `graph_inversion` |
| `legacy_full_stream_crn_diagnostic` | search_randomness | `diagnostic` | no | yes | `graph_inversion` |
| `legacy_scale_search_effort_diagnostic` | proposal_effort | `diagnostic` | no | yes | `graph_inversion` |
| `continuous_posterior_mean_physical_l1_mae_diagnostic_v1` | recommendation_alias | `diagnostic` | no | yes | `graph_policy_canary` |
| `gp_recovery_stages` | mechanism | `internal` | no | no | `mutaco` |
| `covariance_promotion_and_validation` | mechanism | `internal` | no | no | `mutaco` |
| `duplicate_detection_and_raw_ranking` | mechanism | `internal` | no | no | `mutaco` |
| `continuous_terminal_optimizer` | mechanism | `internal` | no | no | `mutaco` |
| `optimizer_numerical_controls` | mechanism | `internal` | no | no | `mutaco` |
| `execution_and_thread_controls` | mechanism | `internal` | no | no | `mutaco` |
