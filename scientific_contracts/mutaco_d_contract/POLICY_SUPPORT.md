# Prompt 3 policy support

This table is generated from `mutaco.policies`; do not edit it independently.

| Family | Identifier | Classification | Surface | Direction | Formula | Required inputs | Simulator calls |
|---|---|---|---|---|---|---|---:|
| exploration | `operator_norm_uncertainty` | D default | `active_core` | maximize | `lambda_max(Sigma_latent_std)` | latent standardized posterior covariance | 0 |
| exploration | `trace_marginal_variance` | diagnostic-only | `diagnostic` | maximize | `trace(Sigma_latent_std)` | latent standardized posterior covariance | 0 |
| beta | `fixed_beta_v1` | D default | `active_core` | configure | `beta = configured finite positive scalar (default 2.0)` | finite positive beta | 0 |
| beta | `joint_chi_square_response_dimension_v1` | approved optional | `active_core` | configure | `beta = sqrt(chi_square_quantile(M, coverage))` | coverage in (0,1); declared response dimension M | 0 |
| recommendation | `final_posterior_standardized_l1` | D default | `active_core` | minimize | `sum_j |mu_final_std,j(theta_i) - target_std,j|` | evaluated theta; final latent posterior mean; standardized target | 0 |
| recommendation | `evaluated_posterior_expected_l2` | approved optional | `compatibility` | minimize | `||mu_final_std(theta_i)-target_std||_2^2 + trace(Sigma_final_latent_std(theta_i))` | evaluated theta; final latent posterior mean/covariance; standardized target | 0 |
| recommendation | `evaluated_posterior_mean_physical_l2` | approved optional | `compatibility` | minimize | `||mu_final_phys(theta_i)-target_phys||_2^2` | evaluated theta; final latent posterior mean; frozen transform; physical target | 0 |
| recommendation | `continuous_posterior_mean_physical_l2` | approved optional | `active_core` | minimize | `min over bounded theta of ||mu_final_phys(theta)-target_phys||_2^2` | bounded parameter specification; final latent posterior mean; frozen transform; physical target; coordinate seed | 0 |
| recommendation | `continuous_posterior_mean_physical_l1_mae` | approved optional | `active_core` | minimize | `min over bounded theta of mean_m |mu_final_phys,m(theta)-target_phys,m|` | bounded parameter specification; final latent posterior mean; frozen transform; physical target; coordinate seed | 0 |
| recommendation | `evaluated_best_observed_physical_mae_v1` | approved optional | `active_core` | minimize | `min over evaluated i of mean_m |y_observed_i,m-target_phys,m|` | evaluated theta; observed physical responses; physical target | 0 |
