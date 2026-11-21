# Posterior numerics support

| Identifier | Classification | Surface | Behavior | Required fit dtype | Additional fit attempts | Simulator calls |
|---|---|---|---|---|---:|---:|
| `native_float32` | diagnostic-only | `diagnostic` | Native fitted-model dtype for posterior mean and covariance | float32 or float64 | 0 | 0 |
| `float32_fit_float64_covariance` | D default | `active_core` | Native float32 fit and mean; cached float64 covariance construction, strict validation, then validated float32 return | float32 | 0 | 0 |
| `full_float64_diagnostic` | diagnostic-only | `diagnostic` | Float64-fitted diagnostic model with native float64 posterior | float64 | 0 | 0 |
