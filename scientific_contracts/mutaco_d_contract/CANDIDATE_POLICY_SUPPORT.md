# Prompt 4B candidate-policy support

This table is generated from `mutaco.acquisition`; do not edit it independently.

| Identifier | Classification | Surface | Behavior | Generation device |
|---|---|---|---|---|
| `scrambled_sobol` | D default | `active_core` | Seeded scrambled Sobol points mapped from [0,1]^P to physical bounds | CPU, then one recorded transfer |
| `iid_uniform` | diagnostic-only | `diagnostic` | Seeded IID uniform points mapped from [0,1]^P to physical bounds | CPU, then one recorded transfer |

Latin hypercube and caller-supplied raw pools are rejected as unsupported.
