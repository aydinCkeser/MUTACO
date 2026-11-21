# Prompt 4A initial-design support

This table is generated from `mutaco.design`; do not edit it independently.

| Identifier | Classification | Surface | Behavior |
|---|---|---|---|
| `scrambled_sobol` | D default | `active_core` | Seeded scrambled Sobol points inside bounds |
| `caller_supplied` | approved optional | `active_core` | Strictly validated caller points in preserved order |
| `iid_uniform` | diagnostic-only | `diagnostic` | Seeded IID uniform points inside bounds |

Latin hypercube sampling is deferred and rejected as unsupported.
