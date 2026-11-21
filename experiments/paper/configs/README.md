# Paper campaign declarations

[Graph inversion declarations](graph_inversion/) specify budgets, participating methods, preparation plans, assessment and recovery variants. Paired JSON and SHA-256 sidecars belong together. Preserve failed attempts and superseded plans when interpreting the provenance of a later accepted campaign. A declaration alone gives no permission to launch work.

The [physical input specification](../../graph_inversion/configs/paper120_v3_target_config.json) fixes the graph model used to generate target responses. The [cohort specification](../../graph_inversion/configs/paper120_v3_cohort_spec.json) supplies target dimensions, bounds, seed and design. These live with graph inversion; historical campaign references resolve both under their original relative names. [Frozen targets](../../graph_inversion/frozen_targets/README.md) are stored inputs rather than campaign outputs.
