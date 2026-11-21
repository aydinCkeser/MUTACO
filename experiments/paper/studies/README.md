# Manuscript studies

A study directory contains its declaration, scientific question and entry
points. Execution and stored results have different owners. Keep a study's
source declaration when interpreting any result produced from it.

| Study | Scientific purpose and inputs | Outputs and execution |
| --- | --- | --- |
| [Graph Paper-120](graph_paper120) | Compares methods on frozen graph targets using budget-specific plans, analysis specifications and ingestion records. | Paper Lab reads these declarations. Refer to the [graph experiment](../../graph_inversion/README.md) for physical callbacks and the [workflow](../workflow/README.md) for reporting and promotion. |
| [Handwritten analytic](analytic_handwritten/README.md) | Compares methods on controlled functions using a declared cohort. | `run.py` reads the campaign; explicit execution requires a selection and confirmation. The tracked selection remains unexecuted. |
| [Frozen GP landscape](analytic_gp_landscape/README.md) | Uses a fixed analytic GP realization as the forward function. The declared frozen-state identity must match. | `run.py` describes the prepared campaign by default. An authorized execution can perform target matching and optional SBI work. |
| [Target-98 oscillator sensitivity](oscillator_physics_post_analysis/README.md) | Examines gauge contrasts and local physical sensitivity for one target. Archived curves and the stored held-out method record support zero-call analysis. | The prospective central-difference worker has an eight-call gate. The raw worker record required by the zero-call reconstruction may be absent from a checkout. |
| [Maximin-12 oscillator sensitivity](oscillator_physics_multi_target_sensitivity/README.md) | Selects twelve separated targets in parameter space and evaluates symmetric perturbations. | The declared worker has a 96-call ceiling. The accepted result bundle includes its own figure producer and provenance. |

See [scientific results](../../../results/) for stored packages. Historical
config paths, module imports and TRUBA launcher paths remain available. Local
study tests and input schemas accompany their owning study. Reading one of
these guides or importing its module does not authorize execution.
