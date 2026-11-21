# Reconstructing paper responses

A recommendation contains parameter coordinates. Drawing a physical response
at those coordinates can require a new simulation, even if the target curve
and the recommended parameters have already been saved.

| Program | Inputs and output | Execution boundary |
| --- | --- | --- |
| [plot_paper120_target.py](plot_paper120_target.py) | Uses one frozen Paper-120 target and the stored B40 recommendations to generate response curves. | Explicit rendering evaluates MUTACO and NPE recommendations with their paired assessment context. |
| [plot_paper120_b40_convergence.py](plot_paper120_b40_convergence.py) | Reads saved per-evaluation B40 diagnostics and can display convergence. Its replay modes regenerate scientific state. | Check the selected mode before execution. The ordinary stored-diagnostics path and explicit replay have different costs. |
| [Target-98 producer](../exploration/plot_paper120_target098_mutaco_only_v1.py) | Generates condition-grid curves and can write response arrays. | Creating the archive can execute three real physical condition-grid evaluations. |
| [Target-98 archive reader](../exploration/target098_response_archive.py) and [renderer](../exploration/render_paper120_target098_archive_v1.py) | Validate saved array shapes, ordering and provenance, then draw saved curves. | The archive renderer consumes retained arrays. These three target-98 sources stay at the paths recorded by the archive. |

New imports start here. Existing `experiments.paper.exploration` module and
CLI addresses remain available for their original operations. For figures
that only require saved values, see the [offline renderer catalog](../figures/README.md).
