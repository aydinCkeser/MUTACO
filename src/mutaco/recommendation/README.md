# Terminal recommendation

The manuscript's Step 4 calls [continuous physical MAE](physical_mae_terminal_recommendation.py) on
the final fitted GP, the frozen response transform, the physical target, the
admissible parameter bounds and the run's seed coordinates. The score is the
mean absolute difference between the inverse-transformed posterior mean and
the target. The [continuous optimizer](_continuous_terminal_optimizer.py) searches this bounded
domain. The result records the selected parameter, score and search diagnostics.
Neither scoring nor recommendation calls the physical simulator.

[Shared contracts](recommendation_contracts.py) describe ordered evaluated
designs, the mean-only posterior interface and the existing continuous-policy
failure type. [Squared physical L2](comparators/physical_squared_l2_recommendation.py) and
[evaluated posterior policies](evaluated_design_recommendation.py) provide other registered choices.
The [observed physical-MAE comparator](../comparators/observed_physical_mae_recommendation.py)
uses recorded responses directly and can return only an evaluated design.

The older import names `physical_mae`, `_continuous`, `physical_l2` and
`policies` resolve to their respective implementation modules. Source readers can
open the implementation links above or inspect the imported module. Exact
historical terminal reproduction reads Git blobs from the recorded original
commits and substitutes its declared optimizer in memory.
