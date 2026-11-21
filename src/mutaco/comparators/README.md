# Comparison methods and diagnostic selectors

[Composite Monte Carlo NEI](composite_mc_nei.py) is the retained multi-output
comparison method. It samples the **joint latent posterior** of previously
evaluated inputs and a candidate. It maps each sampled response back to physical
units before comparing target MAE. Its seeds identify the raw candidates and
shared Monte Carlo draws. Call `propose_composite_mc_nei` with a joint posterior
provider, evaluated inputs, the frozen response transform and physical target.
The result includes a proposed input, raw scores and factorization and restart
diagnostics. The module makes no simulator calls.

The manuscript's principal Step 3 uses [confidence geometry](../acquisition/confidence_geometry.py),
[barrier and uncertainty objectives](../acquisition/acquisition_objectives.py)
and [candidate proposal](../acquisition/candidate_proposal.py).
The paper's [scalar qLogNEI comparator](../../../experiments/benchmarking/methods/scalar_qlognei.py)
uses a scalar GP. Keep those three methods distinct when interpreting saved results.

[Diagnostic policy aliases](diagnostic_policy_aliases.py) names existing option values
for diagnostic experiments. It preserves their registered identifiers. The
`mutaco.diagnostics` and `mutaco.acquisition.nei` imports still work, including
for stored Python objects. Run configurations can continue to use their original
policy IDs.

[Observed physical MAE](observed_physical_mae_recommendation.py) selects the
earliest evaluated design minimizing the average absolute difference between
its recorded physical response and the target. It does not query the final GP.
