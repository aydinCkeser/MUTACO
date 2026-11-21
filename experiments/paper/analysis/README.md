# Offline analysis of recorded experiments

Use [aggregate.py](aggregate.py) to select persisted method records and compute
paired summaries, [timing_report.py](timing_report.py) to interpret clocks, and
[cross_budget_synthesis.py](cross_budget_synthesis.py) for descriptive B40/B100
comparisons. A change of budget is a change of campaign; use the population,
pairing and denominator defined by the declared analysis.

The [evidence checker](evidence_analysis_cli.py) and [cross-budget checker](cross_budget_synthesis_cli.py)
can compare the reported output with archived evidence. Loading saved values
does not run the method that generated them. Materializing a report writes
files. The B40 check needs the corresponding compact export. The
[source-bound evidence adapter](../common/evidence_analysis.py) stays at the
literal path recorded in the B100 preparation manifest. The [figures](../figures/README.md)
guide identifies operations that only render saved arrays.
