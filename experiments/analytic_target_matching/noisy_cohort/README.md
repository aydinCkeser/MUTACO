# Noisy handwritten target cohort

Run a configurable collection of independent noisy MUTACO inversions:

```powershell
python -m experiments.analytic_target_matching.noisy_cohort.run
```

Both `run.py` and `plot.py` also run directly from PyCharm without arguments.
The full configuration uses 100 targets; the smoke configuration uses 4.
Figures are regenerated without simulator or MUTACO calls:

```powershell
python -m experiments.analytic_target_matching.noisy_cohort.plot
```
