"""Compatibility address for experiments.paper.studies.analytic_gp_landscape.run."""

import sys

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    runpy.run_module("experiments.paper.studies.analytic_gp_landscape.run", run_name="__main__")
else:
    from experiments.paper.studies.analytic_gp_landscape import run as _implementation

    sys.modules[__name__] = _implementation
