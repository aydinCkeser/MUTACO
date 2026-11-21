"""Compatibility address for experiments.paper.response_reconstruction.plot_paper120_b40_convergence."""

import sys

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    runpy.run_module("experiments.paper.response_reconstruction.plot_paper120_b40_convergence", run_name="__main__")
else:
    from experiments.paper.response_reconstruction import plot_paper120_b40_convergence as _implementation

    sys.modules[__name__] = _implementation
