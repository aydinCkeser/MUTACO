"""Compatibility address for experiments.paper.common.exact_terminal_bootstrap."""

import sys

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    runpy.run_module("experiments.paper.common.exact_terminal_bootstrap", run_name="__main__")
else:
    from experiments.paper.common import exact_terminal_bootstrap as _implementation

    sys.modules[__name__] = _implementation
