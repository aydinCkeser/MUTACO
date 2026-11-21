"""Compatibility address for experiments.paper.exploration.render_paper120_target098_archive_v1."""

import sys

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    runpy.run_module("experiments.paper.exploration.render_paper120_target098_archive_v1", run_name="__main__")
else:
    from experiments.paper.exploration import render_paper120_target098_archive_v1 as _implementation

    sys.modules[__name__] = _implementation
