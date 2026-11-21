"""Compatibility address for experiments.paper.studies.oscillator_physics_multi_target_sensitivity.study."""

import sys

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    runpy.run_module("experiments.paper.studies.oscillator_physics_multi_target_sensitivity.study", run_name="__main__")
else:
    from experiments.paper.studies.oscillator_physics_multi_target_sensitivity import study as _implementation

    sys.modules[__name__] = _implementation
