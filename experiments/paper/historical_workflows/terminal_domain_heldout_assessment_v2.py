"""Compatibility address for experiments.paper.common.terminal_domain_heldout_assessment_v2."""

import sys

if __name__ == "__main__":
    if not any(option in sys.argv[1:] for option in ("-h", "--help")):
        raise SystemExit(
            "This terminal assessment binds source and data at their original paths. "
            "Run its native command in the isolated historical checkout described in "
            "experiments/paper/historical_workflows/README.md."
        )
    import runpy
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    runpy.run_module("experiments.paper.common.terminal_domain_heldout_assessment_v2", run_name="__main__")
else:
    from experiments.paper.common import terminal_domain_heldout_assessment_v2 as _implementation

    sys.modules[__name__] = _implementation
