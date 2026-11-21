"""Compatibility entry point; offline implementation lives in paper.figures."""
import sys as _sys

if __name__ == "__main__" and not __package__:
    # Preserve standalone-file use without importing the scientific paper facade.
    from pathlib import Path as _Path
    import runpy as _runpy

    _runpy.run_path(str(_Path(__file__).resolve().parents[1] / "figures" / "plot_paper120_terminal_domain_ablation_v1.py"), run_name="__main__")
else:
    from ..figures import plot_paper120_terminal_domain_ablation_v1 as _implementation

    if __name__ == "__main__":
        _implementation._cli()
    else:
        _sys.modules[__name__] = _implementation
