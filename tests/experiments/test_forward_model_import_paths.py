"""Verify source-checkout module identity without constructing a simulator."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = "mutaco_analytic"
EXPERIMENT = "experiments.analytic_target_matching.forward_models"
PROBE = """
import importlib
import pickle
import sys
from pathlib import Path

first, second, owner = sys.argv[1:]
a, b = (importlib.import_module(name) for name in (first, second))
assert a is b
assert Path(a.__file__).resolve().parent == Path(owner)
for suffix, name in [("handwritten", "HandwrittenAnalyticSimulator"),
                     ("gp_landscape", "FrozenGPLandscapeState")]:
    left, right = (importlib.import_module(f"{prefix}.{suffix}") for prefix in (first, second))
    assert left is right and getattr(left, name) is getattr(a, name)
    original_global = f"cmutaco_analytic.{suffix}\\n{name}\\n.".encode()
    assert pickle.loads(original_global) is getattr(right, name)
"""


@pytest.mark.parametrize("first,second", [(PUBLIC, EXPERIMENT), (EXPERIMENT, PUBLIC)])
def test_analytic_forward_models_have_one_public_identity(first, second, tmp_path):
    owner = ROOT / "experiments" / "analytic_target_matching"
    roots = (ROOT, ROOT / "src", owner / "src")
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, roots))}
    subprocess.run(
        [sys.executable, "-c", PROBE, first, second, str(owner / "forward_models")],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
