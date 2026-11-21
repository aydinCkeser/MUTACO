"""Selected exploration entry points, isolated from all real graph evaluations."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from declared_path_edits import original_ast_after_checking_path_migration


ROOT = Path(__file__).resolve().parents[2]
EXPLORATION = ROOT / "experiments/paper/exploration"
NOTEBOOK = "plot_paper120_target_v2.py"
ORIGINAL_NOTEBOOK_AST = "5c9a0bd3901455b3ee6c9173c99fb79577bb7dc129362a49aa3685da911b18af"

# Redirect only the notebook's legacy root for characterization on another
# machine. No numerical expression, input file, or simulator result is replaced.
PROBE = r'''
import ast
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from mutaco_graph import OverdampedScientificSimulator

source, mode, root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
attempts = []
class PhysicalTripwire(BaseException):
    pass
def forbidden(self, *args, **kwargs):
    item = {"entrypoint": "graph_physical_backend"}
    if len(args) == 3:
        theta, context, grid = args
        item.update(theta=theta.tolist(), dtype=str(theta.dtype),
                    context=asdict(context), conditions=len(grid.forces),
                    chunk=kwargs.get("max_conditions_per_chunk"))
    attempts.append(item)
    raise PhysicalTripwire()

with ExitStack() as stack:
    for name in ("evaluate", "evaluate_batch", "evaluate_condition_grid"):
        stack.enter_context(patch.object(OverdampedScientificSimulator, name, forbidden))
    if source.name == "plot_paper120_target_v2.py":
        tree = ast.parse(source.read_text())
        legacy = next(Path(node.value.args[0].value) for node in ast.walk(tree)
                      if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                      and any(isinstance(t, ast.Name) and t.id == "repo" for t in node.targets))
        original_open, original_is_dir = Path.open, Path.is_dir
        def mapped_open(path, *args, **kwargs):
            try:
                path = root / path.relative_to(legacy)
            except ValueError:
                pass
            return original_open(path, *args, **kwargs)
        stack.enter_context(patch.object(Path, "open", mapped_open))
        stack.enter_context(patch.object(Path, "is_dir",
                                        lambda path: True if path == legacy else original_is_dir(path)))
    sys.argv = [str(source)] + (["--help"] if mode == "help" else [])
    outcome = "returned"
    try:
        runpy.run_path(str(source), run_name="__main__" if mode != "import" else "probe_import")
    except PhysicalTripwire:
        outcome = "blocked_physical_attempt"
    except SystemExit as exc:
        outcome = "exit_" + str(exc.code)
print("MUTACO_ENTRYPOINT_PROBE=" + json.dumps({"outcome": outcome, "attempts": attempts}, sort_keys=True))
'''


def probe(name: str, mode: str, temporary: Path) -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(EXPLORATION / name), mode, str(ROOT)],
        cwd=temporary, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                           if line.startswith("MUTACO_ENTRYPOINT_PROBE=")))


@pytest.mark.parametrize("name", [
    "plot_paper120_target.py", "plot_paper120_target098_mutaco_only_v1.py",
    "plot_paper120_b40_convergence.py", "render_paper120_target098_archive_v1.py",
])
def test_existing_guarded_imports_do_not_evaluate(name: str, tmp_path: Path) -> None:
    assert probe(name, "import", tmp_path) == {"outcome": "returned", "attempts": []}


def test_notebook_import_boundary(tmp_path: Path) -> None:
    assert probe(NOTEBOOK, "import", tmp_path) == {"outcome": "returned", "attempts": []}


def test_explicit_notebook_execution_retains_first_physical_request(tmp_path: Path) -> None:
    result = probe(NOTEBOOK, "execute", tmp_path)
    assert result["outcome"] == "blocked_physical_attempt"
    assert len(result["attempts"]) == 1
    request = result["attempts"][0]
    target = json.loads((ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3/target_098.json").read_text())
    assert request["theta"] == target["theta_star"]
    assert request["dtype"] == "torch.float32"
    assert request["conditions"] == 200
    assert request["chunk"] == 10
    assert request["context"]["evaluation_seed"] == target["target_generation_seed"]


def test_notebook_scientific_statements_are_preserved() -> None:
    tree = original_ast_after_checking_path_migration(ROOT, (EXPLORATION / NOTEBOOK).resolve())
    # Allow only a module docstring and a main guard around the original body.
    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body.pop(0)
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.If):
        guard = tree.body[0]
        assert ast.unparse(guard.test) == "__name__ == '__main__'"
        assert not guard.orelse
        tree.body = guard.body
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == ORIGINAL_NOTEBOOK_AST


@pytest.mark.parametrize("name", ["plot_paper120_target.py", "render_paper120_target098_archive_v1.py"])
def test_documented_cli_help_has_no_physical_calls(name: str, tmp_path: Path) -> None:
    assert probe(name, "help", tmp_path) == {"outcome": "exit_0", "attempts": []}
