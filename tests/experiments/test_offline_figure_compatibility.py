"""Supported import/module/file entry points, without scientific execution."""
from __future__ import annotations

import ast
import hashlib
import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from declared_path_edits import original_ast_after_checking_path_migration


ROOT = Path(__file__).resolve().parents[2]
BODY_HASHES = {
    "plot_paper120_primary_boxplot_v1": "ecb6668c8405bbb3d09b97a1d0ade9150b23cb7d886db9366131ac42b471c24f",
    "plot_paper120_primary_parameter_boxplot_v1": "f078dd6aa1edbc0546e341eeac62a7dab730bab5ae0b0f5a0ce3e02650c614aa",
    "plot_paper120_sbi_response_boxplot_v1": "b4d990fa2555800b7af6e156daac7bc780e68af073fe96218edcfcfa670eeae9",
    "plot_paper120_terminal_domain_ablation_v1": "3476269c12d6b42b48a526aed2cb97094cfc46804558116a2f3fcbda1bdfc02f",
    "plot_paper120_local_sensitivity_v6": "a11531e26a83cc8ff67b79783268062f663365cc3d922ad69a82a0844d4124f0",
}


@pytest.mark.parametrize("name", BODY_HASHES)
def test_shared_legacy_module_globals_pickle_and_default_dispatch(name, monkeypatch) -> None:
    old = importlib.import_module("experiments.paper.exploration." + name)
    new = importlib.import_module("experiments.paper.figures." + name)
    assert old is new
    assert Path(new.__file__).parent == ROOT / "experiments/paper/figures"
    assert new.main.__globals__ is vars(old)
    assert new.main.__module__ == "experiments.paper.exploration." + name
    assert pickle.loads(pickle.dumps(new.main)) is new.main
    calls = []
    monkeypatch.setattr(old, "main", lambda: calls.append("existing main"))
    monkeypatch.setattr(sys, "argv", [name])
    new._cli()
    assert calls == ["existing main"]
    monkeypatch.setattr(sys, "argv", [name, "--unknown-option"])
    with pytest.raises(SystemExit) as error:
        new._cli()
    assert error.value.code == 2 and calls == ["existing main"]


@pytest.mark.parametrize("name", BODY_HASHES)
def test_all_original_computational_and_rendering_statements_are_unchanged(name) -> None:
    tree = original_ast_after_checking_path_migration(
        ROOT, ROOT / "experiments/paper/figures" / (name + ".py")
    )
    original = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "_cli":
            assert ast.unparse(node.body[-1]) == "main()"
            assert ast.unparse(node.body[-2]) == "parser.parse_args()"
            continue
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "_legacy_export":
            assert ast.unparse(node.body[0]) == (
                "_legacy_export.__module__ = 'experiments.paper.exploration." + name + "'"
            )
            continue
        if isinstance(node, ast.Delete) and ast.unparse(node) == "del _legacy_export":
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            assert len(node.body) == 1 and ast.unparse(node.body[0]) == "_cli()"
            node.body[0].value.func.id = "main"
        original.append(node)
    tree.body = original
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == BODY_HASHES[name]


PROBE = r'''
from contextlib import ExitStack, redirect_stdout
import importlib
import importlib.abc
import io
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

mode, target = sys.argv[1:3]
blocked = ("torch", "botorch", "gpytorch", "sbi", "mutaco_graph") if mode == "file" else ("sbi",)
class NoOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError("dependency absent: " + fullname, name=fullname)
sys.meta_path.insert(0, NoOptional())
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
attempts = []
def forbidden(*a, **k):
    attempts.append("blocked execution or figure creation")
    raise AssertionError("help/import must not start scientific work or rendering")
with ExitStack() as stack:
    stack.enter_context(patch.object(plt, "subplots", forbidden))
    stack.enter_context(patch.object(Figure, "savefig", forbidden))
    if mode != "file":
        from mutaco_graph import OverdampedScientificSimulator
        from mutaco.surrogate import exact_multitask as gp
        for name in ("evaluate", "evaluate_batch", "evaluate_condition_grid"):
            stack.enter_context(patch.object(OverdampedScientificSimulator, name, forbidden))
        for name in ("_build_model", "fit_exact_multitask_gp", "fit_exact_multitask_gp_with_recovery"):
            stack.enter_context(patch.object(gp, name, forbidden))
    capture = io.StringIO()
    if mode == "import":
        names = target.split(',')
        for index, name in enumerate(names):
            namespaces = ("figures", "exploration") if index % 2 else ("exploration", "figures")
            a = importlib.import_module("experiments.paper." + namespaces[0] + "." + name)
            b = importlib.import_module("experiments.paper." + namespaces[1] + "." + name)
            assert a is b
        code = 0
    else:
        sys.argv = [target, "--help"]
        with redirect_stdout(capture):
            try:
                if mode == "file":
                    runpy.run_path(target, run_name="__main__")
                else:
                    runpy.run_module(target, run_name="__main__")
            except SystemExit as error:
                code = error.code
            else:
                raise AssertionError("help did not terminate before main")
        assert "usage:" in capture.getvalue() and "--help" in capture.getvalue()
    assert code == 0 and not attempts
    assert not any(root in sys.modules for root in blocked)
print("SAFE_RENDERER_ENTRYPOINT")
'''


def probe(mode: str, target: str, temporary: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["MPLBACKEND"] = "Agg"
    environment.setdefault("MPLCONFIGDIR", str(temporary / "mpl"))
    result = subprocess.run([sys.executable, "-c", PROBE, mode, target], cwd=temporary,
                            env=environment, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SAFE_RENDERER_ENTRYPOINT"


def test_mixed_fresh_import_order_without_sbi_or_execution(tmp_path) -> None:
    probe("import", ",".join(BODY_HASHES), tmp_path)


@pytest.mark.parametrize("name", BODY_HASHES)
@pytest.mark.parametrize("namespace", ["exploration", "figures"])
def test_old_new_module_help_has_no_rendering_or_scientific_calls(name, namespace, tmp_path) -> None:
    probe("module", "experiments.paper." + namespace + "." + name, tmp_path)


@pytest.mark.parametrize("name", BODY_HASHES)
@pytest.mark.parametrize("directory", ["exploration", "figures"])
def test_old_new_standalone_help_needs_no_scientific_packages(name, directory, tmp_path) -> None:
    probe("file", str(ROOT / "experiments/paper" / directory / (name + ".py")), tmp_path)
