"""Legacy/new module identity, fixed source bodies, and guarded fresh CLIs."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from declared_path_edits import original_ast_after_checking_path_migration


ROOT = Path(__file__).resolve().parents[2]
NAMES = ("aggregate", "cross_budget_synthesis", "cross_budget_synthesis_cli",
         "evidence_analysis_cli", "timing_report")
# Captured from 9e5732f before relocation; only import paths and the explicit
# metadata compatibility block may differ. All computational AST remains exact.
BODY_HASHES = {
    "aggregate": "1424ac01cad068e9a589b1d42ba07a72e8ac897d489335ddec9e16b08e95afc9",
    "timing_report": "f7513227c3b9df7da9f1eb6c18ae77dc658611aa3057194d1e1309fef3fcc6d7",
    "cross_budget_synthesis": "89cb4f0f8fb4b2cdc6e606f31c8ef7790f42e57bed93a3ec7594b83cd3b02cff",
    "cross_budget_synthesis_cli": "4ac386e09fd83472c24a2be198238ef73e347171e02f0667f91036843db67745",
    "evidence_analysis_cli": "6594b052a3321c4cb9eaecb39bd66872b2bde3266d48ee042dd932c9dbc3b5d3",
}


@pytest.mark.parametrize("name", NAMES)
def test_old_and_new_paths_share_one_module_and_function_globals(name: str) -> None:
    old = importlib.import_module("experiments.paper.common." + name)
    new = importlib.import_module("experiments.paper.analysis." + name)
    assert old is new
    assert Path(new.__file__).parent == ROOT / "experiments/paper/analysis"
    for value in vars(new).values():
        if getattr(value, "__module__", None) == "experiments.paper.common." + name:
            assert getattr(old, value.__name__) is value
            if hasattr(value, "__globals__"):
                assert value.__globals__ is vars(new)


@pytest.mark.parametrize("name", NAMES)
def test_relocated_computational_ast_matches_original(name: str) -> None:
    tree = original_ast_after_checking_path_migration(
        ROOT, ROOT / "experiments/paper/analysis" / (name + ".py")
    )
    metadata = [node for node in tree.body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "_legacy_export"]
    assert len(metadata) == 1
    assert len(metadata[0].body) == 1 and not metadata[0].orelse
    assert ast.unparse(metadata[0].body[0]) == (
        "_legacy_export.__module__ = 'experiments.paper.common." + name + "'"
    )
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))
                 and node not in metadata
                 and not (isinstance(node, ast.Delete)
                          and ast.unparse(node) == "del _legacy_export")]
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == BODY_HASHES[name]


def test_legacy_class_and_bootstrap_pickle_bytes_are_unchanged() -> None:
    from experiments.paper.analysis.aggregate import PaperRecordSet, _bootstrap_mean_interval

    # Exact original global references, captured before moving the source.
    expected = [
        "80049539000000000000008c226578706572696d656e74732e70617065722e636f6d6d6f6e2e616767726567617465948c0e50617065725265636f72645365749493942e",
        "80049543000000000000008c226578706572696d656e74732e70617065722e636f6d6d6f6e2e616767726567617465948c185f626f6f7473747261705f6d65616e5f696e74657276616c9493942e",
    ]
    for value, original in zip((PaperRecordSet, _bootstrap_mean_interval), expected, strict=True):
        assert pickle.dumps(value, protocol=4).hex() == original
        assert pickle.loads(bytes.fromhex(original)) is value
    record = PaperRecordSet(Path("fixed-input"), ())
    assert pickle.loads(pickle.dumps(record)) == record


def test_monkeypatching_legacy_dependency_changes_new_callable(monkeypatch) -> None:
    old = importlib.import_module("experiments.paper.common.aggregate")
    new = importlib.import_module("experiments.paper.analysis.aggregate")

    def rejected(rows):
        raise ValueError("characterized dependency seam")

    monkeypatch.setattr(old, "_validate_cohort", rejected)
    with pytest.raises(ValueError, match="characterized dependency seam"):
        new.summarize_paired(())


def test_new_cli_malformed_rows_keep_original_exception(tmp_path: Path) -> None:
    module = importlib.import_module("experiments.paper.analysis.cross_budget_synthesis_cli")
    from experiments.paper.common.validation import PaperContractError

    assert module.PaperContractError is PaperContractError
    source = tmp_path / "rows.jsonl"
    source.write_text('{"errors":{"invented":{}}}\n')
    with pytest.raises(PaperContractError, match="persisted analysis row error vocabulary mismatch"):
        module._read_jsonl(source)


PROBE = r'''
from contextlib import ExitStack, redirect_stdout
import importlib
import importlib.abc
import io
import json
import runpy
import sys
from unittest.mock import patch
from mutaco_graph import OverdampedScientificSimulator
from mutaco.surrogate import exact_multitask as gp

names = ("aggregate", "cross_budget_synthesis", "cross_budget_synthesis_cli",
         "evidence_analysis_cli", "timing_report")
blocked_optional = ("sbi", "matplotlib", "seaborn", "tensorboard")
assert not any(root in sys.modules for root in blocked_optional)
class NoOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked_optional:
            raise ModuleNotFoundError("optional dependency absent: " + fullname, name=fullname)
sys.meta_path.insert(0, NoOptional())
attempts = []
def forbidden(*args, **kwargs):
    attempts.append("blocked scientific execution")
    raise AssertionError("physical simulation and GP fits forbidden")
with ExitStack() as stack:
    for name in ("evaluate", "evaluate_batch", "evaluate_condition_grid"):
        stack.enter_context(patch.object(OverdampedScientificSimulator, name, forbidden))
    for name in ("_build_model", "fit_exact_multitask_gp", "fit_exact_multitask_gp_with_recovery"):
        stack.enter_context(patch.object(gp, name, forbidden))
    mode = sys.argv[1]
    if mode == "order":
        order = sys.argv[2]
        for index, name in enumerate(names):
            first = "common" if order == "old" or (order == "mixed" and index % 2 == 0) else "analysis"
            second = "analysis" if first == "common" else "common"
            a = importlib.import_module("experiments.paper." + first + "." + name)
            b = importlib.import_module("experiments.paper." + second + "." + name)
            assert a is b
        output = "imports share one implementation"
    else:
        module, root = sys.argv[2:4]
        sys.argv = [module, "--repository-root", root, "--check"]
        captured = io.StringIO()
        with redirect_stdout(captured):
            runpy.run_module(module, run_name="__main__")
        output = captured.getvalue()
    assert not attempts
print(json.dumps({"output": output, "attempts": attempts, "optional_loaded": [n for n in blocked_optional if n in sys.modules]}))
'''


def probe(*arguments: str, temporary: Path) -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", PROBE, *arguments], cwd=temporary,
                            env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["attempts"] == [] and document["optional_loaded"] == []
    return document


@pytest.mark.parametrize("order", ["old", "new", "mixed"])
def test_fresh_import_orders_without_optional_dependencies(order: str, tmp_path: Path) -> None:
    assert probe("order", order, temporary=tmp_path)["output"] == "imports share one implementation"


@pytest.mark.parametrize("namespace", ["common", "analysis"])
@pytest.mark.parametrize("cli", ["evidence_analysis_cli", "cross_budget_synthesis_cli"])
def test_old_and_new_module_commands_reproduce_fixed_archived_packages(
    namespace: str, cli: str, offline_analysis_root: Path, tmp_path: Path,
) -> None:
    # Frozen package identities are independent of the relocated implementation.
    if cli == "evidence_analysis_cli":
        expected = {
            "status": "reproduced", "output": "results/paper/cross_experiment_analysis/b40_v1",
            "package_identity": "cross-experiment-analysis-package-v1-826e4063cfa73b465a112fe64dbaaa7ceac8cbe7ac6d96e8b2892c04605c3c94",
            "discrepancy_count": 0, "scientific_calls": 0, "truba_jobs_submitted": 0,
        }
    else:
        expected = {
            "status": "reproduced", "output": "results/paper/scalar_primary_b40_b100_synthesis_v1",
            "package_identity": "scalar-primary-cross-budget-synthesis-package-v1-b3e3e28b695534b8ec550ed4da7d75a69c3ad028794da050f864535bd8ddab3f",
            "scientific_calls": 0, "truba_jobs_submitted": 0,
        }
    actual = probe("cli", "experiments.paper." + namespace + "." + cli,
                   str(offline_analysis_root), temporary=tmp_path)
    assert actual["output"] == json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
