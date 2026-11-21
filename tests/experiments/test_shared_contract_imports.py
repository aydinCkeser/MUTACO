"""Fresh-process ownership, optional-dependency and legacy identity checks."""
from __future__ import annotations

import ast
import importlib
from itertools import permutations
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from declared_path_edits import original_ast_after_checking_path_migration

from experiments.contracts import PaperContractError, PhysicalCallCounts
from experiments.paper.common import protocol_v3, validation


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/shared_contracts_before_2d.json"
TRIO = (
    "experiments.benchmarking.methods.sbi_zoo_failures",
    "experiments.paper.common.protocol_v3",
    "experiments.benchmarking.methods.direct_posterior_decision",
)
OTHER = (
    "experiments.benchmarking.methods.sbi_zoo_dataset",
    "experiments.benchmarking.methods.sbi_zoo_accounting",
    "experiments.benchmarking.methods.sbi_zoo_inference",
    "experiments.benchmarking.methods.direct_posterior_inference",
    "experiments.graph_inversion.sbi_zoo_qualification",
    "experiments.graph_inversion.sbi_zoo_p2",
    "experiments.graph_inversion.sbi_zoo_p3",
)

PROBE = r'''
import importlib, importlib.abc, json, pickle, random, subprocess, sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
mode, order, fixture = json.loads(sys.argv[1])
blocked = {"sbi"}
if mode == "lean":
    blocked |= {"torch", "numpy", "scipy", "botorch", "gpytorch", "matplotlib", "mutaco", "mutaco_graph"}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError("optional dependency forbidden: " + fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
def forbidden(*args, **kwargs):
    raise AssertionError("scientific or external execution forbidden during imports")
with ExitStack() as stack:
    stack.enter_context(patch.object(subprocess, "Popen", forbidden))
    if mode != "lean":
        import numpy as np
        import torch
        from mutaco.surrogate import exact_multitask as gp
        from mutaco_graph import OverdampedScientificSimulator
        for name in ("_build_model", "fit_exact_multitask_gp", "fit_exact_multitask_gp_with_recovery"):
            stack.enter_context(patch.object(gp, name, forbidden))
        for name in ("__post_init__", "evaluate", "evaluate_batch", "evaluate_condition_grid"):
            stack.enter_context(patch.object(OverdampedScientificSimulator, name, forbidden))
        assert not any(name.startswith("experiments.paper") for name in sys.modules)
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
    python_state = random.getstate()
    for module in order:
        importlib.import_module(module)
    if mode == "pickle":
        saved = json.loads(Path(fixture).read_text())["snapshot"]["pickles"]
        from experiments.contracts import PhysicalCallCounts, PaperContractError
        for versions in saved.values():
            for number, body in versions.items():
                value = pickle.loads(bytes.fromhex(body))
                assert pickle.dumps(value, protocol=int(number)).hex() == body
    assert random.getstate() == python_state
    if mode != "lean":
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        now = np.random.get_state()
        assert now[0] == numpy_state[0] and np.array_equal(now[1], numpy_state[1]) and now[2:] == numpy_state[2:]
    assert not any(name.split('.')[0] in blocked for name in sys.modules)
print("imports and identities verified; scientific execution forbidden")
'''


def probe(mode: str, order: tuple[str, ...], temporary: Path) -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", PROBE, json.dumps([mode, order, str(FIXTURE)])],
                            cwd=temporary, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imports and identities verified; scientific execution forbidden"


@pytest.mark.parametrize("order", list(permutations(TRIO)))
def test_original_cycle_is_gone_for_all_three_module_orders(order, tmp_path) -> None:
    probe("guarded", order, tmp_path)


@pytest.mark.parametrize("reverse", (False, True))
def test_direct_SBI_consumers_do_not_need_an_initialization_workaround(reverse, tmp_path) -> None:
    probe("guarded", OTHER[::-1] if reverse else OTHER, tmp_path)


def test_contracts_and_validation_import_without_any_scientific_packages(tmp_path) -> None:
    probe("lean", ("experiments.contracts", "experiments.paper", "experiments.paper.common",
                   "experiments.paper.common.validation", "experiments.benchmarking.methods.sbi_zoo_failures"), tmp_path)


def test_historical_pickle_loading_and_rewriting_in_a_fresh_process(tmp_path) -> None:
    probe("pickle", ("experiments.benchmarking.methods.sbi_zoo_failures",), tmp_path)


def test_shared_aliases_exception_inheritance_and_canonical_names() -> None:
    assert PhysicalCallCounts is protocol_v3.PhysicalCallCounts
    assert PaperContractError is validation.PaperContractError
    assert PhysicalCallCounts.__module__ is protocol_v3.__name__
    assert PaperContractError.__module__ is validation.__name__
    for name in ("direct_posterior_decision", "direct_posterior_inference", "sbi_zoo_inference"):
        module = importlib.import_module("experiments.benchmarking.methods." + name)
        assert module.PhysicalCallCounts is PhysicalCallCounts
    failures = importlib.import_module("experiments.benchmarking.methods.sbi_zoo_failures")
    assert failures.PaperContractError is PaperContractError
    error = failures.SbiZooScoringContractError("fixed")
    assert isinstance(error, (failures.SbiZooError, PaperContractError))
    assert isinstance(error, RuntimeError) and isinstance(error, ValueError)


@pytest.mark.parametrize("name", ("experiments.paper", "experiments.paper.common"))
def test_every_existing_public_export_resolves_to_the_original_object(name, monkeypatch) -> None:
    package = importlib.import_module(name)
    expected = json.loads(FIXTURE.read_text())["snapshot"]["exports"][name]
    assert list(package.__all__) == expected
    assert set(expected) <= set(dir(package))
    for public in expected:
        module, symbol = package._EXPORTS[public]
        original = getattr(importlib.import_module("." + module, name), symbol)
        assert getattr(package, public) is original
        assert vars(package)[public] is original
    public = expected[0]
    replacement = object()
    monkeypatch.setattr(package, public, replacement)
    assert getattr(package, public) is replacement
    with pytest.raises(AttributeError, match="has no attribute 'absent_export'"):
        getattr(package, "absent_export")


def test_only_imports_metadata_and_definition_locations_changed() -> None:
    # Hash the historical definitions and verify the complete current validator, including its path binding.
    original = json.loads(FIXTURE.read_text())["unchanged_runtime_AST_sha256"]
    import hashlib
    for name, expected in original.items():
        source = (ROOT / name).resolve()
        if name == "experiments/paper/common/validation.py":
            tree = original_ast_after_checking_path_migration(ROOT, source, substitutions=(
                ("from experiments.contracts import PaperContractError",
                 "from experiments.contracts import PaperContractError\nfrom reproducibility.recorded_paths import resolve_recorded_path"),
                ("(root / value).resolve()", "resolve_recorded_path(root, value).resolve()"),
            ))
        else:
            tree = ast.parse(source.read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                       and node.name not in {"PaperContractError", "PhysicalCallCounts"}]
        assert hashlib.sha256(ast.dump(ast.Module(body=definitions, type_ignores=[]), include_attributes=False).encode()).hexdigest() == expected
