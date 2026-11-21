from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import mutaco
import mutaco_analytic


ROOT = Path(__file__).resolve().parents[2]


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for source in path.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_packages_import_and_versions_are_declared() -> None:
    assert mutaco.__version__ == "0.1.0.dev0"
    assert mutaco_analytic.HandwrittenAnalyticSimulator().capabilities.response_spec.dimension == 3


def test_core_and_analytic_packages_preserve_dependency_boundaries() -> None:
    core_imports = _import_roots(ROOT / "src" / "mutaco")
    analytic_imports = _import_roots(
        ROOT / "experiments" / "analytic_target_matching" / "forward_models"
    )
    assert {"gpytorch", "botorch"}.issubset(core_imports)
    assert not core_imports.intersection({"numpy", "cupy", "sbi", "matplotlib"})
    assert "torch" in analytic_imports
    assert not analytic_imports.intersection({"numpy", "cupy", "gpytorch", "botorch", "sbi"})


def test_original_software_distributions_declare_the_release_license() -> None:
    for directory in (ROOT, ROOT / "experiments/analytic_target_matching", ROOT / "simulators/graph-simulator"):
        project = tomllib.loads((directory / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        assert project["license"] == "BSD-3-Clause"
        assert project["license-files"] == ["LICENSE"]
        assert (directory / "LICENSE").is_file()
