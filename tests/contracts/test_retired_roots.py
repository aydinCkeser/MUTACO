"""The distribution and its evidence have canonical owners in a clean checkout."""

from pathlib import Path
import subprocess

import pytest

from reproducibility.recorded_paths import RETIRED_ROOTS, resolve_recorded_path


ROOT = Path(__file__).resolve().parents[2]


def test_git_tree_and_checkout_contain_no_retired_roots():
    # A directory symlink, a plain file with the old name or an empty Git
    # subtree would each violate the same top-level contract.
    names = subprocess.check_output(["git", "ls-tree", "-z", "--name-only", "HEAD"], cwd=ROOT)
    assert RETIRED_ROOTS.isdisjoint(p.decode("utf-8") for p in names.split(b"\0") if p)
    assert not any((ROOT / name).exists() or (ROOT / name).is_symlink() for name in RETIRED_ROOTS)


def test_archived_input_addresses_resolve_without_recreating_directories(tmp_path):
    assert resolve_recorded_path(tmp_path, "data/cohorts/paper120_v3/target_016.json") == (
        tmp_path / "experiments/graph_inversion/frozen_targets/paper120_v3/target_016.json"
    )
    assert resolve_recorded_path(tmp_path, tmp_path / "tools/reproducibility/tensor_export.py") == (
        tmp_path / "reproducibility/tensor_export.py"
    )
    assert resolve_recorded_path(tmp_path, "results/paper") == tmp_path / "results/paper"
    external = tmp_path.parent / "private" / "data"
    assert resolve_recorded_path(tmp_path, external) == external
    with pytest.raises(FileNotFoundError, match="no unambiguous current owner"):
        resolve_recorded_path(tmp_path, "benchmarks/paper")
    with pytest.raises(ValueError, match="parent traversal"):
        resolve_recorded_path(tmp_path, "data/../unbound")
    assert list(tmp_path.iterdir()) == []
