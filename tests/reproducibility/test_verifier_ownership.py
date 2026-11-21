"""Saved-file and standalone CLI checks. No scientific package is required."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("legacy_first", [True, False])
def test_historical_imports_and_pickles_in_original_checkout(tmp_path, legacy_first, legacy_checkout):
    code = """
import importlib, pickle, sys
sys.path.insert(0, sys.argv[1])
pairs = [
    ("tools.reproducibility.verify", "reproducibility.verify", "main"),
    ("tools.reproducibility.verification", "reproducibility.verification", "compare_files"),
    ("experiments.paper.common.hash_pair_inventory_cli", "reproducibility.hash_pair_inventory", "build_inventory"),
]
for old, new, symbol in pairs:
    first, second = (old, new) if sys.argv[2] == "True" else (new, old)
    a, b = importlib.import_module(first), importlib.import_module(second)
    assert a is b
    value = getattr(a, symbol)
    assert value is getattr(b, symbol)
    assert pickle.loads(pickle.dumps(value)) is value
    def replacement(): pass
    setattr(a, symbol, replacement)
    assert getattr(b, symbol) is replacement
assert not {"torch", "numpy", "sbi", "mutaco", "mutaco_graph"} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-B", "-I", "-S", "-c", code, str(legacy_checkout), str(legacy_first)],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("legacy_first", [False, True])
def test_modern_and_legacy_imports_share_verifier_globals_from_configured_launchers(tmp_path, legacy_first):
    code = """
import importlib, pickle, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "src")]
for name, symbol in (("verify", "main"), ("verification", "compare_files")):
    old, new = "tools.reproducibility." + name, "reproducibility." + name
    first, second = (old, new) if sys.argv[2] == "True" else (new, old)
    a, b = importlib.import_module(first), importlib.import_module(second)
    assert a is b
    value = getattr(a, symbol)
    assert pickle.loads(pickle.dumps(value)) is value
    historical = ("c" + old + chr(10) + symbol + chr(10) + ".").encode()
    assert pickle.loads(historical) is value
    setattr(a, symbol, object())
    assert getattr(a, symbol) is getattr(b, symbol)
from reproducibility import hash_pair_inventory
assert pickle.loads(pickle.dumps(hash_pair_inventory.main)) is hash_pair_inventory.main
assert not {"torch", "numpy", "sbi", "mutaco", "mutaco_graph"} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", code, str(ROOT), str(legacy_first)],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_checkout_tensor_export_path_keeps_original_exception_and_function_identity():
    import pickle
    from tools.reproducibility import tensor_export as old
    from reproducibility import tensor_export as new

    assert old is new
    for symbol in ("TensorEvidenceExportError", "save_tensor_evidence"):
        historical = f"ctools.reproducibility.tensor_export\n{symbol}\n.".encode()
        assert pickle.loads(historical) is getattr(new, symbol)


@pytest.mark.parametrize("script", [
    "reproducibility/hash_pair_inventory.py",
    "experiments/paper/common/hash_pair_inventory_cli.py",
])
def test_copied_standalone_inventory_preserves_success_and_diagnostics(tmp_path, script, legacy_checkout):
    # The original single file can be handed to an authorized remote operator.
    # Test that mode with an external temporary directory and no other source.
    standalone = tmp_path / "inventory.py"
    source = ROOT if script.startswith("reproducibility/") else legacy_checkout
    shutil.copyfile(source / script, standalone)
    data = tmp_path / "data"
    data.mkdir()
    for name in ("worker_start", "mutaco", "sobol"):
        value = data / (name + ".json")
        value.write_bytes(b'{"n":-0.0}\n')
        if name != "sobol":
            digest = hashlib.sha256(value.read_bytes()).hexdigest()
            value.with_suffix(".sha256").write_text(digest if name == "mutaco" else "0"*64)
    (data / "orphan.sha256").write_text("1"*64)

    def invoke(directory):
        return subprocess.run([sys.executable, "-B", "-I", "-S", str(standalone), str(directory)],
                              cwd=tmp_path, text=True, capture_output=True, timeout=10)

    result = invoke(data)
    assert (result.returncode, result.stderr) == (0, "")
    inventory = json.loads(result.stdout)
    # Captured via the former standalone script at 9f3ce189.
    assert inventory["inventory_sha256"] == "cc4202b133cc2ba9c30b995895cd43e4d7142f5d8dc98ec06d42fe4843e89808"
    assert inventory["hash_mismatches"] == ["worker_start.json"]
    assert inventory["missing_sidecars"] == ["sobol.json"]
    assert inventory["orphan_sidecars"] == ["orphan.sha256"]
    assert inventory["json_bytes"] == 33 and inventory["scientific_calls"] == 0
    failed = invoke(data / "missing")
    assert failed.returncode == 1 and not failed.stdout
    assert failed.stderr.splitlines()[-1] == "ValueError: inventory root must be a directory"


def test_sealed_compact_sbi_inputs_agree_with_native_historical_verifier(tmp_path, legacy_checkout):
    # The original inventory includes each row file's resolved relative name.
    # Evaluate that name in the historical tree and compare actual canonical
    # bytes independently; no private worker trajectories are required here.
    code = """
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / path) for path in (".", "src", "packages/graph-simulator/src",
                                              "packages/analytic-simulator/src")]
from experiments.paper.common import repaired_sbi_salvage as original
from mutaco import __file__ as package_file
assert Path(package_file).resolve().is_relative_to(root)
cohort = original._cohort_ids(root)
out = {}
for budget in (40, 100):
    path = root / getattr(original, f"B{budget}_COMPACT")
    manifest = json.loads((root / getattr(original, f"B{budget}_COMPACT_MANIFEST")).read_text())
    out[str(budget)] = original._check_compact(
        path, cohort, root, budget=budget, expected_sha256=manifest["paired_records_sha256"])
print(json.dumps(out))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-c", code, str(legacy_checkout)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    checked = json.loads(completed.stdout)
    sealed = json.loads((ROOT / "results/paper/repaired_mutaco_sbi_salvage_design/v1/inventory.json").read_text())
    destinations = {"40": "paper120_mutaco_npe_snpe_v1", "100": "b100_v1"}
    for budget, directory in destinations.items():
        evidence = sealed["b" + budget]["compact_source"]
        assert checked[budget] == evidence
        owner = ROOT / "results/paper/graph_paper120" / directory
        assert owner.is_dir() and not owner.is_symlink()
        assert (owner / "paired_records.jsonl").read_bytes() == (legacy_checkout / evidence["path"]).read_bytes()
