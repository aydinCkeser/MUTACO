"""Exercise historical addresses and saved summaries without running a campaign."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAPER = "experiments.paper."
GROUPS = [
    (PAPER + "common", PAPER + "protocol", (
        "campaign cli cohort_v2 target_freeze_v2 supplied_targets method_specs "
        "protocol_v3 population_v3 validation integrity method_append compact_append"
    )),
    (PAPER + "common", PAPER + "historical_workflows", (
        "exact_source_bootstrap exact_terminal_bootstrap exact_terminal_generation "
        "repaired_sbi_assessment_archive_cli repaired_sbi_assessment_execution "
        "repaired_sbi_assessment_preparation repaired_sbi_assessment_preparation_cli "
        "repaired_sbi_recovery repaired_sbi_recovery_cli repaired_sbi_salvage "
        "repaired_sbi_salvage_cli repaired_sbi_terminal_archive repaired_sbi_terminal_archive_cli "
        "repaired_sbi_terminal_preparation repaired_sbi_terminal_preparation_cli "
        "scalar_primary_b100_archive_cli scalar_primary_b100_preparation_audit_cli "
        "terminal_domain_ablation terminal_domain_heldout_assessment "
        "terminal_domain_heldout_assessment_v2 terminal_lineage_forensics terminal_lineage_forensics_cli"
    )),
    (PAPER + "figures.common", PAPER + "figures",
     "figure_artifacts plot_validation style_registry target_selection validated_inputs"),
    (PAPER + "exploration", PAPER + "response_reconstruction", (
        "plot_paper120_target plot_paper120_b40_convergence plot_paper120_target098_mutaco_only_v1 "
        "render_paper120_target098_archive_v1 target098_response_archive"
    )),
    (PAPER + "exploration", PAPER + "historical_workflows",
     "scalar_primary_replay replay_divergence"),
    ("experiments.benchmarking.simulators", "experiments.benchmarking.simulator_adapters",
     "linear handwritten gp_landscape graph_scientific_d4"),
]

PROBE = """
from importlib import import_module
import json
import sys

def forbid_evaluation(frame, event, args):
    if event != "call":
        return
    name = frame.f_code.co_name
    module = frame.f_globals.get("__name__", "")
    if ((module.startswith(("mutaco_graph", "mutaco_analytic")) and name.startswith("evaluate"))
        or (module.startswith("mutaco.surrogate") and name.startswith(("fit_", "_build_model")))):
        raise AssertionError(f"scientific work during import: {module}.{name}")

sys.setprofile(forbid_evaluation)
from experiments.benchmarking import load_results
assert "experiments.benchmarking.api" not in sys.modules
assert not any(s.startswith("experiments.benchmarking.methods") for s in sys.modules)
for address, alternate in json.loads(sys.argv[1]):
    first, second = import_module(address), import_module(alternate)
    assert first is second, (address, alternate)
    sentinel = object()
    first._migration_patch_probe = sentinel
    assert second._migration_patch_probe is sentinel
    del second._migration_patch_probe
assert "sbi" not in sys.modules
"""


@pytest.mark.parametrize("new_first", [False, True])
def test_fresh_process_bindings_and_saved_view_imports(new_first, tmp_path):
    pairs = [(f"{old}.{name}", f"{new}.{name}") for old, new, names in GROUPS
             for name in names.split()]
    pairs += [("experiments.benchmarking.simulators", "experiments.benchmarking.simulator_adapters")]
    for family, module in (("analytic_handwritten", "run"), ("analytic_gp_landscape", "run"),
                           ("oscillator_physics_multi_target_sensitivity", "study"),
                           ("oscillator_physics_post_analysis", "analysis")):
        pairs.append((PAPER + family + "." + module, PAPER + "studies." + family + "." + module))
    if new_first:
        pairs = [(b, a) for a, b in pairs]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *sys.path])
    result = subprocess.run([sys.executable, "-c", PROBE, json.dumps(pairs)],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_saved_benchmark_view_preserves_original_pickle_and_export_bytes(tmp_path):
    from experiments.benchmarking import api, results

    for name in ("BenchmarkResultSet", "ROW_FIELDS", "load_results", "export_tables",
                 "plot_results", "summarize_results"):
        assert getattr(results, name) is getattr(api, name)

    # Captured from the pre-move implementation at 9f3ce189, with the rows below.
    old_pickle = bytes.fromhex(
        "80049537000000000000008c1c6578706572696d656e74732e62656e63686d61726b696e672e"
        "617069948c1242656e63686d61726b526573756c745365749493942e"
    )
    assert pickle.loads(old_pickle) is results.BenchmarkResultSet
    assert pickle.dumps(results.BenchmarkResultSet, protocol=4) == old_pickle
    rows = [
        dict(target_index=i, method=m, status="completed", target_error_l1=x,
             target_error_l2=2*x, design_error_l2=None, latent_error_l1=None,
             latent_error_l2=None, runtime_seconds=1.0, failure_explanation=None)
        for i, m, x in [(2, "mutaco", 0.25), (3, "cp", 0.75)]
    ]
    (tmp_path / "manifest.json").write_text('{"history":"fixed"}')
    (tmp_path / "metrics.json").write_text(json.dumps({"rows": rows}))
    saved = results.load_results(tmp_path)
    assert list(saved.select(methods=["cp"])) == [rows[1]]
    summary = results.summarize_results(saved)
    paths = results.export_tables(saved, summary, output=tmp_path / "tables")
    assert {p.name: sha256(p.read_bytes()).hexdigest() for p in paths} == {
        "records.csv": "04a57ef0a1eeec2fbfe8da9c47933e9fcf1c19eb84cd932d86b99006b4ddb708",
        "summary_groups.json": "aecf0b79ee5960359763c167bcc7225fcda157a75f45375599e0c331fa6edabd",
    }
    (tmp_path / "metrics.json").write_text('{"rows": {}}')
    with pytest.raises(ValueError, match="^metrics rows must be a list$"):
        results.load_results(tmp_path)


@pytest.mark.parametrize("name", [
    "terminal_domain_heldout_assessment",
    "terminal_domain_heldout_assessment_v2",
])
def test_terminal_workflow_uses_original_240_paired_tasks(name, native_terminal_preparation):
    """Compare all paired tasks to the independently archived package in its current home."""
    _, tasks, preflight = native_terminal_preparation(name)
    version = "v2" if name.endswith("_v2") else "v1"
    recorded = (ROOT / f"results/paper/terminal_domain_heldout_assessment_b40_{version}/tasks.jsonl").read_text()
    assert preflight["gate_passed"] and preflight["simulator_calls"] == 0
    assert len(tasks) == 240
    assert tasks == [json.loads(row) for row in recorded.splitlines()]


@pytest.mark.parametrize("owner", ["common", "historical_workflows"])
def test_exact_source_bootstrap_standalone_help_uses_only_stdlib(owner, tmp_path):
    script = ROOT / "experiments" / "paper" / owner / "exact_source_bootstrap.py"
    result = subprocess.run([sys.executable, "-I", "-S", "-B", str(script), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--source-commit" in result.stdout and "--source-root" in result.stdout



def test_retired_common_cli_files_dispatch_to_their_owners(tmp_path):
    # One fresh process exercises argparse's exit at --help for every owner.
    program = """
import contextlib, io, runpy, sys
from experiments.paper.common import _IMPORT_OWNERS
for name, owner in _IMPORT_OWNERS.items():
    if not name.endswith("_cli"):
        continue
    for address in (owner, "experiments.paper.common." + name):
        sys.argv = [address, "--help"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            try:
                runpy.run_module(address, run_name="__main__")
            except SystemExit as error:
                assert error.code == 0
            else:
                raise AssertionError(address + " did not stop at --help")
        assert "usage:" in output.getvalue()
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), *sys.path]))
    result = subprocess.run([sys.executable, "-B", "-c", program], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
