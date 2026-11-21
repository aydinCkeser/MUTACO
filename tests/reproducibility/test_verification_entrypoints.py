"""Read-only CLI compatibility with pinned implementations and explicit inputs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reproducibility import verification as api
from verification_fixtures import checkpoint_fixture, fit_inputs, test_gate_inputs, write_json


ROOT = Path(__file__).resolve().parents[2]
BEFORE = json.loads((ROOT / "tests/fixtures/offline_verification_before_2c.json").read_text())


def test_original_implementation_bytes_and_function_identity_are_preserved(legacy_checkout: Path) -> None:
    # The independent fixture pins the legacy bytes. Three current bootstraps
    # have new path literals; every other byte, including numerical comparisons,
    # must still match the pinned implementation.
    package_paths = {
        b"configs/environments/mutaco-sbi-lock.json": b"infrastructure/environments/mutaco-sbi-lock.json",
        b"packages/analytic-simulator/src": b"experiments/analytic_target_matching/src",
        b"packages/graph-simulator/src": b"simulators/graph-simulator/src",
    }
    result_path = {b"benchmarks/paper/": b"results/paper/"}
    changed_paths = {
        "refactor_reference": {**package_paths, **result_path},
        "b100_regression": result_path,
        "readability_regression": result_path,
    }
    for name, expected in BEFORE["pinned_implementation_sha256"].items():
        path = Path("infrastructure/truba") / (name + ".py")
        old = (legacy_checkout / path).read_bytes()
        assert hashlib.sha256(old).hexdigest() == expected
        allowed = old
        for previous, current in changed_paths.get(name, {}).items():
            assert previous in old
            allowed = allowed.replace(previous, current)
        assert (ROOT / path).read_bytes() == allowed
    bindings = {
        "canonical": ("refactor_reference", "canonical"),
        "digest": ("refactor_reference", "digest"),
        "numerical_payload": ("refactor_reference", "numerical_payload"),
        "environment_signature": ("refactor_reference", "environment_signature"),
        "compare_files": ("refactor_reference", "compare_files"),
        "first_difference": ("b100_regression", "first_difference"),
        "test_outcomes": ("b100_regression", "test_outcomes"),
        "compare_b100_histories": ("b100_regression", "compare_histories"),
        "verify_inputs": ("readability_regression", "verify_inputs"),
        "history_gate": ("readability_regression", "history_gate"),
        "compare_history": ("readability_regression", "compare_history"),
        "tests_gate": ("readability_regression", "tests_gate"),
        "expected_fit_events": ("fit_event_evidence", "expected_fit_events"),
        "compare_fit_events": ("fit_event_evidence", "compare_fit_events"),
        "load": ("shareable_evidence", "load"), "sha": ("shareable_evidence", "sha"),
        "verify_b40": ("shareable_evidence", "verify"), "verify_b100": ("b100_evidence", "verify"),
    }
    for exported, (module, original) in bindings.items():
        legacy = importlib.import_module("infrastructure.truba." + module)
        assert getattr(api, exported) is getattr(legacy, original)


PROBE = r'''
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import builtins
import importlib.abc
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

module, arguments, root, inputs, readonly = json.loads(sys.argv[1])
root = Path(root)
sys.path[:0] = [str(root), str(root / "src")]
inputs = [Path(p).resolve() for p in inputs]
blocked = {"torch", "botorch", "gpytorch", "sbi", "numpy", "scipy", "matplotlib", "mutaco", "mutaco_graph", "mutaco_analytic"}
class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError("scientific dependency forbidden: " + fullname, name=fullname)
sys.meta_path.insert(0, NoScience())
def no_process(*a, **k):
    raise AssertionError("no subprocess, scheduler or transfer is permitted")
def allowed(path):
    path = Path(path).resolve()
    if any(path.is_relative_to(base) for base in inputs):
        return True
    if path.suffix in {".py", ".pyc"} and any(path.is_relative_to(root / p) for p in ("tools", "src/tools", "infrastructure", "reproducibility")):
        return True
    return path.is_relative_to(Path(sys.base_prefix))
def checked(original):
    def open_file(path, mode="r", *a, **k):
        if not isinstance(path, int):
            assert allowed(path), "unexpected file access: " + str(path)
            assert not (readonly and any(c in mode for c in "wax+")), "read-only command attempted a write"
        return original(path, mode, *a, **k)
    return open_file
out, err = io.StringIO(), io.StringIO()
with ExitStack() as stack:
    stack.enter_context(patch.object(subprocess, "Popen", no_process))
    stack.enter_context(patch.object(builtins, "open", checked(builtins.open)))
    stack.enter_context(patch.object(io, "open", checked(io.open)))
    sys.argv = [module, *arguments]
    with redirect_stdout(out), redirect_stderr(err):
        try:
            runpy.run_module(module, run_name="__main__")
        except SystemExit as error:
            code = error.code
        except Exception as error:
            code = 1
            print(type(error).__name__ + (": " + str(error) if str(error) else ""), file=err)
        else:
            code = 0
    assert not any(name in sys.modules for name in blocked)
print(json.dumps({"exit_code": code, "stdout": out.getvalue(), "last_error_line": err.getvalue().splitlines()[-1] if err.getvalue() else None}))
'''


def run(module: str, arguments: list[str], temporary: Path, *, readonly: bool, source: Path = ROOT) -> dict:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    inputs = [temporary, ROOT / "results/reproducibility/shareable_milestone_v1",
              ROOT / "results/reproducibility/b100_regression_v1"]
    payload = json.dumps([module, arguments, str(source), [str(p) for p in inputs], readonly])
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", PROBE, payload], cwd=temporary,
                            env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def case_arguments(case: str, temporary: Path) -> tuple[str, list[str], list[str]]:
    if case in {"b40", "b100"}:
        package = "shareable_milestone_v1" if case == "b40" else "b100_regression_v1"
        module = "shareable_evidence" if case == "b40" else "b100_evidence"
        flags = ["--directory", str(ROOT / "results/reproducibility" / package)]
        return "infrastructure.truba." + module, ["verify", *flags], [case, *flags]
    if case.startswith("history"):
        budget = 40 if case == "history40" else 100
        if budget == 40:
            first = second = ROOT / "results/reproducibility/shareable_milestone_v1/b40-replay-mathematics.json.gz"
        else:
            first = ROOT / "results/reproducibility/b100_regression_v1/baseline-replay-mathematics.json.gz"
            second = first.with_name("refactor-replay-mathematics.json.gz")
        args = ["history", "--budget", str(budget), "--first", str(first), "--second", str(second)]
        return "infrastructure.truba.readability_regression", args, args
    if case == "tests":
        paths = test_gate_inputs(temporary / "test-inputs")
        args = ["tests", "--source-root", str(paths["source"]), "--references", str(paths["references"]),
                "--input-manifest", str(paths["manifest"]), "--output-root", str(paths["output"])]
        return "infrastructure.truba.readability_regression", args, args
    if case == "fit-events":
        history, observed = fit_inputs()
        a, b = temporary / "history.json", temporary / "events.json"
        write_json(a, history); write_json(b, observed)
        args = ["fit-events", "--first", str(a), "--second", str(b)]
        return "infrastructure.truba.readability_regression", args, args
    a, b = temporary / "a.json.gz", temporary / "b.json.gz"
    write_json(a, checkpoint_fixture()); write_json(b, checkpoint_fixture())
    flags = ["--first", str(a), "--second", str(b)]
    return "infrastructure.truba.refactor_reference", ["compare", *flags, "--output", str(temporary / "report.json")], ["checkpoints", *flags]


@pytest.mark.parametrize("case", sorted(BEFORE["cases"]))
def test_old_new_cli_match_independent_pre_change_stdout_and_exit(case: str, tmp_path: Path, legacy_checkout: Path) -> None:
    module, old, new = case_arguments(case, tmp_path)
    expected = BEFORE["cases"][case]
    assert run(module, old, tmp_path, readonly=False) == expected
    assert run("tools.reproducibility.verify", new, tmp_path, readonly=True, source=legacy_checkout) == expected
    assert run("tools.reproducibility.verify", new, tmp_path, readonly=True) == expected
    assert run("reproducibility.verify", new, tmp_path, readonly=True) == expected


@pytest.mark.parametrize("case", ["b40", "b100", "checkpoints", "history40", "fit-events", "tests"])
def test_cli_failure_results_exit_codes_and_messages_remain_exact(case: str, tmp_path: Path, legacy_checkout: Path) -> None:
    module, old, new = case_arguments(case, tmp_path)
    if case in {"b40", "b100"}:
        # A missing explicitly supplied directory must fail, never fall back to private inputs.
        old[-1] = new[-1] = str(tmp_path / "missing-package")
    elif case == "checkpoints":
        path = tmp_path / "b.json.gz"
        changed = checkpoint_fixture()
        changed["mathematical_checkpoints"]["observations"] = 39
        write_json(path, changed)
    elif case == "history40":
        source = Path(old[old.index("--first") + 1])
        changed = deepcopy(api.load(source)); changed["candidate"] = [-0.0]
        a, b = tmp_path / "history-a.json.gz", tmp_path / "history-b.json.gz"
        write_json(a, changed); changed["candidate"] = [0.0]; write_json(b, changed)
        for args in (old, new):
            args[args.index("--first") + 1] = str(a)
            args[args.index("--second") + 1] = str(b)
    elif case == "fit-events":
        history, observed = fit_inputs()
        observed["events"][1]["message"] = "wrong"
        write_json(tmp_path / "events.json", observed)
    else:
        source = tmp_path / "test-inputs/references/accepted-tests.xml"
        source.write_text("changed reference bytes")
    original = run(module, old, tmp_path, readonly=False)
    assert original["exit_code"] in (1, 2)
    assert "unexpected file access" not in (original["last_error_line"] or "")
    assert "read-only command attempted" not in (original["last_error_line"] or "")
    assert run("tools.reproducibility.verify", new, tmp_path, readonly=True, source=legacy_checkout) == original
    assert run("tools.reproducibility.verify", new, tmp_path, readonly=True) == original
    assert run("reproducibility.verify", new, tmp_path, readonly=True) == original


@pytest.mark.parametrize("action", ["capture", "historical", "derive", "export", "finish", "submit"])
def test_new_entrypoint_rejects_producer_actions_without_any_work(action, tmp_path) -> None:
    result = run("reproducibility.verify", [action], tmp_path, readonly=True)
    assert result["exit_code"] == 2 and result["stdout"] == ""


def test_help_and_required_explicit_paths_are_safe(tmp_path) -> None:
    result = run("reproducibility.verify", ["--help"], tmp_path, readonly=True)
    assert result["exit_code"] == 0
    assert "No GP fitting" in " ".join(result["stdout"].split()) and "tests" in result["stdout"]
    result = run("reproducibility.verify", ["tests"], tmp_path, readonly=True)
    assert result["exit_code"] == 2
    assert "--source-root" in result["last_error_line"] and "--references" in result["last_error_line"]
