"""Execute terminal generation with one verified in-memory source overlay.

The selected historical worktree remains clean.  Only the corrected continuous
optimizer module is replaced in memory before the terminal generator is run.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import sys


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--expected-native-terminal-blob", required=True)
    parser.add_argument("--expected-corrected-terminal-blob", required=True)
    parser.add_argument("--expected-physical-mae-blob", required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--script-arguments-json", required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()

    source_root = arguments.source_root.resolve()
    if _git(source_root, "rev-parse", "HEAD") != arguments.source_commit:
        raise SystemExit("exact terminal source worktree HEAD mismatch")
    if _git(source_root, "status", "--porcelain"):
        raise SystemExit("exact terminal source worktree is not clean")
    terminal_path = "src/mutaco/recommendation/_continuous.py"
    physical_mae_path = "src/mutaco/recommendation/physical_mae.py"
    if _git(source_root, "rev-parse", f"HEAD:{terminal_path}") != arguments.expected_native_terminal_blob:
        raise SystemExit("native terminal blob mismatch")
    if _git(source_root, "rev-parse", f"HEAD:{physical_mae_path}") != arguments.expected_physical_mae_blob:
        raise SystemExit("native physical-MAE blob mismatch")
    overlay = arguments.overlay.resolve()
    if _git(overlay.parent, "hash-object", str(overlay)) != arguments.expected_corrected_terminal_blob:
        raise SystemExit("corrected terminal overlay blob mismatch")

    script = arguments.script.resolve()
    active_root = Path(__file__).resolve().parents[3]
    selected_roots = [
        source_root / "src",
        source_root / "packages" / "analytic-simulator" / "src",
        source_root / "packages" / "graph-simulator" / "src",
    ]

    def outside_active_checkout(item: str) -> bool:
        if not item:
            return False
        try:
            Path(item).resolve().relative_to(active_root)
        except ValueError:
            return True
        return False

    for name in tuple(sys.modules):
        if name == "mutaco" or name.startswith("mutaco."):
            del sys.modules[name]
    sys.path[:] = [str(path) for path in selected_roots] + [str(script.parent)] + [
        item for item in sys.path if outside_active_checkout(item)
    ]

    import mutaco

    if source_root not in Path(mutaco.__file__).resolve().parents:
        raise SystemExit("mutaco did not resolve from the selected native source")
    module_name = "mutaco.recommendation._continuous"
    specification = importlib.util.spec_from_file_location(module_name, overlay)
    if specification is None or specification.loader is None:
        raise SystemExit("unable to load corrected terminal overlay")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    physical_mae = importlib.reload(sys.modules["mutaco.recommendation.physical_mae"])
    loaded_from = Path(inspect.getsourcefile(physical_mae.optimize_continuous_physical) or "").resolve()
    if loaded_from != overlay:
        raise SystemExit("physical-MAE policy did not bind the corrected overlay")

    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated_zero_call",
                    "native_source_commit": arguments.source_commit,
                    "native_terminal_blob": arguments.expected_native_terminal_blob,
                    "corrected_terminal_blob": arguments.expected_corrected_terminal_blob,
                    "physical_mae_blob": arguments.expected_physical_mae_blob,
                    "overlay_loaded_from": str(loaded_from),
                    "terminal_recommendations_generated": 0,
                    "simulator_calls": 0,
                },
                sort_keys=True,
            )
        )
        return

    script_arguments = json.loads(arguments.script_arguments_json)
    if not isinstance(script_arguments, list) or not all(isinstance(item, str) for item in script_arguments):
        raise SystemExit("script arguments must be a JSON string list")
    sys.argv = [str(script), *script_arguments]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
