"""Run a zero-call replay script against an explicitly selected source tree.

The pinned Windows launcher validates the active checkout before entering this
bootstrap.  This module then replaces only the Python import roots, verifies
that ``mutaco`` resolves from the requested detached worktree, and executes the
replay script.  It does not import any experiment or simulator module.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--script-arguments-json", required=True)
    arguments = parser.parse_args()

    source_root = arguments.source_root.resolve()
    if _git(source_root, "rev-parse", "HEAD") != arguments.source_commit:
        raise SystemExit("exact-source worktree HEAD mismatch")
    if _git(source_root, "status", "--porcelain"):
        raise SystemExit("exact-source worktree is not clean")

    script = arguments.script.resolve()
    selected_roots = [
        source_root / "src",
        source_root / "packages" / "analytic-simulator" / "src",
        source_root / "packages" / "graph-simulator" / "src",
    ]
    for name in tuple(sys.modules):
        if name == "mutaco" or name.startswith("mutaco."):
            del sys.modules[name]
    active_root = Path(__file__).resolve().parents[3]

    def outside_active_checkout(item: str) -> bool:
        if not item:
            return False
        try:
            Path(item).resolve().relative_to(active_root)
        except ValueError:
            return True
        return False

    sys.path[:] = [str(path) for path in selected_roots] + [str(script.parent)] + [
        item for item in sys.path if outside_active_checkout(item)
    ]

    import mutaco

    module_path = Path(mutaco.__file__).resolve()
    if source_root not in module_path.parents:
        raise SystemExit(f"mutaco did not resolve from exact source: {module_path}")

    script_arguments = json.loads(arguments.script_arguments_json)
    if not isinstance(script_arguments, list) or not all(
        isinstance(item, str) for item in script_arguments
    ):
        raise SystemExit("script arguments must be a JSON string list")
    sys.argv = [str(script), *script_arguments]
    runpy.run_path(str(script), run_name="__main__")


# Keep historical Python GLOBAL references for trusted serialized records.
for _legacy_export in (
    _git,
    main,
):
    _legacy_export.__module__ = "experiments.paper.common.exact_source_bootstrap"
del _legacy_export


if __name__ == "__main__":
    main()
