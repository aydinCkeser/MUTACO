"""Check path-only edits against the qualified sources before their AST hash checks."""

import ast
from pathlib import Path
import subprocess


QUALIFIED = "16d5e5c4787c29c5daf176881f57ba8d39468530"


PATH_BINDINGS = (
    ("benchmarks/paper/", "results/paper/"),
    ("data/cohorts/paper120_v3", "experiments/graph_inversion/frozen_targets/paper120_v3"),
    ("benchmarks/graph/paper120/", "results/paper/graph_paper120/paper120/"),
)


def original_ast_after_checking_path_migration(
    root: Path, source: Path, *, substitutions=PATH_BINDINGS,
) -> ast.Module:
    """All current AST nodes must match the historical source with only these path bindings."""
    archived = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{QUALIFIED}:{source.relative_to(root).as_posix()}"],
        text=True,
    )
    expected = archived
    for old, new in substitutions:
        expected = expected.replace(old, new)
    assert ast.dump(ast.parse(source.read_text())) == ast.dump(ast.parse(expected))
    return ast.parse(archived)
