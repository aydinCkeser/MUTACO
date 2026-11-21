"""Locate archived repository inputs without changing the addresses recorded in evidence."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

RETIRED_ROOTS = frozenset({"benchmarks", "configs", "data", "packages", "project", "scientific", "tools"})
_LEDGER = Path(__file__).resolve().parents[1] / "docs/decisions/repository_migration_manifest.json"


@lru_cache(maxsize=1)
def _recorded_locations() -> tuple[dict[Path, Path], dict[Path, Path]]:
    """Recover file bindings and unambiguous directory bindings from the reviewed ledger."""
    files = {}
    directories: dict[Path, set[Path | None]] = {}
    for row in json.loads(_LEDGER.read_text(encoding="utf-8"))["files"]:
        old = Path(row["base_path"])
        if old.parts[0] not in RETIRED_ROOTS:
            continue
        owner = Path(row["owner"])
        if old.as_posix() == "tools/reproducibility/tensor_export.py":
            owner = Path("reproducibility/tensor_export.py")
        if owner.parts[0] in RETIRED_ROOTS:
            continue  # Retired package markers have no current owner.
        files[old] = owner
        for directory in old.parents:
            if not directory.parts:
                break
            suffix = old.relative_to(directory).parts
            # A common old directory may have been split between several owners.
            prefix = owner.parents[len(suffix) - 1] if owner.parts[-len(suffix):] == suffix else None
            directories.setdefault(directory, set()).add(prefix)
    consistent = {
        directory: next(iter(targets)) for directory, targets in directories.items()
        if len(targets) == 1 and None not in targets
    }
    return files, consistent


def resolve_recorded_path(repository_root: Path, recorded: str | Path) -> Path:
    """Bind a stored input address to today's checkout for reading.

    Keep the original string for identity hashing, serialization and historical Git
    queries. Ordinary current paths and absolute paths to external inputs pass
    through. An ambiguous retired directory requires its specific recorded member.
    """
    root = Path(repository_root)
    path = Path(recorded)
    if path.is_absolute():
        try:
            path = path.relative_to(root.absolute())
        except ValueError:
            return path
    if not path.parts or path.parts[0] not in RETIRED_ROOTS:
        return root / path
    if ".." in path.parts:
        raise ValueError(f"parent traversal in archived repository address: {recorded}")
    files, directories = _recorded_locations()
    if path in files:
        return root / files[path]
    if path in directories:
        return root / directories[path]
    for parent in path.parents:
        if parent in directories:
            return root / directories[parent] / path.relative_to(parent)
    raise FileNotFoundError(f"no unambiguous current owner for archived address: {recorded}")
