"""Local runtime and pytest-path checks for a Dropbox-hosted checkout."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


FORBIDDEN_ROOT_TEMP_PATTERNS = (".pytest_*", ".pytest_cache", ".pt")


def _resolved(path: str | Path, *, relative_to: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    return candidate.resolve(strict=False)


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def validate_external_pytest_temp(
    repository_root: Path,
    *,
    basetemp: str | Path | None,
    debug_temproot: str | Path | None,
    system_temp: str | Path,
) -> tuple[Path, ...]:
    """Reject every pytest temporary root that resolves inside the checkout."""

    root = repository_root.resolve()
    candidates: list[tuple[str, Path]] = []
    if basetemp is not None:
        candidates.append(("--basetemp", _resolved(basetemp, relative_to=root)))
    if debug_temproot:
        candidates.append(
            ("PYTEST_DEBUG_TEMPROOT", _resolved(debug_temproot, relative_to=root))
        )
    candidates.append(("system temporary directory", _resolved(system_temp, relative_to=root)))
    for source, path in candidates:
        if _inside(path, root):
            raise ValueError(
                f"{source} resolves inside the Dropbox repository ({path}); "
                "use infrastructure/local/run_pytest.ps1 so test data stays under LOCALAPPDATA"
            )
    return tuple(path for _, path in candidates)


def forbidden_repository_temp_directories(repository_root: Path) -> tuple[Path, ...]:
    root = repository_root.resolve()
    found: set[Path] = set()
    for pattern in FORBIDDEN_ROOT_TEMP_PATTERNS:
        found.update(path for path in root.glob(pattern) if path.is_dir())
    return tuple(sorted(found, key=lambda path: path.name))


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.name for path in paths)
