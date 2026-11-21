"""Render one stored-data figure into a new version beneath results/paper/figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory


FIGURES = ("primary-response", "primary-parameters", "sbi-response", "terminal-domain")


def _render(figure: str, repository: Path, staging: Path) -> tuple[Path, Path]:
    """Pass the original producer its validated data and a temporary output directory."""
    if figure == "primary-response":
        from . import plot_paper120_primary_boxplot_v1 as plot

        values, _ = plot.load_frozen_values(repository / "results/paper/scalar_primary_b40_b100_synthesis_v1")
        return plot.render_deterministically(values, staging)
    if figure == "primary-parameters":
        from . import plot_paper120_primary_parameter_boxplot_v1 as plot

        return plot.render_deterministically(plot.load_parameter_errors(repository), staging)
    if figure == "sbi-response":
        from . import plot_paper120_sbi_response_boxplot_v1 as plot

        return plot.render_deterministically(plot.load_frozen_values(repository), staging)
    if figure == "terminal-domain":
        from . import plot_paper120_terminal_domain_ablation_v1 as plot

        values, deltas, interval = plot.load_frozen_evidence(repository)
        return plot.render_deterministically(values, deltas, interval, staging)
    raise ValueError(f"unknown persisted-input figure: {figure!r}")


def render_versioned(figure: str, version: str, repository: Path) -> Path:
    """Publish both checked images in an exclusively created figure version."""
    if figure not in FIGURES or not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("select a supported figure and a version such as v1")
    destination = repository.resolve() / "results/paper/figures" / figure / version
    if destination.resolve() != destination or destination.is_symlink():
        raise ValueError("the new results destination must not traverse a symbolic link")
    if destination.exists():
        raise FileExistsError(f"figure version already exists: {destination}")

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    with TemporaryDirectory(prefix="mutaco-figure-") as temporary:
        artifacts = _render(figure, repository, Path(temporary))
        # A concurrent render can finish first. mkdir must fail in that case.
        if destination.resolve() != destination:
            raise ValueError("the results destination changed during rendering")
        destination.mkdir(parents=True, exist_ok=False)
        published = []
        try:
            for artifact in artifacts:
                path = destination / artifact.name
                with artifact.open("rb") as source, path.open("xb") as output:
                    published.append(path)
                    shutil.copyfileobj(source, output)
        except BaseException:
            for path in published:
                path.unlink(missing_ok=True)
            destination.rmdir()
            raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=FIGURES)
    parser.add_argument("--version", required=True, help="fresh result version, e.g. v1")
    args = parser.parse_args()
    try:
        destination = render_versioned(args.figure, args.version, Path(__file__).resolve().parents[3])
    except (FileExistsError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(destination)


if __name__ == "__main__":
    main()
