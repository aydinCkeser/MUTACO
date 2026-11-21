"""Render target-98 response panels from frozen arrays; no simulator import."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.paper.exploration.target098_response_archive import (
    ARCHIVE_NAME,
    MANIFEST_NAME,
    render_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    outputs = render_figures(
        args.archive_directory / ARCHIVE_NAME,
        args.archive_directory / MANIFEST_NAME,
        args.output_directory,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
