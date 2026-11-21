"""Materialize compact repaired-SBI recovery and replay evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.paper.common.evidence_analysis import canonical_json_text
from .repaired_sbi_recovery import build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    print(
        canonical_json_text(
            build(repository_root=arguments.repository_root, check=arguments.check)
        ),
        end="",
    )


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.repaired_sbi_recovery_cli")


if __name__ == "__main__":
    main()
