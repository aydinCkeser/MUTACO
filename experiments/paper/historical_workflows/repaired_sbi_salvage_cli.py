"""Materialize the zero-call repaired MUTACO--SBI salvage design audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from experiments.paper.common.evidence_analysis import canonical_json_text
from .repaired_sbi_salvage import inventory_salvage, salvage_design, validate_salvage_design
from experiments.paper.protocol.validation import PaperContractError


SPEC = "experiments/paper/studies/graph_paper120/analysis/repaired_mutaco_sbi_salvage_v1.json"
OUTPUT = "results/paper/repaired_mutaco_sbi_salvage_design/v1"


def _content(root: Path) -> dict[str, str]:
    specification = validate_salvage_design(read_json(root / SPEC))
    inventory = inventory_salvage(root, specification)
    inventory_text = canonical_json_text(inventory)
    body = {
        "schema_version": "mutaco-repaired-sbi-salvage-design-package-v1",
        "design_identity": specification["design_identity"],
        "inventory_identity": inventory["inventory_identity"],
        "inventory_content_identity": stable_hash("repaired-sbi-salvage-inventory-file-v1", inventory_text),
        "b40_classification": inventory["b40"]["current_classification"],
        "b100_classification": inventory["b100"]["current_classification"],
        "best_case_new_calls": inventory["combined_call_preview"]["best_case"],
        "worst_case_new_calls": inventory["combined_call_preview"]["worst_case"],
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    manifest = {
        **body,
        "package_identity": stable_hash("repaired-sbi-salvage-design-package-v1", body),
    }
    return {
        "inventory.json": inventory_text,
        "manifest.json": canonical_json_text(manifest),
    }


def build(*, repository_root: Path, check: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    output = root / OUTPUT
    contents = _content(root)
    if check:
        for name, expected in contents.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise PaperContractError(f"tracked salvage-design artifact differs: {name}")
            sidecar = output / f"{name}.sha256"
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked salvage-design sidecar differs: {name}")
        status = "reproduced"
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            (output / f"{name}.sha256").write_text(file_hash(path) + "\n", encoding="ascii", newline="\n")
        status = "materialized"
    manifest = json.loads(contents["manifest.json"])
    return {
        "status": status,
        "output": OUTPUT,
        "package_identity": manifest["package_identity"],
        "best_case_new_calls": manifest["best_case_new_calls"],
        "worst_case_new_calls": manifest["worst_case_new_calls"],
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-spec", action="store_true")
    arguments = parser.parse_args()
    if arguments.print_spec:
        print(canonical_json_text(salvage_design()), end="")
        return
    print(canonical_json_text(build(repository_root=arguments.repository_root, check=arguments.check)), end="")


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.repaired_sbi_salvage_cli")


if __name__ == "__main__":
    main()
