"""Materialize or verify a generic no-response v2 theta declaration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.benchmarking.artifacts import file_hash, write_pretty_json
from experiments.paper.protocol.cohort_v2 import (
    CohortDeclarationSpec,
    materialize_cohort_declaration,
    validate_cohort_declaration,
)


def _spec(document: dict[str, object]) -> CohortDeclarationSpec:
    return CohortDeclarationSpec(
        cohort_id=str(document["cohort_id"]),
        simulator_adapter=str(document["simulator_adapter"]),
        simulator_definition_id=str(document["simulator_definition_id"]),
        simulator_configuration_identity=str(document["simulator_configuration_identity"]),
        simulator_configuration_sha256=str(document["simulator_configuration_sha256"]),
        parameter_profile=str(document["parameter_profile"]),
        parameter_names=tuple(document["parameter_names"]),
        bounds=tuple(tuple(row) for row in document["bounds"]),
        target_count=document["target_count"],
        campaign_seed=document["campaign_seed"],
        target_seed_namespace=str(document["target_seed_namespace"]),
        sobol_skip=document.get("sobol_skip", 0),
        sobol_scramble=document.get("sobol_scramble", True),
        dtype=str(document.get("dtype", "float32")),
        generation_device=str(document.get("generation_device", "cpu")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    document = materialize_cohort_declaration(
        _spec(json.loads(args.spec.read_text(encoding="utf-8")))
    )
    validate_cohort_declaration(document)
    print(json.dumps({
        "target_count": document["target_count"],
        "target_family_id": document["target_family_id"],
        "declaration_hash": document["declaration_hash"],
        "simulator_calls": 0,
    }, indent=2))
    if not args.write:
        sidecar = args.output.with_suffix(".sha256")
        if args.output.exists() or sidecar.exists():
            if not args.output.is_file() or not sidecar.is_file():
                raise SystemExit("incomplete existing declaration")
            if file_hash(args.output) != sidecar.read_text(encoding="ascii").strip():
                raise SystemExit("existing declaration sidecar hash mismatch")
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if existing != document:
                raise SystemExit("existing declaration differs from exact regeneration")
        return
    if not args.confirm_approved:
        raise SystemExit("declaration writing requires --confirm-approved")
    if args.output.exists() or args.output.with_suffix(".sha256").exists():
        raise SystemExit("append-only declaration output already exists")
    write_pretty_json(args.output, document)
    args.output.with_suffix(".sha256").write_text(file_hash(args.output), encoding="ascii")


if __name__ == "__main__":
    main()
