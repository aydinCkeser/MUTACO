"""Preview or execute one strict shard of a graph v3 paper population."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import torch
import mutaco as mutaco_package
import mutaco_graph

from mutaco import stable_hash
from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.graph_inversion.paper_v3_callbacks import (
    GraphPopulationCallbacks,
    method_configuration_identity,
)
from experiments.paper.common.population_v3 import (
    collect_population_v3,
    execute_population_v3,
    load_population_declaration_v3,
    population_call_preview,
    shard_indices,
)
from experiments.paper.protocol.method_specs import (
    load_method_execution_specs,
    selected_method_configuration_paths,
)
from experiments.paper.protocol.supplied_targets import load_supplied_target_set
from experiments.paper.common.validation import PaperContractError


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_hashed(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pretty_json(path, dict(document))
    path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


def _write_worker_terminal(
    output: Path,
    *,
    status: str,
    phase: str,
    source_commit: str,
    declaration_sha256: str,
    physical_calls: Mapping[str, Any] | None,
    failure: BaseException | None = None,
) -> None:
    document = {
        "schema_version": "mutaco-graph-v3-worker-terminal-v1",
        "status": status,
        "phase": phase,
        "source_commit": source_commit,
        "declaration_sha256": declaration_sha256,
        "physical_calls": dict(physical_calls) if physical_calls is not None else None,
        "failure_type": type(failure).__name__ if failure is not None else None,
        "failure_explanation": str(failure) if failure is not None else None,
        "automatic_retry": False,
        "scientific_calls_added_by_terminal_evidence": 0,
    }
    _write_hashed(output / "worker_terminal.json", document)


def _validate_thread_configuration(*, intraop_threads: int, interop_threads: int) -> None:
    if intraop_threads < 1:
        raise SystemExit("worker requires a positive Torch intra-op thread count")
    expected = str(intraop_threads)
    thread_environment = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    if any(value != expected for value in thread_environment.values()):
        raise SystemExit(
            "worker requires OMP/MKL/OPENBLAS thread counts matching "
            f"Torch intra-op threads ({expected})"
        )
    if interop_threads != 1:
        raise SystemExit("worker requires Torch inter-op threads of 1")


def _checked_file(path: Path, expected: str | None = None) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = file_hash(resolved)
    sidecar = resolved.with_suffix(".sha256")
    if sidecar.is_file() and sidecar.read_text(encoding="ascii").strip() != observed:
        raise PaperContractError(f"hash sidecar mismatch: {resolved}")
    if expected is not None and observed != expected:
        raise PaperContractError(f"declared file hash mismatch: {resolved}")
    return resolved


def _target_records(
    target_ledger: Path,
    declaration: Any,
    indices: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    manifest_path = _checked_file(target_ledger / "manifest.json", declaration.target_ledger_hash)
    manifest = read_json(manifest_path)
    manifest_records = manifest.get("records")
    if not isinstance(manifest_records, Mapping):
        raise PaperContractError("target ledger manifest records are malformed")
    identity_to_index: dict[str, int] = {}
    for raw_index, entry in manifest_records.items():
        if not isinstance(entry, Mapping):
            raise PaperContractError("target ledger manifest entry is malformed")
        identity = str(entry.get("target_record_id", ""))
        if not identity or identity in identity_to_index:
            raise PaperContractError("target ledger identities must be nonempty and unique")
        try:
            identity_to_index[identity] = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise PaperContractError("target ledger position is malformed") from exc
    rows = []
    for position in indices:
        identity = declaration.target_record_identities[position]
        if identity not in identity_to_index:
            raise PaperContractError("declared target is absent from frozen target ledger")
        index = identity_to_index[identity]
        candidates = (
            target_ledger / f"target_{index:03d}.json",
            target_ledger / f"target_{index:04d}.json",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"missing frozen target position {index}")
        _checked_file(path)
        row = read_json(path)
        if row.get("target_record_id") != identity:
            raise PaperContractError("target record identity differs from declaration")
        rows.append(dict(row))
    return tuple(rows)


def _supplied_target_records(
    path: Path, declaration: Any, indices: tuple[int, ...]
) -> tuple[dict[str, Any], ...]:
    checked = _checked_file(path, declaration.target_ledger_hash)
    supplied = load_supplied_target_set(
        read_json(checked), comparison=declaration.comparison
    )
    by_identity = {row["target_record_id"]: row for row in supplied}
    selected = []
    for position in indices:
        identity = declaration.target_record_identities[position]
        if identity not in by_identity:
            raise PaperContractError("declared target is absent from supplied target set")
        selected.append(dict(by_identity[identity]))
    return tuple(selected)


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "sbi", "torch", "botorch", "gpytorch", "numpy", "scipy", "nflows",
        "skorch", "tensorboard", "zuko",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_outside_scientific_identity": sys.executable,
        "packages": packages,
        "torch_device": "cpu",
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "source_module_origins": {
            "mutaco": Path(mutaco_package.__file__).resolve().as_posix(),
            "mutaco_graph": Path(mutaco_graph.__file__).resolve().as_posix(),
        },
    }


def _inference_contexts_are_distinct(
    contexts: list[Mapping[str, Any]],
) -> bool:
    identities = [
        stable_hash("paper-v3-inference-context-validation-v1", dict(context))
        for context in contexts
    ]
    return len(set(identities)) == len(identities)


def _validate_paired_contexts(
    targets: tuple[Mapping[str, Any], ...], rows: tuple[Mapping[str, Any], ...]
) -> None:
    for target in targets:
        pair = [
            row
            for row in rows
            if row["target_record_identity"] == target["target_record_id"]
        ]
        inference_contexts = [row["decision"]["inference_context"] for row in pair]
        if not _inference_contexts_are_distinct(inference_contexts):
            raise RuntimeError("v3 method inference contexts are not distinct")
        completed = [row for row in pair if row["status"] == "completed"]
        if completed and any(
            row["assessment_context"] != completed[0]["assessment_context"]
            for row in completed[1:]
        ):
            raise RuntimeError("v3 paired assessment contexts are not identical")
        evidence = [row.get("assessment_randomness_evidence") for row in completed]
        if evidence and any(not isinstance(item, Mapping) for item in evidence):
            raise RuntimeError("graph assessment randomness evidence is missing")
        if evidence and any(
            item.get("coordinate_family_identities")
            != evidence[0].get("coordinate_family_identities")
            or item.get("namespace_seed_manifest_hashes")
            != evidence[0].get("namespace_seed_manifest_hashes")
            for item in evidence[1:]
        ):
            raise RuntimeError("v3 paired assessment randomness namespaces differ")


def _validate_source_module_origins(root: Path) -> dict[str, str]:
    origins = {
        "mutaco": Path(mutaco_package.__file__).resolve(),
        "mutaco_graph": Path(mutaco_graph.__file__).resolve(),
    }
    for name, origin in origins.items():
        try:
            origin.relative_to(root)
        except ValueError as exc:
            raise SystemExit(
                f"{name} resolved outside the exact qualification checkout: {origin}"
            ) from exc
    return {name: origin.as_posix() for name, origin in origins.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--method-specs", type=Path, required=True)
    target_source = parser.add_mutually_exclusive_group(required=True)
    target_source.add_argument("--target-ledger", type=Path)
    target_source.add_argument("--supplied-targets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--maximum-calls", type=int, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--torch-intraop-threads", type=int, required=True)
    parser.add_argument("--torch-interop-threads", type=int, required=True)
    parser.add_argument("--require-all-successful", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()

    declaration_path = _checked_file(args.declaration)
    declaration = load_population_declaration_v3(read_json(declaration_path))
    root = Path(__file__).resolve().parents[2]
    method_spec_path = _checked_file(args.method_specs)
    method_specs = load_method_execution_specs(
        read_json(method_spec_path), source=method_spec_path
    )
    declared_methods = tuple(method.method for method in declaration.methods)
    method_paths = selected_method_configuration_paths(
        method_specs, declaration.methods, root=root
    )
    for method in declaration.methods:
        if method.configuration_identity != method_configuration_identity(
            method.method, method_paths[method.method]
        ):
            raise SystemExit(f"{method.method} configuration hash differs from declaration")
    selected = shard_indices(
        len(declaration.target_record_identities), rank=args.rank, workers=args.workers
    )
    targets = (
        _target_records(args.target_ledger.resolve(), declaration, selected)
        if args.target_ledger is not None
        else _supplied_target_records(args.supplied_targets.resolve(), declaration, selected)
    )
    preview = population_call_preview(len(targets), declaration.methods)
    preview_document = {
        "schema_version": "mutaco-graph-v3-worker-preview-v1",
        "campaign_id": declaration.campaign_id,
        "rank": args.rank,
        "workers": args.workers,
        "target_positions": list(selected),
        "target_indices": [int(target["target_index"]) for target in targets],
        "target_source": "frozen_ledger" if args.target_ledger is not None else "supplied_frozen_y_star",
        "method_spec_sha256": file_hash(method_spec_path),
        "methods": list(declared_methods),
        "call_preview": preview,
        "maximum_calls_argument": args.maximum_calls,
        "scientific_calls_executed": 0,
    }
    print(json.dumps(preview_document, indent=2))
    if args.maximum_calls != preview["maximum_total_calls"]:
        raise SystemExit("worker requires its exact shard call cap")
    if not args.execute:
        return
    if not args.confirm_approved:
        raise SystemExit("worker execution requires --confirm-approved")
    if args.expected_commit is None:
        raise SystemExit("worker execution requires --expected-commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("worker execution requires the exact clean expected commit")
    _validate_source_module_origins(root)
    if args.output.exists():
        raise SystemExit("append-only worker output already exists")
    _validate_thread_configuration(
        intraop_threads=args.torch_intraop_threads,
        interop_threads=args.torch_interop_threads,
    )
    torch.set_num_threads(args.torch_intraop_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    environment = _environment()
    if environment["packages"]["sbi"] != "0.26.1":
        raise SystemExit("worker requires exact sbi 0.26.1")
    if environment["torch_cuda_available"]:
        raise SystemExit("graph v3 Linux qualification is CPU-only and does not qualify CUDA")

    output = args.output.resolve()
    output.mkdir(parents=True)
    _write_hashed(
        output / "worker_start.json",
        {
            "schema_version": "mutaco-graph-v3-worker-start-v1",
            "started_utc": _utc_now(),
            "source_commit": head,
            "declaration_path": declaration_path.as_posix(),
            "declaration_sha256": file_hash(declaration_path),
            "method_spec_path": method_spec_path.as_posix(),
            "method_spec_sha256": file_hash(method_spec_path),
            "target_source_sha256": declaration.target_ledger_hash,
            "method_configuration_sha256": {
                method: file_hash(path) for method, path in method_paths.items()
            },
            "preview": preview_document,
            "environment": environment,
            "no_automatic_retry": True,
        },
    )
    callbacks = GraphPopulationCallbacks(
        method_configurations=method_paths,
        output=output,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    calls: Mapping[str, Any] | None = None
    phase = "population_execution"
    try:
        result = execute_population_v3(
            comparison=declaration.comparison,
            targets=targets,
            methods=declaration.methods,
            output=output,
            maximum_calls=args.maximum_calls,
            run_method=callbacks.run_method,
            evaluate_inference=callbacks.evaluate_inference,
            evaluate_assessment=callbacks.evaluate_assessment,
            execution_provenance={
                "source_commit": head,
                "declaration_identity": declaration.document()["declaration_identity"],
                "declaration_sha256": file_hash(declaration_path),
            },
        )
        calls = result["physical_calls"]
        phase = "worker_collection"
        collection, rows = collect_population_v3(
            output,
            target_identities=tuple(str(target["target_record_id"]) for target in targets),
            methods=declared_methods,
        )
        if not collection["complete"]:
            raise RuntimeError("worker target-method collection is not complete")
        if args.require_all_successful:
            phase = "exact_successful_cap_validation"
            if calls != {
                "maximum_calls": args.maximum_calls,
                "attempted_calls": args.maximum_calls,
                "successful_calls": args.maximum_calls,
                "failed_calls": 0,
            } or any(row["status"] != "completed" for row in rows):
                raise RuntimeError(
                    f"qualification did not complete every slot at its successful cap: {calls}"
                )
        phase = "paired_context_validation"
        _validate_paired_contexts(targets, rows)
    except Exception as exc:
        _write_worker_terminal(
            output,
            status="failed",
            phase=phase,
            source_commit=head,
            declaration_sha256=file_hash(declaration_path),
            physical_calls=calls,
            failure=exc,
        )
        raise

    validation = {
        "schema_version": (
            "mutaco-graph-v3-linux-qualification-validation-v1"
            if args.require_all_successful
            else "mutaco-graph-v3-worker-validation-v1"
        ),
        "status": "passed",
        "mode": "qualification" if args.require_all_successful else "population_shard",
        "source_commit": head,
        "target_construction_calls": 0,
        "recommendation_calls": 0,
        "physical_calls": calls,
        "all_calls_successful": calls["failed_calls"] == 0,
        "completed_slots": sum(row["status"] == "completed" for row in rows),
        "failed_slots": sum(row["status"] == "failed" for row in rows),
        "all_method_decisions_frozen_before_assessment": True,
        "paired_context_validation": "passed",
        "sbi_version": environment["packages"]["sbi"],
        "device": "cpu",
        "no_retry": True,
        "no_performance_claim": True,
    }
    if args.require_all_successful:
        validation.update({
            "mechanics_only": True,
            "paper120_method_calls": 0,
            "exact_physical_calls": args.maximum_calls,
        })
    validation_path = (
        output / "qualification_validation.json"
        if args.require_all_successful
        else output / "worker_validation.json"
    )
    _write_hashed(validation_path, validation)
    _write_worker_terminal(
        output,
        status="completed",
        phase=(
            "qualification_validation_persisted"
            if args.require_all_successful
            else "worker_validation_persisted"
        ),
        source_commit=head,
        declaration_sha256=file_hash(declaration_path),
        physical_calls=calls,
    )
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.json":
            inventory.append({
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_hash(path),
            })
    _write_hashed(
        output / "artifact_inventory.json",
        {
            "schema_version": "mutaco-graph-v3-linux-qualification-inventory-v1",
            "generated_utc": _utc_now(),
            "files": inventory,
            "inventory_identity": stable_hash("graph-v3-linux-qualification-inventory-v1", inventory),
        },
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
