"""Preview or execute target-only construction from a reviewed v2 theta ledger.

The default operation is computation-free preview.  Execution requires both
``--execute`` and ``--confirm-approved`` and has no method-dispatch surface.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.benchmarking.simulators import get_simulator_adapter
from experiments.benchmarking.artifacts import file_hash
from experiments.paper.protocol.cohort_v2 import validate_cohort_declaration
from experiments.paper.protocol.target_freeze_v2 import (
    execute_target_freeze,
    preview_target_freeze,
)
from mutaco import SimulationContext, checked_evaluate


def _indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-target-calls", type=int, required=True)
    parser.add_argument("--target-indices")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    parser.add_argument("--torch-intraop-threads", type=int)
    parser.add_argument("--torch-interop-threads", type=int)
    args = parser.parse_args()
    declaration_sidecar = args.declaration.with_suffix(".sha256")
    if not declaration_sidecar.is_file() or file_hash(args.declaration) != declaration_sidecar.read_text(encoding="ascii").strip():
        raise SystemExit("declaration file hash sidecar is missing or invalid")
    declaration = validate_cohort_declaration(
        json.loads(args.declaration.read_text(encoding="utf-8"))
    )
    selected = _indices(args.target_indices)
    preview = preview_target_freeze(
        declaration, args.output,
        maximum_calls=args.maximum_target_calls, target_indices=selected,
    )
    if args.config is not None and file_hash(args.config) != declaration["simulator_configuration_sha256"]:
        raise SystemExit("simulator configuration hash differs from declaration")
    print(json.dumps(preview.document(), indent=2))
    if not args.execute:
        return
    if preview.planned_calls > args.maximum_target_calls:
        raise SystemExit("target call cap rejects the preview before execution")
    if not args.confirm_approved:
        raise SystemExit("target execution requires --confirm-approved")
    if args.config is None:
        raise SystemExit("target execution requires --config")
    if (
        args.torch_intraop_threads is None
        or args.torch_interop_threads is None
        or args.torch_intraop_threads < 1
        or args.torch_interop_threads < 1
    ):
        raise SystemExit(
            "target execution requires positive --torch-intraop-threads and --torch-interop-threads"
        )
    torch.set_num_threads(args.torch_intraop_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    config: Mapping[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = get_simulator_adapter(str(declaration["simulator_adapter"]))
    device, dtype = torch.device("cpu"), torch.float32
    build_target_simulator = getattr(adapter, "build_target_simulator", None)
    if build_target_simulator is None:
        raise SystemExit(
            f"simulator adapter {declaration['simulator_adapter']!r} has no target-only construction surface"
        )
    simulator = build_target_simulator(config, device=device, dtype=dtype)
    root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if dirty:
        raise SystemExit("target execution requires an exact clean Git commit")
    provenance = {
        "source_commit": source_commit,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "configuration_path": args.config.as_posix(),
        "configuration_sha256": file_hash(args.config),
        "declaration_path": args.declaration.as_posix(),
        "declaration_file_sha256": file_hash(args.declaration),
        "interpreter_path_recorded_outside_scientific_identity": sys.executable,
    }

    def evaluate(record: Mapping[str, Any]):
        result = checked_evaluate(
            simulator,
            torch.tensor(record["theta_star"], device=device, dtype=dtype),
            device=device,
            dtype=dtype,
            context=SimulationContext(
                str(declaration["target_family_id"]),
                "target",
                int(record["target_index"]),
                0,
                int(record["target_generation_seed"]),
            ),
        )
        return result.values.detach().cpu().tolist(), result.metadata

    result = execute_target_freeze(
        declaration, args.output,
        maximum_calls=args.maximum_target_calls,
        target_indices=selected,
        evaluate_target=evaluate,
        execution_provenance=provenance,
    )
    print(json.dumps(result["manifest"], indent=2))


if __name__ == "__main__":
    main()
