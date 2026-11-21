"""Preview or execute one capped graph primary-Q1 worker shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from mutaco import SimulationContext, checked_evaluate
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark.qualification import (
    load_nei_qualification_declaration,
)
from experiments.paper.primary_benchmark.qualification_execution import (
    execute_qualification_worker,
    qualification_worker_preview,
)
from .primary_q1_driver import GraphPrimaryQ1Driver


def _hashed(path: Path) -> dict:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise PaperContractError(f"missing hash-paired Q1 target: {path}")
    if file_hash(path) != sidecar.read_text(encoding="ascii").strip():
        raise PaperContractError(f"Q1 target hash mismatch: {path}")
    return read_json(path)


def validate_target_ledger(declaration, targets_root: Path) -> dict[str, dict]:
    """Load and validate the complete frozen target ledger without simulation."""

    targets = {}
    for path in sorted(targets_root.glob("target_*.json")):
        record = _hashed(path)
        if record.get("declaration_hash") != declaration.target_ledger_hash:
            raise PaperContractError("Q1 target belongs to the wrong frozen ledger")
        targets[str(record["target_record_id"])] = record
    if tuple(targets) != declaration.target_record_identities:
        raise PaperContractError("Q1 target response coverage/order differs from declaration")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--maximum-calls", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--torch-intraop-threads", type=int, required=True)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    declaration = load_nei_qualification_declaration(read_json(args.declaration))
    targets = validate_target_ledger(declaration, args.targets_root)
    if not args.config.is_file():
        raise PaperContractError("Q1 graph configuration is missing")
    preview = qualification_worker_preview(
        declaration,
        rank=args.rank,
        workers=args.workers,
        maximum_calls=args.maximum_calls,
    )
    print(json.dumps({
        **preview,
        "target_ledger_validated": True,
        "target_count_validated": len(targets),
        "configuration_sha256": file_hash(args.config),
        "job_submitted": False,
        "scientific_calls_executed": 0,
    }, indent=2, sort_keys=True))
    if not args.execute:
        return
    if not args.confirm_approved:
        raise SystemExit("Q1 execution requires --confirm-approved")
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).strip()
    if commit != args.expected_commit or dirty:
        raise SystemExit("Q1 execution requires the pinned clean source commit")
    torch.set_num_threads(args.torch_intraop_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    config = read_json(args.config)
    simulator = GraphScientificD4Adapter().build_target_simulator(
        config, device=torch.device("cpu"), dtype=torch.float32
    )
    driver = GraphPrimaryQ1Driver(
        declaration=declaration,
        configuration_path=args.config,
    )

    def run_trajectory(target_id, budget, tier, snapshots, evaluate):
        return driver.run_trajectory(targets[target_id], budget, tier, snapshots, evaluate)

    def evaluate_search(target_id, budget, theta, context):
        del target_id, budget
        result = checked_evaluate(
            simulator,
            torch.tensor(theta, dtype=torch.float32),
            device=torch.device("cpu"),
            dtype=torch.float32,
            context=SimulationContext(
                str(context["family_identity"]),
                "primary_q1_search",
                int(context["evaluation_index"]) - 1,
                0,
                int(context["evaluation_seed"]),
            ),
        )
        return result.values.detach().cpu().tolist()

    execute_qualification_worker(
        declaration=declaration,
        output=args.output,
        rank=args.rank,
        workers=args.workers,
        maximum_calls=args.maximum_calls,
        source_commit=commit,
        run_trajectory=run_trajectory,
        evaluate_search=evaluate_search,
    )


if __name__ == "__main__":
    main()
