"""Preview or execute one repaired MUTACO--SBI assessment worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from mutaco import SimulationContext, checked_evaluate
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter
from experiments.graph_inversion.paper_v3_callbacks import _assessment_randomness_evidence
from experiments.paper.common.protocol_v3 import AssessmentEvaluationV3
from experiments.paper.historical_workflows.repaired_sbi_assessment_execution import (
    execute_worker,
    worker_preview,
)
from experiments.paper.historical_workflows.repaired_sbi_assessment_preparation import (
    OUTPUT,
    TARGET_CONFIG_SHA256,
    assessment_declaration,
    preparation_inventory,
)
from experiments.paper.common.validation import PaperContractError


def _tasks(path: Path) -> list[dict]:
    sidecar = path.parent / f"{path.name}.sha256"
    if not path.is_file() or not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != file_hash(path):
        raise PaperContractError("assessment task hash pair is invalid")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _targets(root: Path) -> dict[str, dict]:
    result = {}
    for position in range(120):
        path = root / f"target_{position:03d}.json"
        sidecar = path.with_suffix(".sha256")
        if not path.is_file() or not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != file_hash(path):
            raise PaperContractError("assessment target hash pair is invalid")
        target = read_json(path)
        if target.get("target_index") != position:
            raise PaperContractError("assessment target ordering mismatch")
        result[str(target["target_record_id"])] = target
    if len(result) != 120:
        raise PaperContractError("assessment target coverage is incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--maximum-calls", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--torch-intraop-threads", type=int, required=True)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--validate-all-shards", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    declaration = assessment_declaration()
    frozen_tasks, _ = preparation_inventory(root, declaration)
    tasks = _tasks(args.tasks)
    if tasks != frozen_tasks:
        raise PaperContractError("assessment task file differs from canonical preparation")
    targets = _targets(args.targets_root)
    if file_hash(args.config) != TARGET_CONFIG_SHA256:
        raise PaperContractError("assessment simulator configuration hash mismatch")
    preview = worker_preview(
        tasks, rank=args.rank, workers=args.workers, maximum_calls=args.maximum_calls
    )
    if args.validate_all_shards:
        for rank in range(args.workers):
            worker_preview(tasks, rank=rank, workers=args.workers, maximum_calls=args.maximum_calls)
    print(json.dumps({
        **preview,
        "all_shards_validated": args.validate_all_shards,
        "target_count_validated": len(targets),
        "configuration_sha256": file_hash(args.config),
    }, indent=2, sort_keys=True))
    if not args.execute:
        return
    if not args.confirm_approved:
        raise SystemExit("assessment execution requires --confirm-approved")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    ).strip()
    if commit != args.expected_commit or dirty:
        raise SystemExit("assessment execution requires pinned clean source")
    torch.set_num_threads(args.torch_intraop_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    simulator = GraphScientificD4Adapter().build_target_simulator(
        read_json(args.config), device=torch.device("cpu"), dtype=torch.float32
    )

    def evaluate(theta, context, target):
        result = checked_evaluate(
            simulator,
            torch.tensor(theta, dtype=torch.float32),
            device=torch.device("cpu"),
            dtype=torch.float32,
            context=SimulationContext(
                str(context["family_identity"]),
                str(context["purpose"]),
                int(context["logical_index"]),
                int(context["replicate"]),
                int(context["evaluation_seed"]),
            ),
        )
        evidence = _assessment_randomness_evidence(context, result.metadata)
        return AssessmentEvaluationV3(
            tuple(float(value) for value in result.values.detach().cpu().tolist()),
            evidence["compact"], evidence["raw"],
        )

    manifest = read_json(root / OUTPUT / "manifest.json")
    execute_worker(
        tasks=tasks,
        targets=targets,
        output=args.output,
        rank=args.rank,
        workers=args.workers,
        maximum_calls=args.maximum_calls,
        source_commit=commit,
        declaration_identity=declaration["declaration_identity"],
        package_identity=manifest["package_identity"],
        evaluate=evaluate,
    )


if __name__ == "__main__":
    main()
