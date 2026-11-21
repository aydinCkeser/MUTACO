"""Preview or execute the three-call graph primary assessment rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)
from experiments.graph_inversion.run_primary_integration_smoke_worker import (
    _evaluate_graph_simulator,
    _load_smoke_target,
)
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark import (
    execute_primary_assessment_rescue,
    load_hashed_json,
    load_primary_assessment_rescue_declaration,
    load_primary_population_declaration,
    primary_assessment_rescue_preview,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-declaration", type=Path, required=True)
    parser.add_argument("--primary-declaration", type=Path, required=True)
    parser.add_argument("--decision-freeze", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-calls", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--torch-intraop-threads", type=int, required=True)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    if args.maximum_calls != 3:
        raise PaperContractError("primary assessment rescue requires an exact three-call cap")
    rescue = load_primary_assessment_rescue_declaration(read_json(args.rescue_declaration))
    primary = load_primary_population_declaration(read_json(args.primary_declaration))
    freeze = load_hashed_json(args.decision_freeze)
    freeze_sha = file_hash(args.decision_freeze)
    targets = _load_smoke_target(
        args.targets_root,
        target_id=rescue["target_id"],
        target_ledger_hash=primary.target_ledger_hash,
    )
    if file_hash(args.config) != rescue["configuration_sha256"]:
        raise PaperContractError("primary assessment rescue configuration mismatch")
    preview = primary_assessment_rescue_preview(
        rescue=rescue,
        primary=primary,
        freeze=freeze,
        freeze_sha256=freeze_sha,
    )
    print(json.dumps({
        **preview,
        "maximum_calls_argument": args.maximum_calls,
        "target_ledger_validated": True,
        "configuration_sha256": file_hash(args.config),
    }, indent=2, sort_keys=True))
    if not args.execute:
        return
    if not args.confirm_approved:
        raise SystemExit("primary assessment rescue execution requires --confirm-approved")
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
        raise SystemExit("primary assessment rescue requires pinned clean source")
    torch.set_num_threads(args.torch_intraop_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    simulator = GraphScientificD4Adapter().build_target_simulator(
        read_json(args.config), device=torch.device("cpu"), dtype=torch.float32
    )
    target = targets[rescue["target_id"]]
    execute_primary_assessment_rescue(
        rescue=rescue,
        primary=primary,
        freeze=freeze,
        freeze_sha256=freeze_sha,
        target_response=target["target_response"],
        evaluate_assessment=lambda theta, context: _evaluate_graph_simulator(
            simulator, theta, context
        ),
        output=args.output,
        source_commit=commit,
    )


if __name__ == "__main__":
    main()
