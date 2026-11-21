"""Unified, zero-call Paper Lab foundation command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .consolidation import build_consolidated_bundle, verify_consolidated_bundle
from .ingestion import validate_ingestion_bindings
from .promotion import verify_promotion
from .reporting import build_report, validate_report_source
from .schemas import load_and_validate
from .transport import export_compact_population, fetch_compact_export


def _emit(document: dict[str, object]) -> None:
    print(json.dumps(document, indent=2, sort_keys=True))


def _reserved(action: str) -> None:
    raise SystemExit(
        f"{action} is deliberately unavailable in the Paper Lab foundation; "
        "a separately approved implementation is required"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    study = stages.add_parser("study")
    study_actions = study.add_subparsers(dest="action", required=True)
    study_validate = study_actions.add_parser("validate")
    study_validate.add_argument("spec", type=Path)

    run = stages.add_parser("run")
    run_actions = run.add_subparsers(dest="action", required=True)
    run_plan = run_actions.add_parser("plan")
    run_plan.add_argument("study", type=Path)
    run_plan.add_argument("execution", type=Path)
    run_actions.add_parser("submit")

    ingest = stages.add_parser("ingest")
    ingest_actions = ingest.add_subparsers(dest="action", required=True)
    ingest_validate = ingest_actions.add_parser("validate")
    ingest_validate.add_argument("spec", type=Path)
    ingest_validate.add_argument("--source", action="append", default=[])
    ingest_build = ingest_actions.add_parser("build")
    ingest_build.add_argument("spec", type=Path)
    ingest_build.add_argument("--source", action="append", default=[])
    ingest_build.add_argument("--repository-root", type=Path, default=Path.cwd())
    ingest_build.add_argument("--output", type=Path, required=True)
    ingest_build.add_argument("--confirm-authorized", action="store_true")
    ingest_export = ingest_actions.add_parser("export")
    ingest_export.add_argument("source", type=Path)
    ingest_export.add_argument("--output-prefix", type=Path, required=True)
    ingest_export.add_argument("--confirm-authorized", action="store_true")
    ingest_fetch = ingest_actions.add_parser("fetch")
    ingest_fetch.add_argument("source_prefix")
    ingest_fetch.add_argument("--destination", type=Path, required=True)
    ingest_fetch.add_argument("--confirm-authorized", action="store_true")

    report = stages.add_parser("report")
    report_actions = report.add_subparsers(dest="action", required=True)
    report_validate = report_actions.add_parser("validate")
    report_validate.add_argument("spec", type=Path)
    report_validate.add_argument("--repository-root", type=Path, default=Path.cwd())
    report_build = report_actions.add_parser("build")
    report_build.add_argument("spec", type=Path)
    report_build.add_argument("--repository-root", type=Path, default=Path.cwd())
    report_build.add_argument("--confirm-authorized", action="store_true")
    report_build.add_argument("--check", action="store_true")

    figure = stages.add_parser("figure")
    figure_actions = figure.add_subparsers(dest="action", required=True)
    figure_validate = figure_actions.add_parser("validate")
    figure_validate.add_argument("spec", type=Path)
    figure_actions.add_parser("build")

    explore = stages.add_parser("explore")
    explore_actions = explore.add_subparsers(dest="action", required=True)
    explore_validate = explore_actions.add_parser("validate")
    explore_validate.add_argument("spec", type=Path)

    verify = stages.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    promote = stages.add_parser("promote")
    promote_actions = promote.add_subparsers(dest="action", required=True)
    promote_verify = promote_actions.add_parser("verify")
    promote_verify.add_argument("spec", type=Path)
    promote_verify.add_argument("--repository-root", type=Path, default=Path.cwd())
    promote_verify.add_argument("--state", type=Path)
    promote_verify.add_argument("--registry", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "study" and args.action == "validate":
        document = load_and_validate(args.spec)
        _emit({"status": "validated", "identity": document["study_identity"], "scientific_calls": 0})
    elif args.stage == "run" and args.action == "plan":
        study = load_and_validate(args.study)
        execution = load_and_validate(args.execution)
        if execution["study_identity"] != study["study_identity"]:
            raise SystemExit("execution does not reference the validated study identity")
        _emit({
            "status": "planned_unexecuted",
            "study_identity": study["study_identity"],
            "execution_identity": execution["execution_identity"],
            "scientific_calls": 0,
            "job_submitted": False,
        })
    elif args.stage == "ingest" and args.action == "validate":
        _emit(validate_ingestion_bindings(load_and_validate(args.spec), args.source))
    elif args.stage == "ingest" and args.action == "build":
        _emit(build_consolidated_bundle(
            load_and_validate(args.spec), args.source,
            repository_root=args.repository_root, output=args.output,
            confirm_authorized=args.confirm_authorized,
        ))
    elif args.stage == "ingest" and args.action == "export":
        _emit(export_compact_population(
            args.source, args.output_prefix,
            confirm_authorized=args.confirm_authorized,
        ))
    elif args.stage == "ingest" and args.action == "fetch":
        _emit(fetch_compact_export(
            args.source_prefix, args.destination,
            confirm_authorized=args.confirm_authorized,
        ))
    elif args.stage == "report" and args.action == "validate":
        _emit(validate_report_source(load_and_validate(args.spec), args.repository_root))
    elif args.stage == "report" and args.action == "build":
        _emit(build_report(
            load_and_validate(args.spec), args.repository_root,
            confirm_authorized=args.confirm_authorized, check=args.check,
        ))
    elif args.stage == "figure" and args.action == "validate":
        document = load_and_validate(args.spec)
        _emit({
            "status": "discussion_pending",
            "figure_identity": document["figure_identity"],
            "scientific_calls": 0,
            "figures_written": 0,
        })
    elif args.stage == "explore" and args.action == "validate":
        document = load_and_validate(args.spec)
        _emit({
            "status": "validated_non_paper",
            "exploration_identity": document["exploration_identity"],
            "scientific_calls": 0,
        })
    elif args.stage == "verify":
        _emit(verify_consolidated_bundle(args.bundle))
    elif args.stage == "promote" and args.action == "verify":
        state_path = args.state or args.repository_root / "results/paper/promotion_state.toml"
        registry_path = args.registry or args.repository_root / "results/experiment_registry.toml"
        _emit(verify_promotion(
            load_and_validate(args.spec), args.repository_root,
            state_path=state_path, registry_path=registry_path,
        ))
    else:
        _reserved(f"{args.stage} {getattr(args, 'action', '')}".strip())


if __name__ == "__main__":
    main()
