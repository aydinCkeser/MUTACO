"""Verify explicitly supplied saved evidence, without producing new evidence.

No GP fitting, simulator calls, pytest execution, scheduler actions, transfer,
automatic raw-reference lookup or output-file writing is exposed here. Run as
python -m reproducibility.verify from the repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import verification as checks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name, help_text in (
        ("b40", "verify a portable B40 package; does not check private originals"),
        ("b100", "verify a portable B100 package; does not check private originals"),
    ):
        command = actions.add_parser(name, help=help_text)
        command.add_argument("--directory", type=Path, required=True)
    for name, help_text in (
        ("checkpoints", "compare saved full packets, including their stored environment identities"),
        ("history", "compare saved history/accounting documents; never replay the algorithm"),
        ("fit-events", "compare saved observed events with saved fit-record expectations"),
    ):
        command = actions.add_parser(name, help=help_text)
        command.add_argument("--first", type=Path, required=True)
        command.add_argument("--second", type=Path, required=True)
        if name == "history":
            command.add_argument("--budget", type=int, choices=(40, 100), required=True)
    command = actions.add_parser("tests", help="verify existing JUnit outcomes and checksum-bound inputs; never run pytest")
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--references", type=Path, required=True)
    command.add_argument("--input-manifest", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True,
                         help="existing directory containing junit.xml and status.json; read only")
    args = parser.parse_args(argv)

    if args.action == "b40":
        result = checks.verify_b40(args.directory)
    elif args.action == "b100":
        result = checks.verify_b100(args.directory)
    elif args.action == "checkpoints":
        result = checks.compare_files(args.first, args.second)
        # Preserve the original comparator's unsorted stdout and failure timing.
        # This interface intentionally omits the original diagnostic-file write.
        print(json.dumps(result))
        if (not result["artifacts_self_consistent"] or not result["same_environment"]
                or not result["scientific_exact_equal"] or any(result["physical_backend_attempts"])):
            raise SystemExit(2)
        return
    elif args.action == "history":
        result = checks.compare_history(checks.load(args.first), checks.load(args.second), args.budget)
        if not (result["baseline_gate_passed"] and result["candidate_gate_passed"]
                and result["scientific_exact_equal"]):
            raise SystemExit(2)
    elif args.action == "fit-events":
        result = checks.compare_fit_events(checks.load(args.first), checks.load(args.second))
        if not (result["events_exact_equal"] and result["event_boundaries_preserved"]):
            raise SystemExit(2)
    else:
        specification = checks.verify_inputs(args.source_root, args.references, args.input_manifest)
        result = checks.tests_gate(
            args.output_root, args.references,
            new_test_prefixes=specification["new_test_prefixes"],
            expected_added=specification["expected_added_tests"],
            expectation_revision=specification.get("test_expectation_revision", checks.HISTORICAL_TEST_EXPECTATION),
        )
        if not result["test_gate_passed"]:
            raise SystemExit(2)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
