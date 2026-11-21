"""Fail-closed metadata gates for controlled narrow readability extractions.

Numerical work reuses refactor_reference.py; optional observation writes a separate event sidecar. Private saved baseline
packets are checksum-bound inputs; no original GP fit is rerun by this workflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from infrastructure.truba.b100_regression import KNOWN_NEI, first_difference, test_outcomes
from infrastructure.truba.refactor_reference import canonical, digest
from infrastructure.truba.shareable_evidence import load, sha, write

NEW_TESTS = (
    "test_pre_run_validation_success_order_and_seeds[",
    "test_pre_run_validation_first_failure[",
    "test_pre_run_validation_exception_boundary[",
)


HISTORICAL_TEST_EXPECTATION = "historical-nei-failure-v1"
CORRECTED_TEST_EXPECTATION = "corrected-nei-fixture-v2"
DEFAULT_INPUT_MANIFEST = "infrastructure/truba/corrected_test_reference_inputs_v2.json"
CORRECTED_NEW_TESTS = (
    "test_fit_result_real_success_warm_history_and_events[",
    "test_fit_result_controlled_failures_keep_partial_state[",
    "test_fit_result_controlled_recovery_counts_only_executed_attempts[",
    "test_fit_result_record_allocation_precedes_failed_status_handling[",
)
SINGULAR_NEI_CASES = tuple(
    ("tests.acquisition.test_nei",
     f"test_baseline_factorization_rejects_exactly_singular_covariance[dtype{i}]")
    for i in range(2)
)


def verify_inputs(source, references, input_manifest=DEFAULT_INPUT_MANIFEST):
    manifest = load(source / input_manifest)
    for name, expected in manifest["raw_inputs"].items():
        if sha(references / name) != expected:
            raise ValueError(f"private reference checksum differs: {name}")
    if manifest.get("test_expectation_revision") == CORRECTED_TEST_EXPECTATION:
        fixture = manifest["corrected_fixture"]
        if sha(source / fixture["path"]) != fixture["sha256"]:
            raise ValueError("corrected NEI test revision checksum differs")
    return manifest


def history_gate(report, budget):
    rounds = budget - 16
    accounting = {
        "committed_fit_events": rounds + 1, "committed_observations": budget,
        "committed_proposal_events": rounds, "committed_recommendation_events": 1,
        "initial_observations": 16, "planned_observations": budget,
        "sequential_observations": rounds, "training_data_versions": rounds + 1,
    }
    return (report["status"] == "passed" and report["production_status"] == "completed"
            and report["budget"] == budget and report["target"] == (16 if budget == 40 else 0)
            and report["record_pointer"] == ("/records/48" if budget == 40 else "/records/0")
            and report["archived_observations_consumed"] == budget
            and report["ambient_rng_preserved"] and report["physical_simulator_calls"] == 0
            and report["torch_intraop_threads"] == 20 and report["torch_interop_threads"] == 1
            and report["logical_replay_accounting"] == accounting)


def compare_history(first, second, budget):
    x, y = canonical(first), canonical(second)
    return {"baseline_gate_passed": history_gate(first, budget),
            "candidate_gate_passed": history_gate(second, budget),
            "scientific_exact_equal": digest(x) == digest(y),
            "baseline_sha256": digest(x), "candidate_sha256": digest(y),
            "first_difference": first_difference(x, y), "tolerances": "none",
            "excluded": "unchanged predeclared timers and execution provenance only"}


def tests_gate(directory, references, *, new_test_prefixes=CORRECTED_NEW_TESTS,
               expected_added=18, expectation_revision=CORRECTED_TEST_EXPECTATION):
    before = test_outcomes(references / "accepted-tests.xml")
    after = test_outcomes(directory / "junit.xml")
    status = load(directory / "status.json")
    failures = {k for k, v in after.items() if v in ("failure", "error")}
    if expectation_revision == CORRECTED_TEST_EXPECTATION:
        valid_reference = (before.get(KNOWN_NEI) in {"failure", "passed"}
                           and all(before.get(k) in {None, "passed"}
                                   for k in SINGULAR_NEI_CASES))
        expected = {**before, KNOWN_NEI: "passed",
                    **{k: "passed" for k in SINGULAR_NEI_CASES}}
        changed = [list(k) for k, v in expected.items() if after.get(k) != v]
        added = {k: v for k, v in after.items() if k not in expected}
        passed = (valid_reference and not failures and status["pytest_exit"] == 0
                  and status["physical_backend_attempts"] == 0 and not changed
                  and len(added) == expected_added and all(
                      k[0] == "tests.orchestration.test_runner"
                      and k[1].startswith(tuple(new_test_prefixes)) and v == "passed"
                      for k, v in added.items()))
        return {"test_gate_passed": passed, "pytest_exit": status["pytest_exit"],
                "counts": {k: sum(v == k for v in after.values())
                           for k in ("passed", "failure", "error", "skipped")},
                "changed_accepted_cases": changed, "added_passed_cases": len(added),
                "known_NEI_failure_present": KNOWN_NEI in failures,
                "physical_backend_attempts": status["physical_backend_attempts"],
                "test_expectation_revision": expectation_revision,
                "declared_NEI_correction": before.get(KNOWN_NEI) == "failure",
                "required_singular_cases": [list(k) for k in SINGULAR_NEI_CASES]}
    if expectation_revision != HISTORICAL_TEST_EXPECTATION:
        raise ValueError(f"unknown test expectation revision: {expectation_revision!r}")
    changed = [list(k) for k, v in before.items() if after.get(k) != v]
    added = {k: v for k, v in after.items() if k not in before}
    passed = (failures == {KNOWN_NEI} and status["pytest_exit"] == 1
              and status["physical_backend_attempts"] == 0 and not changed
              and len(added) == expected_added and all(
                  k[0] == "tests.orchestration.test_runner"
                  and k[1].startswith(tuple(new_test_prefixes)) and v == "passed" for k, v in added.items()))
    return {"test_gate_passed": passed, "pytest_exit": status["pytest_exit"],
            "counts": {k: sum(v == k for v in after.values())
                       for k in ("passed", "failure", "error", "skipped")},
            "changed_accepted_cases": changed, "added_passed_cases": len(added),
            "known_NEI_failure_present": KNOWN_NEI in failures,
            "physical_backend_attempts": status["physical_backend_attempts"]}


def finish(root, stage, execution_exit, scientific_exit):
    report = {"last_stage": stage, "execution_exit": execution_exit,
              "scientific_exit": scientific_exit, "scientific_gate_passed": scientific_exit == 0,
              "original_baseline_reexecuted": False, "physical_simulator_calls_authorized": 0}
    write(root / "job-status.json", report)
    with (root / "SHA256SUMS").open("x") as stream:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS" and "temp" not in p.relative_to(root).parts:
                stream.write(sha(p) + "  " + str(p.relative_to(root)) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "inputs", "history", "tests", "finish", "fit-events"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--input-manifest", type=Path,
                        help="checksum-bound input/test specification for a later extraction")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--budget", type=int, choices=(40, 100))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--execution-exit", type=int)
    parser.add_argument("--scientific-exit", type=int)
    args = parser.parse_args()
    if args.action == "validate":
        assert not args.output_root.exists()
        for budget in (40, 100):
            parent = args.source_root / f"results/paper/scalar_primary_trajectory_archives_v3r2/b{budget}-v3r2"
            name = f"scalar-primary-b{budget}-v3r2"
            manifest = load(parent / (name + ".manifest.json"))
            assert sha(parent / (name + ".tar.gz")) == manifest["archive_sha256"]
        result = {"nodes": 1, "tasks": 1, "cpus": 20, "memory_gib": 20,
                  "maximum_minutes": 20, "scientific_calls": 0}
        import json
        print(json.dumps(result, sort_keys=True))
        result = {"scientific_calls_executed": 0, "job_submitted": False,
                  "original_baseline_reexecuted": False}
    elif args.action == "inputs":
        result = verify_inputs(args.source_root, args.references, args.input_manifest) if args.input_manifest else verify_inputs(args.source_root, args.references)
    elif args.action == "history":
        result = compare_history(load(args.first), load(args.second), args.budget)
        if args.output:
            write(args.output, result)
        if not (result["baseline_gate_passed"] and result["candidate_gate_passed"]
                and result["scientific_exact_equal"]):
            raise SystemExit(2)
    elif args.action == "fit-events":
        from infrastructure.truba.fit_event_evidence import compare_fit_events
        result = compare_fit_events(load(args.first), load(args.second))
        if args.output:
            write(args.output, result)
        if not (result["events_exact_equal"] and result["event_boundaries_preserved"]):
            raise SystemExit(2)
    elif args.action == "tests":
        source = args.source_root or Path(__file__).resolve().parents[2]
        specification = verify_inputs(
            source, args.references, args.input_manifest or DEFAULT_INPUT_MANIFEST)
        result = tests_gate(
            args.output_root, args.references,
            new_test_prefixes=specification["new_test_prefixes"],
            expected_added=specification["expected_added_tests"],
            expectation_revision=specification.get(
                "test_expectation_revision", HISTORICAL_TEST_EXPECTATION))
        if args.output:
            write(args.output, result)
        if not result["test_gate_passed"]:
            raise SystemExit(2)
    else:
        result = finish(args.output_root, args.stage, args.execution_exit, args.scientific_exit)
    import json
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
