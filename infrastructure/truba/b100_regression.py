"""Metadata gates for one archived B100 baseline/refactor comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from infrastructure.truba.refactor_reference import canonical, digest, environment_signature
from infrastructure.truba.shareable_evidence import load, write

BASELINE = "aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3"
ACCEPTED = "242fac08c1c3dc0041651a731c20ed228969ed6a"
ARCHIVE_SHA256 = "e30befedd72b85bd0eda8fe1d7e33ed6ef10acd0c880b1bd6aca60865e4a49c7"
HISTORICAL_SOURCE = "649235bcb7e8773f5f4b0425ad111314d7d3b1f9"
KNOWN_NEI = ("tests.acquisition.test_nei", "test_baseline_factorization_promotes_without_jitter")
ACCOUNTING = {
    "committed_fit_events": 85, "committed_observations": 100,
    "committed_proposal_events": 84, "committed_recommendation_events": 1,
    "initial_observations": 16, "planned_observations": 100,
    "sequential_observations": 84, "training_data_versions": 85,
}


def history_gate(report):
    return (report["status"] == "passed" and report["budget"] == 100
            and report["target"] == 0 and report["record_pointer"] == "/records/0"
            and report["archived_observations_consumed"] == 100
            and report["ambient_rng_preserved"] and report["physical_simulator_calls"] == 0
            and report["logical_replay_accounting"] == ACCOUNTING)


def first_difference(a, b, path=""):
    if type(a) is not type(b):
        return {"path": path, "kind": "type", "expected_type": type(a).__name__, "actual_type": type(b).__name__}
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return {"path": path, "kind": "keys"}
        for key in a:
            result = first_difference(a[key], b[key], path + "/" + str(key))
            if result is not None:
                return result
    elif isinstance(a, list):
        if len(a) != len(b):
            return {"path": path, "kind": "length", "expected": len(a), "actual": len(b)}
        for index, (x, y) in enumerate(zip(a, b)):
            result = first_difference(x, y, path + "/" + str(index))
            if result is not None:
                return result
    elif a != b or (isinstance(a, float) and a.hex() != b.hex()):
        return {"path": path, "kind": "value", "expected": a, "actual": b}
    return None


def compare_histories(a, b):
    x, y = canonical(a), canonical(b)
    return {"baseline_gate_passed": history_gate(a), "refactor_gate_passed": history_gate(b),
            "scientific_exact_equal": digest(x) == digest(y),
            "baseline_sha256": digest(x), "refactor_sha256": digest(y),
            "first_difference": first_difference(x, y),
            "excluded": "original predeclared timers and execution provenance only", "tolerances": "none"}


def test_outcomes(path):
    return {(c.attrib["classname"], c.attrib["name"]):
            next((tag for tag in ("failure", "error", "skipped") if c.find(tag) is not None), "passed")
            for c in ET.parse(path).iter("testcase")}


def test_gate(directory, baseline_directory=None):
    cases = test_outcomes(directory / "junit.xml")
    status = load(directory / "status.json")
    failures = {key for key, value in cases.items() if value in ("failure", "error")}
    passed = failures == {KNOWN_NEI} and status["pytest_exit"] == 1 and status["physical_backend_attempts"] == 0
    changed, added = [], {}
    if baseline_directory is not None:
        before = test_outcomes(baseline_directory / "junit.xml")
        changed = [list(k) for k, value in before.items() if cases.get(k) != value]
        added = {k: value for k, value in cases.items() if k not in before}
        passed = (passed and not changed and len(added) == 6
                  and all(k[0] == "tests.orchestration.test_runner"
                          and k[1].startswith("test_response_preparation_failure_preserves_committed_state[")
                          and v == "passed" for k, v in added.items()))
    return {"test_gate_passed": passed, "pytest_exit": status["pytest_exit"],
            "counts": {k: sum(v == k for v in cases.values()) for k in ("passed", "failure", "error", "skipped")},
            "physical_backend_attempts": status["physical_backend_attempts"],
            "changed_baseline_cases": changed, "added_cases": [{"case": list(k), "outcome": v} for k, v in sorted(added.items())],
            "known_NEI_failure_present": KNOWN_NEI in failures}


def finish(root, stage, execution_exit, scientific_exit):
    def optional(name):
        p = root / name
        return load(p) if p.exists() else {}
    baseline = optional("baseline-history.json.gz")
    candidate = optional("refactor-history.json.gz")
    comparison = optional("checkpoint-comparison.json")
    full = optional("full-replay-comparison.json")
    result = {"last_stage": stage, "execution_exit": execution_exit, "scientific_exit": scientific_exit,
              "historical_baseline_status": baseline.get("status", "not_executed"),
              "refactor_historical_status": candidate.get("status", "not_executed"),
              "baseline_vs_refactor": "passed" if scientific_exit == 0 else "not_established",
              "scientific_gate_passed": scientific_exit == 0,
              "historical_reference_failure_is_not_refactor_failure": True,
              "baseline_observations_consumed": baseline.get("archived_observations_consumed"),
              "refactor_observations_consumed": candidate.get("archived_observations_consumed"),
              "first_historical_divergence": baseline.get("first_divergence"),
              "refactor_first_divergence": candidate.get("first_divergence"),
              "checkpoint_scientific_exact": comparison.get("scientific_exact_equal"),
              "full_replay_scientific_exact": full.get("scientific_exact_equal"),
              "physical_simulator_calls_authorized": 0,
              "recorded_physical_backend_attempts": [r.get("physical_simulator_calls") for r in (baseline, candidate) if r]}
    write(root / "job-status.json", result)
    with (root / "SHA256SUMS").open("x") as stream:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS" and "temp" not in p.relative_to(root).parts:
                stream.write(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(root)) + "\n")
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "history", "tests", "environment", "compare-history", "finish"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--execution-exit", type=int)
    parser.add_argument("--scientific-exit", type=int)
    args = parser.parse_args()
    if args.action == "validate":
        assert not args.output_root.exists()
        manifest = load(args.source_root / "results/paper/scalar_primary_trajectory_archives_v3r2/b100-v3r2/scalar-primary-b100-v3r2.manifest.json")
        assert manifest["archive_sha256"] == ARCHIVE_SHA256 and not manifest["raw_posterior_artifacts_included"]
        campaign = load(args.source_root / "results/paper/graph_scalar_primary_b100_v1/manifest.json")
        assert campaign["scientific_source"]["commit"] == HISTORICAL_SOURCE
        print(json.dumps({"nodes": 1, "tasks": 1, "cpus": 20, "memory_gib": 8, "maximum_minutes": 20, "scientific_calls": 0}))
        print(json.dumps({"scientific_calls_executed": 0, "job_submitted": False, "baseline": BASELINE, "accepted": ACCEPTED,
                          "physical_simulator_calls_authorized": 0, "budget": 100, "target": 0}))
        return
    if args.action == "finish":
        finish(args.output_root, args.stage, args.execution_exit, args.scientific_exit)
        return
    if args.action == "history":
        document = load(args.first)
        result = {"history_gate_passed": history_gate(document), "status": document["status"],
                  "observations": document["archived_observations_consumed"], "first_divergence": document.get("first_divergence")}
        passed = result["history_gate_passed"]
    elif args.action == "tests":
        result = test_gate(args.first, args.second)
        passed = result["test_gate_passed"]
    elif args.action == "environment":
        a, b = load(args.first), load(args.second)
        passed = environment_signature(a["environment"]) == environment_signature(b["environment"])
        result = {"same_qualified_B40_environment": passed, "normalization": "predeclared unordered runtime-library records only"}
    else:
        result = compare_histories(load(args.first), load(args.second))
        passed = result["baseline_gate_passed"] and result["refactor_gate_passed"] and result["scientific_exact_equal"]
    if args.output:
        write(args.output, result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
