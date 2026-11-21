"""Derive portable B100 results from verified private worker artifacts."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from infrastructure.truba import b100_regression as gate
from infrastructure.truba.refactor_reference import canonical, digest, numerical_payload
from infrastructure.truba.shareable_evidence import PRIVATE_PATH, environment_summary, load, sanitize, seal, sha, write


def resource_summary(raw):
    path = raw / "execution/scheduler-accounting.txt"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.read_text().splitlines(), delimiter="|"))
    main = next(row for row in rows if row["JobIDRaw"].isdigit())
    batch = next(row for row in rows if row["JobIDRaw"].endswith(".batch"))
    reported = batch["MaxRSS"]
    peak_kib = int(reported[:-1]) if reported.endswith("K") else None
    requested = main["ReqMem"]
    requested_gib = float(requested[:-1]) if requested.endswith("G") else None
    exceeds = peak_kib is not None and requested_gib is not None and peak_kib > requested_gib * 1024**2
    return {"job_id": int(main["JobIDRaw"]), "jobs_submitted": 1,
            "partition": main["Partition"], "state": main["State"],
            "exit_code": main["ExitCode"], "elapsed_seconds": int(main["ElapsedRaw"]),
            "allocated_cpus": int(main["AllocCPUS"]), "requested_memory": requested,
            "reported_batch_MaxRSS": reported, "reported_peak_rss_kib": peak_kib,
            "reported_peak_rss_gib": peak_kib / 1024**2 if peak_kib is not None else None,
            "reported_peak_exceeds_request": exceeds,
            "future_memory_allocation_requires_review": exceeds,
            "interpretation": "Scheduler figures retained; no inference about memory enforcement. Do not treat 8 GiB as a validated peak bound." if exceeds else "Scheduler figures retained."}


def export(raw, output):
    workers = raw / "workers"
    for line in (workers / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(workers / name) == expected, name
    output.mkdir(parents=True, exist_ok=False)
    inventory = {str(p.relative_to(raw)): sha(p) for p in sorted(raw.rglob("*")) if p.is_file()}
    status = load(workers / "job-status.json")
    derived = {}
    histories, captures = {}, {}
    for label in ("baseline", "refactor"):
        path = workers / (label + "-history.json.gz")
        if path.exists():
            history = load(path)
            histories[label] = history
            if gate.history_gate(history):
                value = canonical(history)
                assert not PRIVATE_PATH.search(json.dumps(value))
                dest = label + "-replay-mathematics.json.gz"
                write(output / dest, value)
                derived[dest] = {"raw_path": str(path.relative_to(raw)), "raw_sha256": sha(path),
                                 "derived_sha256": sha(output / dest), "mathematical_sha256": digest(value),
                                 "origin": "reconstructed original/candidate states; original declared timer/execution exclusions only"}
        path = workers / (label + "-capture.json.gz")
        if path.exists():
            capture = load(path)
            captures[label] = capture
            assert capture["source"]["scientific_commit"] == (gate.BASELINE if label == "baseline" else gate.ACCEPTED)
            reference = capture["first_adaptive"]["capture"]["reference"]
            assert reference["historical_execution_source_commit"] == gate.HISTORICAL_SOURCE
            value = numerical_payload(capture)
            assert digest(value) == capture["scientific_hash"]
            assert not PRIVATE_PATH.search(json.dumps(value))
            dest = label + "-checkpoint-mathematics.json.gz"
            write(output / dest, value)
            derived[dest] = {"raw_path": str(path.relative_to(raw)), "raw_sha256": sha(path),
                             "derived_sha256": sha(output / dest), "mathematical_sha256": digest(value),
                             "origin": "reconstructed first production fit/proposal plus explicitly conditional final-data cold-fit checkpoint"}
    reports = {}
    for name in ("job-status.json", "baseline-tests-gate.json", "refactor-tests-gate.json",
                 "baseline-history-gate.json", "refactor-history-gate.json",
                 "qualified-environment-gate.json", "full-replay-comparison.json", "checkpoint-comparison.json"):
        path = workers / name
        if path.exists():
            reports[name] = sanitize(load(path))
    write(output / "derived-reports.json", {"derived": True, "byte_identical_to_raw": False,
                                           "not_a_raw_acceptance_packet": True, "reports": reports})
    write(output / "raw-evidence-index.json", {"role": "hash-only index of private originals", "raw_files_sha256": inventory})
    history_summary = {}
    for label, h in histories.items():
        history_summary[label] = sanitize({k: h.get(k) for k in (
            "status", "production_status", "budget", "target", "record_pointer",
            "archived_observations_consumed", "ambient_rng_preserved", "physical_simulator_calls",
            "logical_replay_accounting", "frozen_transform_identity", "first_divergence",
            "historical_gp_parameter_snapshots_available")})
    environment = environment_summary(captures["baseline"]["environment"]) if "baseline" in captures else None
    if environment is not None:
        environment["torch_backend_summary"] = [
            line.strip() for line in captures["baseline"]["environment"]["torch_build"].splitlines()
            if any(line.strip().startswith(prefix) for prefix in ("- Intel(R)", "- OpenMP", "- LAPACK", "- CPU capability"))
            and not PRIVATE_PATH.search(line)]
    manifest = {"schema_version": "mutaco-b100-shareable-regression-v1", "derived": True,
                "byte_identical_to_raw": False, "date": "2026-10-26",
                "published_milestone": gate.ACCEPTED, "milestone_push_count": 1,
                "baseline_scientific_source": gate.BASELINE, "accepted_refactor_source": gate.ACCEPTED,
                "historical_execution_source": gate.HISTORICAL_SOURCE,
                "archive_sha256": gate.ARCHIVE_SHA256,
                "case": {"budget": 100, "target": 0, "record": "/records/0", "initial": 16, "adaptive": 84,
                         "independent_of_B40": True},
                "scientific_outcome": sanitize(status), "histories": history_summary,
                "resources": resource_summary(raw),
                "execution_harness_commit": captures.get("baseline", {}).get("source", {}).get("harness_commit"),
                "scientific_source_files_sha256": {k: v["source"]["core_files_sha256"] for k, v in captures.items()},
                "mathematical_derivatives": derived, "numerical_environment_summary": environment,
                "raw_environment_identity_policy": "Original raw fingerprints remain private; sanitized summaries do not replace strict environment acceptance.",
                "original_historical_GP_snapshots_available": False,
                "newly_computed_GP_values": "reconstructed; conditional cold final-data checkpoint is not historical terminal warm state",
                "reference_established": "baseline" in histories and gate.history_gate(histories["baseline"]),
                "baseline_refactor_exact_equivalence_established": status["scientific_gate_passed"],
                "tolerances_changed": False, "candidates_substituted": False,
                "production_algorithm_changed": False, "B40_evidence_modified": False,
                "new_physical_simulator_evaluations": 0, "additional_GitHub_pushes": 0,
                "no_further_cluster_job_authorized": True,
                "artifact_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write(output / "manifest.json", manifest)
    seal(output)
    return manifest


def verify(directory, raw=None):
    manifest = load(directory / "manifest.json")
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(directory / name) == expected, name
    for name, expected in manifest["artifact_sha256"].items():
        assert sha(directory / name) == expected, name
    for name, info in manifest["mathematical_derivatives"].items():
        assert sha(directory / name) == info["derived_sha256"]
        assert digest(load(directory / name)) == info["mathematical_sha256"]
        if raw is not None:
            original = raw / info["raw_path"]
            assert sha(original) == info["raw_sha256"]
            value = load(original)
            value = canonical(value) if "-replay-" in name else numerical_payload(value)
            assert value == load(directory / name)
    if raw is not None:
        assert manifest["resources"] == resource_summary(raw)
        for name, expected in load(directory / "raw-evidence-index.json")["raw_files_sha256"].items():
            assert sha(raw / name) == expected, name
    return {"derived_integrity_verified": True, "private_original_bytes_verified": raw is not None,
            "reference_established": manifest["reference_established"],
            "baseline_refactor_equivalence_established": manifest["baseline_refactor_exact_equivalence_established"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify"))
    parser.add_argument("--raw-source", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "export":
        assert args.raw_source is not None
        export(args.raw_source, args.directory)
    print(json.dumps(verify(args.directory, args.raw_source), sort_keys=True))


if __name__ == "__main__":
    main()
