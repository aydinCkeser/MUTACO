"""Portable derived evidence; exact raw acceptance packets remain private."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

from infrastructure.truba.refactor_reference import canonical, compare_files, digest, numerical_payload

REFERENCE = Path("benchmarks/reproducibility/truba_refactor_reference_v1")
REFACTOR = Path("benchmarks/reproducibility/frozen_response_helper_v1")
DIAGNOSIS = Path("configs/environments/mutaco-sbi-macos-arm64-replay-diagnosis.json")
PUBLISHED_BASE = "ccc45090146bdee3f632e7c1084df77daeff9ddb"
ACCEPTED_SOURCE = "d7b5b1267419ca4a95001c2784cfb83b0a020e7b"
PRIVATE_PATH = re.compile(r"(?:git\+)?file://[^\s\"'<>]+|/(?:Users|arf/(?:home|scratch)|private/tmp|tmp)/[^\s\"'<>]+")
PRIVATE_FIELDS = {"python_executable", "login_node", "compute_host", "compute_node",
                  "scientific_import", "binding_import", "snapshot_repository",
                  "remote_root", "log_root", "output_root", "source_root", "runtime_python"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def write(path: Path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        with path.open("xb") as stream:
            with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped:
                zipped.write(data)
    else:
        with path.open("xb") as stream:
            stream.write(data)


def sanitize(value):
    """Remove operational strings only; scientific payloads bypass this function."""
    if isinstance(value, dict):
        return {k: ("<private operational metadata omitted>" if k in PRIVATE_FIELDS else sanitize(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        if value.startswith(("/Users/", "/arf/home/", "/arf/scratch/", "/private/tmp/", "/tmp/")):
            return "<private operational path omitted>"
        value = PRIVATE_PATH.sub("<private operational path omitted>", value)
        value = re.sub(r"\b--account=\S+", "--account=<private>", value)
        return re.sub(r"\b(?:barbun\d+|arf-ui\d+(?:\.ib)?)\b", "<private host>", value)
    return value


def environment_summary(environment):
    """This is provenance, never a replacement for the raw environment fingerprint."""
    return {
        "role": "derived backend summary; strict environment verification requires private raw packets",
        "architecture": environment["architecture"],
        "cpu_model": environment["cpu_model"],
        "python": environment["python"],
        "packages": environment["packages"],
        "device": "cpu",
        "fit_dtype": "torch.float32",
        "posterior_covariance_policy": "float32_fit_float64_covariance",
        "torch_intraop_threads": 20,
        "torch_interop_threads": 1,
        "thread_environment": environment["thread_environment"],
        "libraries": [{k: row[k] for k in ("internal_api", "user_api", "prefix", "version",
                                          "architecture", "threading_layer", "num_threads") if k in row}
                      for row in environment["runtime_threadpools"]],
    }


def raw_inventory(root):
    files = [p for folder in (REFERENCE, REFACTOR) for p in sorted((root / folder).rglob("*")) if p.is_file()]
    files.append(root / DIAGNOSIS)
    return {str(p.relative_to(root)): sha(p) for p in files}


def check_raw(root):
    """Retain all original checksum and exact-environment/science checks."""
    for folder in (REFERENCE, REFACTOR):
        for line in (root / folder / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert sha(root / folder / name) == expected, name
        manifest = load(root / folder / "manifest.json")
        for name, expected in manifest["artifact_sha256"].items():
            assert sha(root / folder / name) == expected, name
    baseline = root / REFERENCE / "baseline-b40.json.gz"
    for other in (root / REFERENCE / "repeat-b40.json.gz",
                  root / REFACTOR / "workers/candidate-b40.json.gz"):
        result = compare_files(baseline, other)
        assert result["artifacts_self_consistent"] and result["same_environment"]
        assert result["scientific_exact_equal"] and not any(result["physical_backend_attempts"])
    return raw_inventory(root)


def reports(root):
    baseline = load(root / REFERENCE / "baseline-b40.json.gz")
    candidate = load(root / REFACTOR / "workers/candidate-b40.json.gz")
    assert numerical_payload(baseline) == numerical_payload(candidate)
    assert digest(numerical_payload(baseline)) == baseline["scientific_hash"] == candidate["scientific_hash"]
    history = [load(root / p) for p in
               (REFERENCE / "historical-b40.json.gz", REFACTOR / "workers/candidate-historical-b40.json.gz")]
    assert canonical(history[0]) == canonical(history[1])
    assert all(h["status"] == "passed" and h["archived_observations_consumed"] == 40
               and h["ambient_rng_preserved"] and h["physical_simulator_calls"] == 0 for h in history)
    base_report = load(root / REFERENCE / "manifest.json")
    refactor_report = load(root / REFACTOR / "manifest.json")
    base_report["environment"] = environment_summary(baseline["environment"])
    refactor_report["truba"]["environment"] = environment_summary(candidate["environment"])
    diagnosis = load(root / DIAGNOSIS)
    diagnosis.pop("verbose_working_artifacts_not_permanent_archival_storage", None)
    return {
        "b40-scientific-payload.json.gz": (numerical_payload(baseline), [
            str(REFERENCE / "baseline-b40.json.gz"), str(REFERENCE / "repeat-b40.json.gz"),
            str(REFACTOR / "workers/candidate-b40.json.gz")], "exact mathematical payload"),
        "b40-replay-mathematics.json.gz": (canonical(history[0]), [
            str(REFERENCE / "historical-b40.json.gz"), str(REFACTOR / "workers/candidate-historical-b40.json.gz")],
            "exact mathematical replay fields; original predeclared timing/execution exclusions only"),
        "baseline-summary.json": (sanitize(base_report), [str(REFERENCE / "manifest.json")], "sanitized derived report"),
        "refactor-summary.json": (sanitize(refactor_report), [str(REFACTOR / "manifest.json")], "sanitized derived report"),
        "macos-diagnostic-summary.json": (sanitize(diagnosis), [str(DIAGNOSIS)], "sanitized derived report"),
    }


def derive(root, output):
    inventory = check_raw(root)
    output.mkdir(parents=True, exist_ok=False)
    mapping = {}
    for name, (value, origins, role) in reports(root).items():
        if "payload" not in name and "mathematics" not in name:
            value = {"derived": True, "byte_identical_to_raw": False,
                     "not_a_raw_acceptance_packet": True, "report": value}
        encoded = json.dumps(value, sort_keys=True)
        assert not PRIVATE_PATH.search(encoded), "operational path in derived artifact"
        write(output / name, value)
        mapping[name] = {"derived_sha256": sha(output / name), "role": role,
                         "raw_sources": [{"path": p, "sha256": inventory[p]} for p in origins]}
    write(output / "raw-evidence-index.json", {
        "role": "hash-only inventory; raw contents retained privately",
        "original_commit": ACCEPTED_SOURCE, "raw_files_sha256": inventory})
    manifest = {
        "schema_version": "mutaco-shareable-evidence-v1", "derived": True,
        "raw_evidence_modified": False, "byte_identical_to_raw": False,
        "remote_checkpoint": PUBLISHED_BASE, "accepted_source_commit": ACCEPTED_SOURCE,
        "baseline_source_commit": "aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3",
        "capture_harness_commit": "27f654a8e590d3aadb89a2791c91f1e1491a1aad",
        "validated_source_snapshot": "31e161d0112ffd920e4f105ed498be39b9b4ec68",
        "scientific_payload_sha256": digest(load(output / "b40-scientific-payload.json.gz")),
        "replay_mathematics_sha256": digest(load(output / "b40-replay-mathematics.json.gz")),
        "derived_files": mapping,
        "raw_index_sha256": sha(output / "raw-evidence-index.json"),
        "verification_scope": "Portable hashes/math only. Exact environment acceptance still uses unmodified raw comparator and private original packets.",
        "private_inputs_required": "An independently verified restore of the original private branch backup; never substitute these summaries for raw packets.",
        "baseline_tests": {"passed": 169, "skipped": 8, "known_nei_failure": 1},
        "refactor_tests": {"passed": 175, "skipped": 8, "known_nei_failure": 1, "new_failures": 0},
        "b40": {"target": 16, "record": "/records/48", "observations": 40, "fits": 25,
                "proposals": 24, "recommendations": 1, "historical_replay": "passed",
                "baseline_refactor_scientific_exact": True, "ambient_rng_preserved": True,
                "physical_simulator_calls": 0},
        "b100_reference_established": False,
        "no_new_simulations_or_cluster_jobs_in_hygiene_task": True,
    }
    write(output / "manifest.json", manifest)
    seal(output)
    return verify(output, root)


def seal(output):
    with (output / "SHA256SUMS").open("w") as stream:
        for p in sorted(output.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                stream.write(sha(p) + "  " + str(p.relative_to(output)) + "\n")


def verify(output, root=None):
    manifest = load(output / "manifest.json")
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(output / name) == expected, name
    assert sha(output / "raw-evidence-index.json") == manifest["raw_index_sha256"]
    for name, details in manifest["derived_files"].items():
        assert sha(output / name) == details["derived_sha256"], name
    assert digest(load(output / "b40-scientific-payload.json.gz")) == manifest["scientific_payload_sha256"]
    assert digest(load(output / "b40-replay-mathematics.json.gz")) == manifest["replay_mathematics_sha256"]
    if root is not None:
        assert check_raw(root) == load(output / "raw-evidence-index.json")["raw_files_sha256"]
        for name, (value, _, role) in reports(root).items():
            actual = load(output / name)
            assert (actual if role.startswith("exact") else actual["report"]) == value, name
    return {"derived_integrity_verified": True, "raw_original_checks_verified": root is not None,
            "scientific_payload_sha256": manifest["scientific_payload_sha256"],
            "replay_mathematics_sha256": manifest["replay_mathematics_sha256"],
            "new_numerical_calculations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("derive", "verify"))
    parser.add_argument("--raw-source", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "derive":
        assert args.raw_source is not None, "derivation requires original private evidence"
        result = derive(args.raw_source, args.directory)
    else:
        result = verify(args.directory, args.raw_source)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
