"""Small explicit inputs; no GP reconstruction or private reference discovery."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


KNOWN_NEI = ("tests.acquisition.test_nei", "test_baseline_factorization_promotes_without_jitter")
SINGULAR = tuple(("tests.acquisition.test_nei",
                  f"test_baseline_factorization_rejects_exactly_singular_covariance[dtype{i}]")
                 for i in range(2))


def fit_inputs() -> tuple[dict, dict]:
    history = {"reconstructed_gp_fits": [{"training_version": 0, "status": "success",
                                          "fit_id": "fit-0", "model_id": "model-0"}]}
    events = [
        {"kind": "transform_fitted", "phase": "initial", "sequential_round": None},
        {"kind": "gp_fit_started", "phase": "fit", "logical_index": None,
         "sequential_round": 0, "message": "GP fit for training version 0 started", "details": {}},
        {"kind": "gp_fit_finished", "phase": "fit", "logical_index": None,
         "sequential_round": 0, "message": "GP fit for training version 0: success",
         "details": {"fit_id": "fit-0", "model_id": "model-0"}},
        {"kind": "recommendation_started", "phase": "recommendation", "sequential_round": None},
    ]
    return history, {"origin": "fixed synthetic event fixture, not historical GP states", "events": events}


def checkpoint_fixture() -> dict:
    # Fixed minimal saved-packet schema; hashes are independent standard-library
    # calculations, rather than a GP fit or a new implementation of the verifier.
    environment = {name: "fixed" for name in (
        "platform", "architecture", "cpu_model", "python", "packages", "torch_build",
        "numpy_build", "scipy_build", "torch_parallel_info", "thread_environment")}
    environment["runtime_threadpools"] = [{"filepath": "fixed-library.so", "num_threads": 20}]
    scientific = {
        "first": {"candidate": [1.0, -0.0]}, "transform": None, "fits": [],
        "logical_accounting": None, "runner_oracle_accounting": {},
        "checkpoints": {"observations": 40},
    }
    encode = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"environment": environment, "environment_signature_schema": "runtime-library-set-v2",
            "environment_signature": encode(environment), "scientific_hash": encode(scientific),
            "physical_backend_attempts": 0,
            "first_adaptive": {"capture": {"reconstructed": scientific["first"], "replay_outcome": {}}},
            "mathematical_checkpoints": scientific["checkpoints"]}


def write_json(path: Path, value) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as stream:
            json.dump(value, stream)
    else:
        path.write_text(json.dumps(value))


def test_gate_inputs(root: Path) -> dict[str, Path]:
    source, references, output = (root / name for name in ("source", "references", "output"))
    for directory in (source / "tests/acquisition", references, output):
        directory.mkdir(parents=True)
    fixture = source / "tests/acquisition/test_nei.py"
    fixture.write_text("# fixed test revision, never executed\n")

    def xml(path, records):
        suite = ET.Element("testsuite")
        for (classname, name), status in records.items():
            case = ET.SubElement(suite, "testcase", classname=classname, name=name)
            if status != "passed":
                ET.SubElement(case, status)
        ET.ElementTree(suite).write(path)

    before = {KNOWN_NEI: "failure", ("fixed", "existing"): "passed", ("fixed", "optional"): "skipped"}
    after = {**before, KNOWN_NEI: "passed", **{case: "passed" for case in SINGULAR}}
    xml(references / "accepted-tests.xml", before)
    xml(output / "junit.xml", after)
    write_json(output / "status.json", {"pytest_exit": 0, "physical_backend_attempts": 0})
    profile = source / "input-manifest.json"
    write_json(profile, {
        "test_expectation_revision": "corrected-nei-fixture-v2",
        "raw_inputs": {"accepted-tests.xml": hashlib.sha256((references / "accepted-tests.xml").read_bytes()).hexdigest()},
        "corrected_fixture": {"path": "tests/acquisition/test_nei.py", "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()},
        "new_test_prefixes": [], "expected_added_tests": 0,
    })
    return {"source": source, "references": references, "output": output, "manifest": profile}


test_gate_inputs.__test__ = False
