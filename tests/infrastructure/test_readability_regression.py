"""Metadata rejection gates; no numerical reconstruction or simulator calls."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from infrastructure.truba import readability_regression as gate
from infrastructure.truba.b100_regression import KNOWN_NEI
from infrastructure.truba.shareable_evidence import sha


def report(budget):
    rounds = budget - 16
    return {
        "status": "passed", "production_status": "completed", "budget": budget,
        "target": 16 if budget == 40 else 0,
        "record_pointer": "/records/48" if budget == 40 else "/records/0",
        "archived_observations_consumed": budget, "ambient_rng_preserved": True,
        "physical_simulator_calls": 0, "torch_intraop_threads": 20,
        "torch_interop_threads": 1,
        "logical_replay_accounting": {
            "committed_fit_events": rounds + 1, "committed_observations": budget,
            "committed_proposal_events": rounds, "committed_recommendation_events": 1,
            "initial_observations": 16, "planned_observations": budget,
            "sequential_observations": rounds, "training_data_versions": rounds + 1,
        },
    }


@pytest.mark.parametrize("budget", [40, 100])
def test_history_gate_preserves_exact_science_and_ignores_only_declared_timing(budget):
    a = report(budget)
    b = deepcopy(a)
    a["timings"], b["timings"] = {"total_seconds": 1.0}, {"total_seconds": 2.0}
    assert gate.compare_history(a, b, budget)["scientific_exact_equal"]
    b["candidate"] = [0.0]
    a["candidate"] = [-0.0]
    result = gate.compare_history(a, b, budget)
    assert not result["scientific_exact_equal"]
    assert result["first_difference"]["path"] == "/candidate/0"
    b["ambient_rng_preserved"] = False
    assert not gate.compare_history(a, b, budget)["candidate_gate_passed"]


@pytest.mark.parametrize("field,value", [("physical_simulator_calls", 1),
                                          ("archived_observations_consumed", 99),
                                          ("torch_intraop_threads", 1)])
def test_history_gate_rejects_invalid_accounting(field, value):
    value_report = report(100)
    value_report[field] = value
    assert not gate.history_gate(value_report, 100)


def test_private_inputs_reject_changed_raw_bytes(tmp_path):
    source = tmp_path / "source"
    refs = tmp_path / "refs"
    (source / "infrastructure/truba").mkdir(parents=True)
    refs.mkdir()
    raw = refs / "raw.json"
    raw.write_text('{"original": true}')
    (source / gate.DEFAULT_INPUT_MANIFEST).write_text(
        json.dumps({"raw_inputs": {raw.name: sha(raw)}}))
    gate.verify_inputs(source, refs)
    raw.write_text('{"original": false}')
    with pytest.raises(ValueError, match="checksum differs"):
        gate.verify_inputs(source, refs)


def test_test_gate_rejects_changed_existing_outcomes(monkeypatch, tmp_path):
    before = {KNOWN_NEI: "failure", ("existing", "case"): "passed"}
    after = {**before, **{("tests.orchestration.test_runner", gate.NEW_TESTS[0] + str(i) + "]"): "passed" for i in range(29)}}
    monkeypatch.setattr(gate, "test_outcomes", lambda path: before if path.name == "accepted-tests.xml" else after)
    monkeypatch.setattr(gate, "load", lambda path: {"pytest_exit": 1, "physical_backend_attempts": 0})
    assert gate.tests_gate(tmp_path, tmp_path, expectation_revision=gate.HISTORICAL_TEST_EXPECTATION, new_test_prefixes=gate.NEW_TESTS, expected_added=29)["test_gate_passed"]
    after[("existing", "case")] = "skipped"
    assert not gate.tests_gate(tmp_path, tmp_path, expectation_revision=gate.HISTORICAL_TEST_EXPECTATION, new_test_prefixes=gate.NEW_TESTS, expected_added=29)["test_gate_passed"]


def test_later_extraction_changes_only_expected_new_test_cases(monkeypatch, tmp_path):
    before = {KNOWN_NEI: "failure", ("existing", "case"): "passed"}
    after = {**before, ("tests.orchestration.test_runner", "new_collection_case"): "passed"}
    monkeypatch.setattr(gate, "test_outcomes", lambda path: before if path.name == "accepted-tests.xml" else after)
    monkeypatch.setattr(gate, "load", lambda path: {"pytest_exit": 1, "physical_backend_attempts": 0})
    assert not gate.tests_gate(tmp_path, tmp_path, expectation_revision=gate.HISTORICAL_TEST_EXPECTATION, new_test_prefixes=gate.NEW_TESTS, expected_added=29)["test_gate_passed"]
    assert gate.tests_gate(tmp_path, tmp_path, new_test_prefixes=("new_collection_case",), expected_added=1, expectation_revision=gate.HISTORICAL_TEST_EXPECTATION)["test_gate_passed"]
    after[("existing", "case")] = "skipped"
    assert not gate.tests_gate(tmp_path, tmp_path, new_test_prefixes=("new_collection_case",), expected_added=1, expectation_revision=gate.HISTORICAL_TEST_EXPECTATION)["test_gate_passed"]


def _corrected_gate_fixture(monkeypatch):
    before = {KNOWN_NEI: "failure", ("existing", "case"): "passed",
              ("existing", "optional"): "skipped"}
    after = {**before, KNOWN_NEI: "passed",
             **{key: "passed" for key in gate.SINGULAR_NEI_CASES}}
    status = {"pytest_exit": 0, "physical_backend_attempts": 0}
    monkeypatch.setattr(gate, "test_outcomes", lambda p: before if p.name == "accepted-tests.xml" else after)
    monkeypatch.setattr(gate, "load", lambda p: status)
    return before, after, status


def test_corrected_expectation_accepts_only_declared_fixture_transition(monkeypatch, tmp_path):
    before, after, _ = _corrected_gate_fixture(monkeypatch)
    original = deepcopy(before)
    result = gate.tests_gate(tmp_path, tmp_path, expected_added=0)
    assert result["test_gate_passed"] and result["declared_NEI_correction"]
    assert result["test_expectation_revision"] == gate.CORRECTED_TEST_EXPECTATION
    assert not result["known_NEI_failure_present"]
    assert before == original and after[KNOWN_NEI] == "passed"


def test_corrected_expectation_accepts_corrected_reference_without_exemptions(monkeypatch, tmp_path):
    before, after, _ = _corrected_gate_fixture(monkeypatch)
    before.update(after)
    result = gate.tests_gate(tmp_path, tmp_path, expected_added=0)
    assert result["test_gate_passed"] and not result["declared_NEI_correction"]


@pytest.mark.parametrize("change", [
    "nei_failure", "nei_error", "nei_skipped", "missing_nei", "missing_singular",
    "singular_failure", "singular_skip", "other_failure", "other_error",
    "new_skip", "changed_skip", "removed_skip", "extra_case", "pytest_exit",
    "physical_call", "invalid_reference",
])
def test_corrected_expectation_rejects_every_unapproved_outcome(monkeypatch, tmp_path, change):
    before, after, status = _corrected_gate_fixture(monkeypatch)
    if change.startswith("nei_"): after[KNOWN_NEI] = change[4:]
    elif change == "missing_nei": after.pop(KNOWN_NEI)
    elif change == "missing_singular": after.pop(gate.SINGULAR_NEI_CASES[0])
    elif change == "singular_failure": after[gate.SINGULAR_NEI_CASES[0]] = "failure"
    elif change == "singular_skip": after[gate.SINGULAR_NEI_CASES[0]] = "skipped"
    elif change == "other_failure": after[("existing", "case")] = "failure"
    elif change == "other_error": after[("existing", "case")] = "error"
    elif change == "new_skip": after[("existing", "case")] = "skipped"
    elif change == "changed_skip": after[("existing", "optional")] = "passed"
    elif change == "removed_skip": after.pop(("existing", "optional"))
    elif change == "extra_case": after[("unexpected", "case")] = "passed"
    elif change == "pytest_exit": status["pytest_exit"] = 1
    elif change == "physical_call": status["physical_backend_attempts"] = 1
    elif change == "invalid_reference": before[KNOWN_NEI] = "skipped"
    assert not gate.tests_gate(tmp_path, tmp_path, expected_added=0)["test_gate_passed"]


def test_corrected_expectation_rejects_unknown_versions(monkeypatch, tmp_path):
    _corrected_gate_fixture(monkeypatch)
    with pytest.raises(ValueError, match="unknown test expectation revision"):
        gate.tests_gate(tmp_path, tmp_path, expectation_revision="unregistered")


def test_corrected_expectation_does_not_change_historical_verdict(monkeypatch, tmp_path):
    before, after, status = _corrected_gate_fixture(monkeypatch)
    after.clear(); after.update(before); status["pytest_exit"] = 1
    assert gate.tests_gate(tmp_path, tmp_path, expected_added=0,
                           expectation_revision=gate.HISTORICAL_TEST_EXPECTATION)["test_gate_passed"]
    assert not gate.tests_gate(tmp_path, tmp_path, expected_added=0)["test_gate_passed"]
    after[KNOWN_NEI] = "passed"; status["pytest_exit"] = 0
    assert not gate.tests_gate(tmp_path, tmp_path, expected_added=0,
                               expectation_revision=gate.HISTORICAL_TEST_EXPECTATION)["test_gate_passed"]


def test_corrected_expectation_retains_explicit_new_case_limits(monkeypatch, tmp_path):
    _, after, _ = _corrected_gate_fixture(monkeypatch)
    after[("tests.orchestration.test_runner", gate.CORRECTED_NEW_TESTS[0] + "dtype0]")] = "passed"
    assert not gate.tests_gate(tmp_path, tmp_path, expected_added=0)["test_gate_passed"]
    assert gate.tests_gate(tmp_path, tmp_path, expected_added=1)["test_gate_passed"]


def test_corrected_expectation_default_cli_checks_fixture_revision(tmp_path, monkeypatch, capsys):
    import sys
    import xml.etree.ElementTree as ET
    source, refs, output = tmp_path / "source", tmp_path / "refs", tmp_path / "output"
    for p in (source / "infrastructure/truba", source / "tests/acquisition", refs, output):
        p.mkdir(parents=True, exist_ok=True)
    fixture = source / "tests/acquisition/test_nei.py"; fixture.write_text("# fixture revision\n")
    def xml(path, cases):
        tree = ET.Element("testsuite")
        for (cls, name), outcome in cases.items():
            case = ET.SubElement(tree, "testcase", classname=cls, name=name)
            if outcome != "passed": ET.SubElement(case, outcome)
        ET.ElementTree(tree).write(path)
    xml(refs / "accepted-tests.xml", {KNOWN_NEI: "failure"})
    xml(output / "junit.xml", {KNOWN_NEI: "passed", **{k: "passed" for k in gate.SINGULAR_NEI_CASES}})
    (output / "status.json").write_text(json.dumps({"pytest_exit": 0, "physical_backend_attempts": 0}))
    profile = {"test_expectation_revision": gate.CORRECTED_TEST_EXPECTATION,
               "raw_inputs": {"accepted-tests.xml": sha(refs / "accepted-tests.xml")},
               "corrected_fixture": {"path": "tests/acquisition/test_nei.py", "sha256": sha(fixture)},
               "new_test_prefixes": [], "expected_added_tests": 0}
    (source / gate.DEFAULT_INPUT_MANIFEST).write_text(json.dumps(profile))
    monkeypatch.setattr(sys, "argv", ["gate", "tests", "--source-root", str(source),
                                      "--references", str(refs), "--output-root", str(output)])
    gate.main()
    assert json.loads(capsys.readouterr().out)["test_gate_passed"]
    fixture.write_text("# unreviewed fixture revision\n")
    with pytest.raises(ValueError, match="test revision checksum differs"):
        gate.main()
