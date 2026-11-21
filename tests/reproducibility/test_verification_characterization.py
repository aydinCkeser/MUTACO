"""Contracts recorded before adding the read-only verification entry point."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from infrastructure.truba import refactor_reference as packets
from infrastructure.truba import readability_regression as regression
from infrastructure.truba import b100_regression as histories
from infrastructure.truba import fit_event_evidence as events
from infrastructure.truba import shareable_evidence as b40
from infrastructure.truba import b100_evidence as b100
from verification_fixtures import checkpoint_fixture, fit_inputs, test_gate_inputs, write_json


ROOT = Path(__file__).resolve().parents[2]


def test_independent_archived_portable_results_are_exact() -> None:
    assert b40.verify(ROOT / "results/reproducibility/shareable_milestone_v1") == {
        "derived_integrity_verified": True, "raw_original_checks_verified": False,
        "scientific_payload_sha256": "12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c",
        "replay_mathematics_sha256": "687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782",
        "new_numerical_calculations": 0,
    }
    assert b100.verify(ROOT / "results/reproducibility/b100_regression_v1") == {
        "derived_integrity_verified": True, "private_original_bytes_verified": False,
        "reference_established": True, "baseline_refactor_equivalence_established": True,
    }


def test_exact_types_signed_zero_and_first_difference_order() -> None:
    assert histories.first_difference({"x": -0.0}, {"x": 0.0}) == {
        "path": "/x", "kind": "value", "expected": -0.0, "actual": 0.0}
    assert histories.first_difference([1], [True]) == {
        "path": "/0", "kind": "type", "expected_type": "int", "actual_type": "bool"}
    assert histories.first_difference({"b": 1, "a": 2}, {"b": 3, "a": 4})["path"] == "/b"
    assert histories.first_difference([1], []) == {"path": "", "kind": "length", "expected": 1, "actual": 0}
    assert packets.digest(-0.0) != packets.digest(0.0)


def test_fixed_checkpoint_schema_rejects_tampered_values(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    original = checkpoint_fixture()
    write_json(a, original); write_json(b, original)
    result = packets.compare_files(a, b)
    assert result["artifacts_self_consistent"] and result["same_environment"] and result["scientific_exact_equal"]
    changed = deepcopy(original)
    changed["first_adaptive"]["capture"]["reconstructed"]["candidate"][1] = 0.0
    write_json(b, changed)
    result = packets.compare_files(a, b)
    assert not result["artifacts_self_consistent"] and not result["scientific_exact_equal"]


def test_fixed_fit_events_are_not_generated_by_the_expected_event_function() -> None:
    history, observed = fit_inputs()
    result = events.compare_fit_events(history, observed)
    assert result["events_exact_equal"] and result["event_boundaries_preserved"]
    assert result["expected_event_count"] == result["observed_event_count"] == 2
    observed["events"][1]["message"] = "wrong"
    assert not events.compare_fit_events(history, observed)["events_exact_equal"]


@pytest.mark.parametrize("change", ["missing", "extra", "failure", "changed_skip"])
def test_fixed_test_inventory_rejects_undeclared_changes(change, tmp_path: Path) -> None:
    paths = test_gate_inputs(tmp_path)
    specification = regression.verify_inputs(paths["source"], paths["references"], paths["manifest"])
    options = {"expected_added": 0, "new_test_prefixes": [],
               "expectation_revision": specification["test_expectation_revision"]}
    assert regression.tests_gate(paths["output"], paths["references"], **options)["test_gate_passed"]
    path = paths["output"] / "junit.xml"
    tree = ET.parse(path)
    if change == "missing": tree.getroot().remove(tree.getroot()[0])
    elif change == "extra": ET.SubElement(tree.getroot(), "testcase", classname="unexpected", name="extra")
    elif change == "failure": ET.SubElement(tree.getroot()[1], "failure")
    else: tree.getroot()[2].remove(tree.getroot()[2][0])
    tree.write(path)
    assert not regression.tests_gate(paths["output"], paths["references"], **options)["test_gate_passed"]


def test_reference_paths_are_explicit_and_checksum_checks_remain_strict(tmp_path: Path) -> None:
    paths = test_gate_inputs(tmp_path)
    # The old contract checks declared files, not directory-set equality.
    (paths["references"] / "unlisted.txt").write_text("not read or counted")
    regression.verify_inputs(paths["source"], paths["references"], paths["manifest"])
    reference = paths["references"] / "accepted-tests.xml"
    reference.write_text("changed")
    with pytest.raises(ValueError, match="private reference checksum differs: accepted-tests.xml"):
        regression.verify_inputs(paths["source"], paths["references"], paths["manifest"])
    reference.unlink()
    with pytest.raises(FileNotFoundError):
        regression.verify_inputs(paths["source"], paths["references"], paths["manifest"])
