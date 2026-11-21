from copy import deepcopy
import math
import pytest
from infrastructure.truba import b100_regression as gate


def valid_history():
    return {"status": "passed", "budget": 100, "target": 0, "record_pointer": "/records/0",
            "archived_observations_consumed": 100, "ambient_rng_preserved": True,
            "physical_simulator_calls": 0, "logical_replay_accounting": dict(gate.ACCOUNTING),
            "reconstructed_gp_fits": [{"parameter": 1.0, "elapsed_seconds": 7.0}]}


@pytest.mark.parametrize("field,value", [
    ("status", "diverged"), ("archived_observations_consumed", 99),
    ("ambient_rng_preserved", False), ("physical_simulator_calls", 1),
    ("record_pointer", "/records/1"), ("budget", 40),
])
def test_incomplete_or_unsafe_replay_cannot_pass(field, value):
    report = valid_history()
    report[field] = value
    assert not gate.history_gate(report)


def test_B100_requires_85_fits_84_proposals_and_one_terminal_recommendation():
    report = valid_history()
    assert gate.history_gate(report)
    report["logical_replay_accounting"]["committed_fit_events"] = 84
    assert not gate.history_gate(report)


def test_full_history_exact_gate_excludes_only_predeclared_metadata():
    original = valid_history()
    candidate = deepcopy(original)
    candidate["reconstructed_gp_fits"][0]["elapsed_seconds"] = 99
    assert gate.compare_histories(original, candidate)["scientific_exact_equal"]
    candidate["reconstructed_gp_fits"][0]["parameter"] = math.nextafter(1.0, 2.0)
    result = gate.compare_histories(original, candidate)
    assert not result["scientific_exact_equal"]
    assert result["first_difference"]["path"] == "/reconstructed_gp_fits/0/parameter"


def test_signed_zero_difference_is_reported():
    result = gate.first_difference({"value": -0.0}, {"value": 0.0})
    assert result["path"] == "/value"
