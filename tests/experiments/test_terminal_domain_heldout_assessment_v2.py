from __future__ import annotations

import struct
from types import MappingProxyType

from experiments.paper.common.aggregate import _bootstrap_mean_interval
from experiments.paper.common.terminal_domain_heldout_assessment_v2 import (
    BOOTSTRAP_SEED,
    _append_json,
    _canonical_theta,
    _context,
    _load_hashed,
    canonicalize_result_metadata,
    serializer_self_check,
    verdict_for,
    v1_failure_metadata_shape,
)


def test_serializer_handles_top_level_and_nested_mappingproxy() -> None:
    value = MappingProxyType(
        {"top": MappingProxyType({"nested": MappingProxyType({"value": 3})})}
    )
    assert canonicalize_result_metadata(value) == {"top": {"nested": {"value": 3}}}


def test_serializer_handles_mappingproxy_in_lists_tuples_and_scalar_vectors() -> None:
    value = MappingProxyType(
        {
            "items": (
                MappingProxyType({"vector": (1, 2.5, True, None, "x")}),
                [MappingProxyType({"ordinary": [3, 4]})],
            )
        }
    )
    assert canonicalize_result_metadata(value) == {
        "items": [{"vector": [1, 2.5, True, None, "x"]}, [{"ordinary": [3, 4]}]]
    }


def test_exact_v1_failure_shape_is_canonical_and_deterministic() -> None:
    metadata = v1_failure_metadata_shape()
    first = canonicalize_result_metadata(metadata)
    second = canonicalize_result_metadata(metadata)
    assert first == second
    assert first["seed_manifest"]["thermal_noise/0"] == 14
    assert '"seed_manifest"' in __import__("json").dumps(first, sort_keys=True, allow_nan=False)
    assert serializer_self_check()["simulator_calls"] == 0


def test_serializer_persist_reload_and_mock_write_ahead(tmp_path) -> None:
    calls = {"simulator_constructed": 0, "simulator_calls": 0}
    intent = tmp_path / "mock.intent.json"
    outcome = tmp_path / "mock.outcome.json"
    _append_json(intent, {
        "status": "attempted_pending_outcome",
        "call_identity": "mock-call",
        "simulator_calls": 0,
    })
    mock_response = [float(index) / 20.0 for index in range(20)]
    _append_json(outcome, {
        "status": "success",
        "call_identity": "mock-call",
        "response": mock_response,
        "simulator_result_metadata": canonicalize_result_metadata(v1_failure_metadata_shape()),
        "simulator_calls": 0,
    })
    assert _load_hashed(intent)["status"] == "attempted_pending_outcome"
    reloaded = _load_hashed(outcome)
    assert reloaded["response"] == mock_response
    assert reloaded["simulator_result_metadata"]["seed_manifest"]["initial_state/0"] == 13
    assert calls == {"simulator_constructed": 0, "simulator_calls": 0}


def test_float32_recommendation_freeze_and_context_pairing() -> None:
    theta, raw_hex, identity = _canonical_theta([0.5, 0.7, 1.2, 1.0])
    assert bytes.fromhex(raw_hex) == struct.pack("<4f", *theta)
    assert len(identity) == 64
    first = _context(0, "target-a")
    repeat = _context(0, "target-a")
    other = _context(1, "target-b")
    assert first == repeat
    assert first["evaluation_seed"] != other["evaluation_seed"]
    assert first["family_identity"] != other["family_identity"]


def test_bootstrap_sign_and_verdict_rules() -> None:
    negative = _bootstrap_mean_interval(
        [-1.0] * 120, seed=BOOTSTRAP_SEED, resamples=100
    )
    positive = _bootstrap_mean_interval(
        [1.0] * 120, seed=BOOTSTRAP_SEED, resamples=100
    )
    assert negative == [-1.0, -1.0]
    assert positive == [1.0, 1.0]
    assert verdict_for(complete_pairs=120, interval=negative).endswith("SUPPORTED")
    assert verdict_for(complete_pairs=120, interval=positive).endswith("REVERSED")
    assert verdict_for(complete_pairs=120, interval=[-0.1, 0.1]).endswith("NOT_DETECTED")
    assert verdict_for(complete_pairs=119, interval=negative).endswith("INCOMPLETE")


def test_zero_call_preparation_has_complete_paired_inventory(native_terminal_preparation) -> None:
    declaration, tasks, preflight = native_terminal_preparation("terminal_domain_heldout_assessment_v2")
    assert declaration["call_accounting"]["attempted"] == 0
    assert preflight["gate_passed"] is True
    assert preflight["simulator_calls"] == 0
    assert len(tasks) == 240
    for position in range(120):
        pair = [task for task in tasks if task["target_position"] == position]
        assert [task["policy"] for task in pair] == ["B_prime", "C_prime"]
        assert pair[0]["assessment_context"] == pair[1]["assessment_context"]
        assert pair[0]["theta_float32_little_endian_hex"] != pair[1]["theta_float32_little_endian_hex"]
