from __future__ import annotations

import struct

from experiments.paper.common.aggregate import _bootstrap_mean_interval
from experiments.paper.common.terminal_domain_heldout_assessment import (
    BOOTSTRAP_SEED,
    _canonical_theta,
    _context,
    verdict_for,
)


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
    declaration, tasks, preflight = native_terminal_preparation("terminal_domain_heldout_assessment")
    assert declaration["call_accounting"]["attempted"] == 0
    assert preflight["gate_passed"] is True
    assert preflight["simulator_calls"] == 0
    assert len(tasks) == 240
    for position in range(120):
        pair = [task for task in tasks if task["target_position"] == position]
        assert [task["policy"] for task in pair] == ["B_prime", "C_prime"]
        assert pair[0]["assessment_context"] == pair[1]["assessment_context"]
        assert pair[0]["theta_float32_little_endian_hex"] != pair[1]["theta_float32_little_endian_hex"]
