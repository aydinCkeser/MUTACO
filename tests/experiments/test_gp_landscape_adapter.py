from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.benchmarking.simulators import FrozenGPLandscapeAdapter
from mutaco import SimulationContext, checked_evaluate


ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    return json.loads(
        (ROOT / "experiments/paper/analytic_gp_landscape/configs/supporting_campaign.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gp_landscape_adapter_shapes_bounds_replay_rng_dtype_and_device(dtype):
    config = _config()
    config["dtype"] = str(dtype).removeprefix("torch.")
    adapter = FrozenGPLandscapeAdapter()
    bank = adapter.target_bank(config, device=torch.device("cpu"), dtype=dtype)
    assert bank.theta.shape == (12, 6)
    assert torch.all(bank.theta >= -1) and torch.all(bank.theta <= 1)
    simulator = adapter.build_simulator(config, device=torch.device("cpu"), dtype=dtype)
    assert simulator.capabilities.response_spec.dimension == 4
    context = SimulationContext("test", "evaluation", 0, 0, 123)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    first = checked_evaluate(simulator, bank.theta[0], device=torch.device("cpu"), dtype=dtype, context=context)
    second = checked_evaluate(simulator, bank.theta[0], device=torch.device("cpu"), dtype=dtype, context=context)
    after = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    assert first.values.shape == (4,)
    assert first.values.dtype == dtype and first.values.device.type == "cpu"
    assert torch.equal(first.values, second.values)
    assert before[0] == after[0]
    assert np.array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])
    assert first.metadata["transport_call_count"] == 1


def test_gp_landscape_adapter_rejects_state_hash_mismatch():
    config = _config()
    config["simulator"]["frozen_state"]["expected_state_hash"] = "wrong"
    with pytest.raises(ValueError, match="state hash mismatch"):
        FrozenGPLandscapeAdapter().target_bank(
            config, device=torch.device("cpu"), dtype=torch.float32
        )
