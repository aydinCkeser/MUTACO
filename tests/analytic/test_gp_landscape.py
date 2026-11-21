from __future__ import annotations

import copy

import pytest
import torch

from mutaco.simulator import SimulationContext, checked_evaluate
from mutaco_analytic import (
    FrozenGPLandscapeConfig,
    FrozenGPLandscapeSimulator,
    FrozenGPLandscapeState,
)


def _context(seed: int) -> SimulationContext:
    return SimulationContext("science-landscape", "search", 0, 0, seed)


@pytest.mark.parametrize("parameter_dim", [2, 6])
def test_dimension_generic_scalar_and_batch_contract(parameter_dim: int) -> None:
    config = FrozenGPLandscapeConfig(
        parameter_dim=parameter_dim,
        response_dim=4,
        feature_count=16,
        latent_rank=2,
    )
    state = FrozenGPLandscapeState.generate(config, landscape_seed=77)
    simulator = FrozenGPLandscapeSimulator(state)
    scalar_theta = torch.zeros(parameter_dim, dtype=torch.float64)
    batch_theta = torch.zeros((3, parameter_dim), dtype=torch.float64)
    scalar = checked_evaluate(
        simulator, scalar_theta, device=scalar_theta.device, dtype=scalar_theta.dtype
    )
    batch = checked_evaluate(
        simulator, batch_theta, device=batch_theta.device, dtype=batch_theta.dtype
    )
    assert scalar.values.shape == (4,)
    assert batch.values.shape == (3, 4)
    assert simulator.capabilities.parameter_spec.dimension == parameter_dim


def test_landscape_state_is_seeded_persisted_hashed_and_replayable() -> None:
    config = FrozenGPLandscapeConfig(2, 3, feature_count=12, latent_rank=2)
    first = FrozenGPLandscapeState.generate(config, landscape_seed=11)
    second = FrozenGPLandscapeState.generate(config, landscape_seed=11)
    changed = FrozenGPLandscapeState.generate(config, landscape_seed=12)
    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint
    document = first.to_document()
    restored = FrozenGPLandscapeState.from_document(document)
    assert restored == first
    assert restored.fingerprint == first.fingerprint


def test_landscape_state_rejects_hash_tampering() -> None:
    state = FrozenGPLandscapeState.generate(FrozenGPLandscapeConfig(2, 2), landscape_seed=3)
    document = copy.deepcopy(state.to_document())
    document["output_bias"][0] += 0.5
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenGPLandscapeState.from_document(document)


def test_frozen_buffers_are_device_resident_and_reused() -> None:
    state = FrozenGPLandscapeState.generate(
        FrozenGPLandscapeConfig(2, 2, feature_count=8),
        landscape_seed=4,
    )
    simulator = FrozenGPLandscapeSimulator(state)
    address = simulator._frequencies.untyped_storage().data_ptr()
    theta = torch.zeros(2, dtype=torch.float64)
    _ = simulator.evaluate(theta)
    _ = simulator.evaluate(theta)
    assert simulator._frequencies.untyped_storage().data_ptr() == address
    assert simulator._frequencies.device == theta.device
    assert simulator._frequencies.dtype == theta.dtype


def test_deterministic_and_noisy_landscapes_have_separate_replay_semantics() -> None:
    state = FrozenGPLandscapeState.generate(FrozenGPLandscapeConfig(2, 3), landscape_seed=44)
    theta = torch.tensor((0.15, -0.3), dtype=torch.float64)
    deterministic = FrozenGPLandscapeSimulator(state)
    latent_first = checked_evaluate(
        deterministic, theta, device=theta.device, dtype=theta.dtype
    )
    latent_second = checked_evaluate(
        deterministic, theta, device=theta.device, dtype=theta.dtype
    )
    noisy = FrozenGPLandscapeSimulator(state, observation_noise_std=(0.1, 0.1, 0.1))
    noisy_first = checked_evaluate(
        noisy, theta, device=theta.device, dtype=theta.dtype, context=_context(101)
    )
    noisy_second = checked_evaluate(
        noisy, theta, device=theta.device, dtype=theta.dtype, context=_context(101)
    )
    noisy_changed = checked_evaluate(
        noisy, theta, device=theta.device, dtype=theta.dtype, context=_context(102)
    )
    assert torch.equal(latent_first.values, latent_second.values)
    assert torch.equal(noisy_first.values, noisy_second.values)
    assert not torch.equal(noisy_first.values, noisy_changed.values)
    assert noisy_first.metadata["state_hash"] == state.fingerprint
    assert "not_general_performance_evidence" in noisy_first.metadata["interpretation"]


def test_generation_and_evaluation_are_ambient_rng_neutral() -> None:
    before = torch.random.get_rng_state().clone()
    state = FrozenGPLandscapeState.generate(FrozenGPLandscapeConfig(2, 2), landscape_seed=8)
    simulator = FrozenGPLandscapeSimulator(state, observation_noise_std=(0.1, 0.1))
    _ = simulator.evaluate(torch.zeros(2, dtype=torch.float64), _context(9))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_simulator_contains_no_hidden_target_or_inference_gp_state() -> None:
    state = FrozenGPLandscapeState.generate(FrozenGPLandscapeConfig(2, 2), landscape_seed=5)
    simulator = FrozenGPLandscapeSimulator(state)
    assert not hasattr(simulator, "theta_star")
    assert not hasattr(simulator, "model")
    assert not hasattr(simulator, "likelihood")
