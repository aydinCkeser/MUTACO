from __future__ import annotations

import inspect

import pytest
import torch

from mutaco.simulator import SimulationContext, checked_evaluate
from mutaco_analytic import (
    THETA_STAR,
    HandwrittenAnalyticSimulator,
    analytic_response,
)
from mutaco_analytic import handwritten


def _context(seed: int) -> SimulationContext:
    return SimulationContext("science-analytic", "search", 0, 0, seed)


def test_historical_equations_are_preserved_exactly_in_float64() -> None:
    theta = torch.tensor(THETA_STAR, dtype=torch.float64)
    expected = torch.tensor(
        (
            0.43,
            torch.sin(torch.tensor(0.35 * torch.pi, dtype=torch.float64)).item() - 0.40,
            -0.109375,
        ),
        dtype=torch.float64,
    )
    torch.testing.assert_close(analytic_response(theta), expected, rtol=0.0, atol=1e-15)


def test_primary_reference_is_torch_only_and_does_not_expose_theta_star_in_metadata() -> None:
    source = inspect.getsource(handwritten)
    assert "import numpy" not in source
    simulator = HandwrittenAnalyticSimulator()
    result = checked_evaluate(
        simulator,
        torch.tensor((0.2, -0.4), dtype=torch.float64),
        device=torch.device("cpu"),
        dtype=torch.float64,
        context=_context(4),
    )
    assert "theta_star" not in result.metadata
    assert result.metadata["device"] == "cpu"


def test_seeded_noise_replays_and_does_not_consume_ambient_rng() -> None:
    simulator = HandwrittenAnalyticSimulator(noise_mode="task_diagonal_gaussian")
    theta = torch.tensor((0.2, -0.4), dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    first = checked_evaluate(
        simulator, theta, device=theta.device, dtype=theta.dtype, context=_context(123)
    )
    second = checked_evaluate(
        simulator, theta, device=theta.device, dtype=theta.dtype, context=_context(123)
    )
    changed = checked_evaluate(
        simulator, theta, device=theta.device, dtype=theta.dtype, context=_context(124)
    )
    assert torch.equal(first.values, second.values)
    assert not torch.equal(first.values, changed.values)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_constant_output_is_exact_and_unnoised() -> None:
    simulator = HandwrittenAnalyticSimulator(
        noise_mode="task_diagonal_gaussian",
        constant_output=7.25,
    )
    theta = torch.tensor((0.1, 0.2), dtype=torch.float64)
    result = checked_evaluate(
        simulator, theta, device=theta.device, dtype=theta.dtype, context=_context(9)
    )
    assert result.values.shape == (4,)
    assert result.values[-1].item() == 7.25


def test_float32_is_explicitly_supported_but_not_the_default() -> None:
    simulator = HandwrittenAnalyticSimulator(dtype=torch.float32)
    theta = torch.tensor((0.1, 0.2), dtype=torch.float32)
    result = checked_evaluate(
        simulator, theta, device=theta.device, dtype=theta.dtype, context=_context(2)
    )
    assert result.values.dtype == torch.float32
    assert HandwrittenAnalyticSimulator().dtype == torch.float64


@pytest.mark.cuda
def test_handwritten_simulator_uses_same_code_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("compatible Torch CUDA runtime is unavailable")
    device = torch.device("cuda:0")
    simulator = HandwrittenAnalyticSimulator(device=device)
    theta = torch.tensor(THETA_STAR, dtype=torch.float64, device=device)
    result = checked_evaluate(
        simulator, theta, device=device, dtype=torch.float64, context=_context(7)
    )
    assert result.values.device == device
