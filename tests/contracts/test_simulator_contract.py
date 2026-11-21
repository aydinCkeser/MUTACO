from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from mutaco.simulator import (
    SimulationContext,
    SimulationResult,
    SimulatorCapabilities,
    TensorContractError,
    checked_evaluate,
    checked_evaluate_batch,
)
from mutaco.types import ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec
from mutaco_analytic import HandwrittenAnalyticSimulator


CPU = torch.device("cpu")
DTYPE = torch.float64


def test_scalar_and_batch_contract_preserve_device_dtype_and_input() -> None:
    simulator = HandwrittenAnalyticSimulator()
    theta = torch.tensor((0.2, -0.4), dtype=DTYPE)
    before = theta.clone()
    scalar = checked_evaluate(simulator, theta, device=CPU, dtype=DTYPE)
    batch = checked_evaluate(
        simulator,
        torch.stack((theta, theta)),
        device=CPU,
        dtype=DTYPE,
    )
    assert scalar.values.shape == (3,)
    assert batch.values.shape == (2, 3)
    assert scalar.values.device == theta.device
    assert scalar.values.dtype == theta.dtype
    assert scalar.values.untyped_storage().data_ptr() != theta.untyped_storage().data_ptr()
    assert torch.equal(theta, before)


@pytest.mark.parametrize(
    "theta, message",
    [
        (torch.zeros(3, dtype=DTYPE), "shape"),
        (torch.tensor((0.0, float("inf")), dtype=DTYPE), "finite"),
        (torch.tensor((2.0, 0.0), dtype=DTYPE), "bounds"),
        (torch.zeros(2, dtype=torch.float32), "dtype"),
        (torch.zeros(4, dtype=DTYPE)[::2], "contiguous"),
    ],
)
def test_request_contract_fails_closed(theta: torch.Tensor, message: str) -> None:
    with pytest.raises(TensorContractError, match=message):
        checked_evaluate(
            HandwrittenAnalyticSimulator(),
            theta,
            device=CPU,
            dtype=DTYPE,
        )


def _capabilities() -> SimulatorCapabilities:
    return SimulatorCapabilities(
        ParameterSpec((ParameterAxis("theta", -1.0, 1.0),)),
        ResponseSpec((ResponseAxis("response"),)),
        "malicious-test-simulator",
        deterministic=True,
    )


@dataclass
class MutatingSimulator:
    capabilities: SimulatorCapabilities = _capabilities()

    def evaluate(self, theta: torch.Tensor, context=None) -> SimulationResult:
        theta.add_(1.0)
        return SimulationResult(theta.clone())


@dataclass
class AliasingSimulator:
    capabilities: SimulatorCapabilities = _capabilities()

    def evaluate(self, theta: torch.Tensor, context=None) -> SimulationResult:
        return SimulationResult(theta)


@dataclass
class InvalidResponseSimulator:
    value: torch.Tensor
    capabilities: SimulatorCapabilities = _capabilities()

    def evaluate(self, theta: torch.Tensor, context=None) -> SimulationResult:
        return SimulationResult(self.value)


def test_input_mutation_and_response_aliasing_are_rejected() -> None:
    theta = torch.tensor((0.0,), dtype=DTYPE)
    with pytest.raises(TensorContractError, match="mutated"):
        checked_evaluate(MutatingSimulator(), theta.clone(), device=CPU, dtype=DTYPE)
    with pytest.raises(TensorContractError, match="alias"):
        checked_evaluate(AliasingSimulator(), theta.clone(), device=CPU, dtype=DTYPE)


class ContextRecordingSimulator:
    def __init__(self) -> None:
        self.capabilities = SimulatorCapabilities(
            ParameterSpec((ParameterAxis("theta", -1.0, 1.0),)),
            ResponseSpec((ResponseAxis("response"),)),
            "context-recording-simulator",
            deterministic=False,
            supports_batch=False,
            supports_context=True,
        )
        self.contexts: list[SimulationContext] = []

    def evaluate(
        self, theta: torch.Tensor, context: SimulationContext | None = None
    ) -> SimulationResult:
        assert context is not None
        self.contexts.append(context)
        return SimulationResult(
            torch.tensor(
                [float(context.evaluation_seed % 997)],
                dtype=theta.dtype,
                device=theta.device,
            )
        )


def test_checked_batch_requires_unique_context_per_row_and_scalar_fallback() -> None:
    simulator = ContextRecordingSimulator()
    theta = torch.tensor([[-0.5], [0.0], [0.5]], dtype=DTYPE)
    contexts = tuple(
        SimulationContext("batch-test", "training", index, 0, 100 + index)
        for index in range(3)
    )
    result = checked_evaluate_batch(
        simulator, theta, contexts, device=CPU, dtype=DTYPE
    )
    assert result.values.shape == (3, 1)
    assert result.metadata["execution_mode"] == "transparent_scalar_fallback"
    assert tuple(item.evaluation_seed for item in simulator.contexts) == (100, 101, 102)
    with pytest.raises(TensorContractError, match="unique"):
        checked_evaluate_batch(
            simulator,
            theta[:2].contiguous(),
            (
                SimulationContext("batch-test", "training", 0, 0, 100),
                SimulationContext("batch-test", "training", 0, 0, 100),
            ),
            device=CPU,
            dtype=DTYPE,
        )


@pytest.mark.parametrize(
    "value, message",
    [
        (torch.zeros(2, dtype=DTYPE), "shape"),
        (torch.zeros(1, dtype=torch.float32), "dtype"),
        (torch.tensor((float("nan"),), dtype=DTYPE), "finite"),
    ],
)
def test_invalid_responses_are_rejected(value: torch.Tensor, message: str) -> None:
    with pytest.raises(TensorContractError, match=message):
        checked_evaluate(
            InvalidResponseSimulator(value),
            torch.tensor((0.0,), dtype=DTYPE),
            device=CPU,
            dtype=DTYPE,
        )
