"""Unindexed CUDA requests retain explicit simulator and acquisition placement checks."""
from types import SimpleNamespace

import pytest
import torch

from mutaco import ParameterAxis, ParameterSpec, SimulationContext, StandardizedTarget
from mutaco.acquisition.candidate_proposal import _validate_inputs
from mutaco.sequential_search.sequential_runner import _contract_device
from mutaco.simulator import TensorContractError
from mutaco_analytic import HandwrittenAnalyticSimulator


def test_only_unindexed_cuda_uses_the_current_device(monkeypatch):
    selected = torch.device("cuda")
    cpu = torch.device("cpu")
    explicit = torch.device("cuda:2")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert _contract_device(selected) == torch.device("cuda:3")
    assert _contract_device(explicit) is explicit
    assert _contract_device(cpu) is cpu
    assert selected.index is None


def test_acquisition_keeps_strict_indexed_cpu_placement():
    # Torch creates an ordinary CPU tensor for a cpu:0 selector. The historic
    # CPU contract still compares against the surrogate's declared device.
    cpu = torch.device("cpu")
    theta = torch.zeros((1, 1), dtype=torch.float64, device=cpu)
    surrogate = SimpleNamespace(parameter_dim=1, transform_id="fixture",
                                dtype=theta.dtype, device=torch.device("cpu:0"))
    target = StandardizedTarget(theta[0], ("y",), surrogate.transform_id)
    with pytest.raises(ValueError, match="target must match surrogate dtype and device"):
        _validate_inputs(surrogate, parameter_spec=ParameterSpec((ParameterAxis("x", -1., 1.),)),
                         target=target, evaluated_theta=theta)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA allocation")
def test_cuda_selector_preserves_anchored_rng_and_rejects_misplaced_tensors():
    request = torch.device("cuda")
    concrete = torch.device("cuda", torch.cuda.current_device())
    simulator = HandwrittenAnalyticSimulator(device=request, noise_mode="task_diagonal_gaussian")
    explicit = HandwrittenAnalyticSimulator(device=concrete, noise_mode="task_diagonal_gaussian")
    theta = torch.tensor([0.2, -0.4], dtype=torch.float64, device=concrete)
    context = SimulationContext("selector-test", "search", 0, 0, 94)
    before_cpu = torch.get_rng_state().clone()
    before_gpu = torch.cuda.get_rng_state(concrete).clone()
    original_selection = simulator.evaluate(theta, context)
    assert torch.equal(original_selection.values, explicit.evaluate(theta, context).values)
    assert original_selection.metadata["device"] == "cuda"
    assert simulator.device == request
    assert torch.equal(torch.get_rng_state(), before_cpu)
    assert torch.equal(torch.cuda.get_rng_state(concrete), before_gpu)
    wrong = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        wrong = torch.device("cuda", (concrete.index + 1) % torch.cuda.device_count())
        with torch.cuda.device(wrong):
            assert torch.equal(simulator.evaluate(theta, context).values, original_selection.values)
    with pytest.raises(TensorContractError, match="simulator device and dtype"):
        simulator.evaluate(theta.to(wrong), context)

    params = ParameterSpec((ParameterAxis("x", -1., 1.),))
    surrogate = SimpleNamespace(parameter_dim=1, dtype=theta.dtype, device=request, transform_id="fixture")
    target = StandardizedTarget(theta[:1], ("y",), surrogate.transform_id)
    evaluated = theta[:1].reshape(1, 1)
    assert _validate_inputs(surrogate, parameter_spec=params, target=target,
                            evaluated_theta=evaluated).device == concrete
    with pytest.raises(ValueError, match="evaluated_theta must match surrogate dtype and device"):
        _validate_inputs(surrogate, parameter_spec=params, target=target, evaluated_theta=evaluated.to(wrong))
    wrong_target = StandardizedTarget(theta[:1].to(wrong), ("y",), surrogate.transform_id)
    with pytest.raises(ValueError, match="target must match surrogate dtype and device"):
        _validate_inputs(surrogate, parameter_spec=params, target=wrong_target, evaluated_theta=evaluated)
