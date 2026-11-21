"""Strict Torch simulator boundary."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import simulator_adapter_contracts
from .simulator_adapter_contracts import (
    BatchedSimulatorProtocol,
    RetryableSimulationError,
    RetryableSimulationReason,
    SimulatorCapabilityDriftError,
    SimulatorCapabilitySnapshot,
    SimulationContext,
    SimulationResult,
    SimulatorCapabilities,
    SimulatorMetadataContractError,
    SimulatorProtocol,
    TensorContractError,
    checked_evaluate,
    checked_evaluate_batch,
    validate_sequential_simulator_capabilities,
)

protocol = _bind_legacy_module(simulator_adapter_contracts, __name__ + ".protocol")

__all__ = [
    "BatchedSimulatorProtocol",
    "RetryableSimulationError",
    "RetryableSimulationReason",
    "SimulatorCapabilityDriftError",
    "SimulatorCapabilitySnapshot",
    "SimulationContext",
    "SimulationResult",
    "SimulatorCapabilities",
    "SimulatorMetadataContractError",
    "SimulatorProtocol",
    "TensorContractError",
    "checked_evaluate",
    "checked_evaluate_batch",
    "validate_sequential_simulator_capabilities",
]
