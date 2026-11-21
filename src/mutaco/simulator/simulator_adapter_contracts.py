"""Device-resident simulator protocol with fail-closed validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

from mutaco.identity.metadata import (
    StrictJsonValue,
    StrictMetadataError,
    freeze_strict_metadata,
    strict_metadata_document,
)
from mutaco.identity.canonical import stable_hash
from mutaco.types import ParameterSpec, ResponseSpec


class TensorContractError(ValueError):
    """A simulator request or response violated the declared tensor contract."""


class SimulatorMetadataContractError(TensorContractError):
    """Simulator definition or result metadata is not strict JSON-compatible."""


class RetryableSimulationReason(str, Enum):
    """Classified infrastructure failures eligible for configured runner retries.

    A retry repeats the same scientific realization; it is not an additional
    observation or authorization to retry arbitrary physical/model errors.
    """

    TRANSIENT_RUNTIME = "transient_runtime"
    TRANSIENT_TRANSPORT = "transient_transport"
    TRANSIENT_RESOURCE = "transient_resource"


class RetryableSimulationError(RuntimeError):
    """An explicitly classified infrastructure failure; arbitrary errors do not retry."""

    def __init__(
        self,
        reason: RetryableSimulationReason | str,
        explanation: str | None = None,
    ) -> None:
        try:
            self.reason = RetryableSimulationReason(reason)
        except ValueError as exc:
            raise ValueError(f"unknown retryable simulation reason: {reason!r}") from exc
        if explanation is not None and not isinstance(explanation, str):
            raise TypeError("retryable simulation explanation must be a string or None")
        self.explanation = explanation
        message = self.reason.value
        if explanation:
            message = f"{message}: {explanation}"
        super().__init__(message)


class SimulatorCapabilityDriftError(ValueError):
    """A simulator's declared capabilities changed after the run snapshot."""


@dataclass(frozen=True)
class SimulationContext:
    """Scientific realization address and seed for one logical evaluation.

    purpose separates target, training and assessment streams; logical/repeat
    indices identify a realization. attempt_index records a retry without changing
    randomness_address(). The runner reuses evaluation_seed across those attempts
    so retrying transport does not silently draw a different physical response.
    """
    scientific_run_id: str
    purpose: str
    logical_index: int
    repeat_index: int
    evaluation_seed: int
    attempt_index: int = 0

    def __post_init__(self) -> None:
        if not self.scientific_run_id or not self.purpose:
            raise ValueError("simulation context identities must be nonempty")
        for name in ("logical_index", "repeat_index", "evaluation_seed", "attempt_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def randomness_address(self) -> tuple[str, str, int, int]:
        """Retry attempts deliberately reuse the same scientific realization."""

        return (
            self.scientific_run_id,
            self.purpose,
            self.logical_index,
            self.repeat_index,
        )


@dataclass(frozen=True)
class SimulatorCapabilities:
    """Pre-run declaration of physical coordinates, task order and backend support.

    parameter_spec and response_spec define [P] inputs and [M] outputs; definition
    metadata is recursively frozen and contributes to scientific identity. Result
    metadata describes an individual evaluation instead. Batch/context flags
    declare supported interfaces, not permission for extra physical evaluations.
    """
    parameter_spec: ParameterSpec
    response_spec: ResponseSpec
    simulator_identity: str
    deterministic: bool
    supports_batch: bool = False
    supports_context: bool = True
    definition_metadata: Mapping[str, StrictJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.simulator_identity.strip():
            raise ValueError("simulator identity must be nonempty")
        for name in ("deterministic", "supports_batch", "supports_context"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        try:
            metadata = freeze_strict_metadata(
                self.definition_metadata,
                label="simulator definition metadata",
            )
        except StrictMetadataError as exc:
            raise SimulatorMetadataContractError(str(exc)) from exc
        object.__setattr__(self, "definition_metadata", metadata)

    def scientific_document(self) -> dict[str, Any]:
        return {
            "simulator_identity": self.simulator_identity,
            "parameter_spec": [
                {
                    "name": axis.name,
                    "lower": axis.lower,
                    "upper": axis.upper,
                    "unit": axis.unit,
                    "transformation": axis.transformation,
                    "role": axis.role,
                }
                for axis in self.parameter_spec.axes
            ],
            "response_spec": [
                {"name": axis.name, "unit": axis.unit}
                for axis in self.response_spec.axes
            ],
            "deterministic": self.deterministic,
            "supports_batch": self.supports_batch,
            "supports_context": self.supports_context,
            "definition_metadata": strict_metadata_document(self.definition_metadata),
        }


@dataclass(frozen=True)
class SimulatorCapabilitySnapshot:
    """Stable pre-run declaration used to reject mid-run capability drift."""

    capability_id: str
    document: Mapping[str, StrictJsonValue]

    def __post_init__(self) -> None:
        if not self.capability_id.startswith("simulator-capabilities-v1-"):
            raise ValueError("capability_id has the wrong identity prefix")
        object.__setattr__(
            self,
            "document",
            freeze_strict_metadata(self.document, label="simulator capabilities"),
        )

    @classmethod
    def capture(cls, capabilities: SimulatorCapabilities) -> "SimulatorCapabilitySnapshot":
        validate_sequential_simulator_capabilities(capabilities)
        document = capabilities.scientific_document()
        return cls(
            capability_id=stable_hash("simulator-capabilities-v1", document),
            document=freeze_strict_metadata(document, label="simulator capabilities"),
        )

    def validate_current(self, capabilities: SimulatorCapabilities) -> None:
        current = SimulatorCapabilitySnapshot.capture(capabilities)
        if current.capability_id != self.capability_id:
            raise SimulatorCapabilityDriftError(
                "simulator capabilities changed after the run snapshot"
            )


def validate_sequential_simulator_capabilities(
    capabilities: SimulatorCapabilities,
) -> None:
    """Validate prerequisites required by the later sequential D runner."""

    if not capabilities.deterministic and not capabilities.supports_context:
        raise ValueError("stochastic sequential simulators require context support")
    if any(
        axis.transformation != "identity"
        for axis in capabilities.parameter_spec.axes
    ):
        raise ValueError(
            "sequential D supports identity parameter transformations only"
        )


@dataclass(frozen=True)
class SimulationResult:
    values: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SimulatorProtocol(Protocol):
    @property
    def capabilities(self) -> SimulatorCapabilities: ...

    def evaluate(
        self,
        theta: torch.Tensor,
        context: SimulationContext | None = None,
    ) -> SimulationResult: ...


@runtime_checkable
class BatchedSimulatorProtocol(SimulatorProtocol, Protocol):
    def evaluate_batch(
        self,
        theta: torch.Tensor,
        contexts: Sequence[SimulationContext],
    ) -> SimulationResult: ...


def _validate_tensor_common(
    value: torch.Tensor,
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TensorContractError(f"{label} must be a torch.Tensor")
    if value.device != device:
        raise TensorContractError(f"{label} device {value.device} does not match {device}")
    if value.dtype != dtype:
        raise TensorContractError(f"{label} dtype {value.dtype} does not match {dtype}")
    if not value.is_contiguous():
        raise TensorContractError(f"{label} must be contiguous")
    if value.requires_grad:
        raise TensorContractError(f"{label} must not require gradients")
    if not bool(torch.isfinite(value).all()):
        raise TensorContractError(f"{label} must contain only finite values")


def _storage_address(value: torch.Tensor) -> int:
    return value.untyped_storage().data_ptr()


def checked_evaluate(
    simulator: SimulatorProtocol,
    theta: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    context: SimulationContext | None = None,
) -> SimulationResult:
    """Validate one simulator-adapter evaluation through a fail-closed boundary.

    Accept bounded theta [P], or [H,P] when batch support is declared; require
    finite contiguous tensors of the caller's dtype/device without gradients.
    Return [M]/[H,M] in response-spec order. Reject mutated input, aliased output,
    invalid shapes or metadata; freeze returned metadata without changing values.
    This calls the supplied adapter once, whether physical, fake or replay; it
    does not implement retries or determine whether real physics was computed.
    """

    capabilities = simulator.capabilities
    _validate_tensor_common(theta, label="theta", device=device, dtype=dtype)
    expected_parameter_dim = capabilities.parameter_spec.dimension
    if theta.ndim == 1:
        if theta.shape != (expected_parameter_dim,):
            raise TensorContractError(
                f"theta shape {tuple(theta.shape)} does not match ({expected_parameter_dim},)"
            )
        expected_response_shape = (capabilities.response_spec.dimension,)
    elif theta.ndim == 2 and capabilities.supports_batch:
        if theta.shape[1] != expected_parameter_dim:
            raise TensorContractError(
                f"theta batch shape {tuple(theta.shape)} has the wrong parameter dimension"
            )
        expected_response_shape = (theta.shape[0], capabilities.response_spec.dimension)
    else:
        raise TensorContractError("theta must be [P], or [H,P] for a batch-capable simulator")

    lower = torch.tensor(capabilities.parameter_spec.lower, dtype=dtype, device=device)
    upper = torch.tensor(capabilities.parameter_spec.upper, dtype=dtype, device=device)
    if bool((theta < lower).any()) or bool((theta > upper).any()):
        raise TensorContractError("theta lies outside the declared parameter bounds")
    before = theta.clone()
    result = simulator.evaluate(theta, context)
    if not isinstance(result, SimulationResult):
        raise TensorContractError("simulator must return SimulationResult")
    if not torch.equal(theta, before):
        raise TensorContractError("simulator mutated its caller-owned theta input")
    _validate_tensor_common(result.values, label="response", device=device, dtype=dtype)
    if result.values.shape != expected_response_shape:
        raise TensorContractError(
            f"response shape {tuple(result.values.shape)} does not match {expected_response_shape}"
        )
    if _storage_address(result.values) == _storage_address(theta):
        raise TensorContractError("response must not alias the caller-owned theta storage")
    if not isinstance(result.metadata, Mapping) or not all(
        isinstance(key, str) for key in result.metadata
    ):
        raise TensorContractError("response metadata must be a string-keyed mapping")
    try:
        metadata = freeze_strict_metadata(
            result.metadata,
            label="simulator result metadata",
        )
    except StrictMetadataError as exc:
        raise SimulatorMetadataContractError(str(exc)) from exc
    return SimulationResult(result.values, metadata)


def checked_evaluate_batch(
    simulator: SimulatorProtocol,
    theta: torch.Tensor,
    contexts: Sequence[SimulationContext],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SimulationResult:
    """Evaluate ordered physical rows [H,P] with one unique context per row.

    Seeds and logical randomness addresses must both be distinct. Use the native
    batch interface when available, otherwise a transparent scalar fallback;
    returned [H,M] rows retain input order. One transport batch can contain several
    logical physical evaluations, so transport and observation counts differ.
    """

    capabilities = simulator.capabilities
    _validate_tensor_common(theta, label="theta batch", device=device, dtype=dtype)
    if theta.ndim != 2 or theta.shape[1] != capabilities.parameter_spec.dimension:
        raise TensorContractError(
            "theta batch must have shape [H,P] for the declared parameter dimension"
        )
    if len(contexts) != theta.shape[0] or any(
        not isinstance(context, SimulationContext) for context in contexts
    ):
        raise TensorContractError("batched evaluation requires one context per theta row")
    seeds = tuple(context.evaluation_seed for context in contexts)
    addresses = tuple(context.randomness_address() for context in contexts)
    if len(set(seeds)) != len(seeds) or len(set(addresses)) != len(addresses):
        raise TensorContractError(
            "batched evaluation contexts must have unique seeds and logical addresses"
        )
    lower = torch.tensor(capabilities.parameter_spec.lower, dtype=dtype, device=device)
    upper = torch.tensor(capabilities.parameter_spec.upper, dtype=dtype, device=device)
    if bool((theta < lower).any()) or bool((theta > upper).any()):
        raise TensorContractError("theta batch lies outside the declared parameter bounds")

    before = theta.clone()
    if capabilities.supports_batch:
        if not isinstance(simulator, BatchedSimulatorProtocol):
            raise TensorContractError(
                "simulator declares batch support without evaluate_batch"
            )
        result = simulator.evaluate_batch(theta, contexts)
        mode = "declared_batch"
    else:
        rows = tuple(
            checked_evaluate(
                simulator,
                row.contiguous(),
                device=device,
                dtype=dtype,
                context=context,
            ).values
            for row, context in zip(theta, contexts)
        )
        result = SimulationResult(torch.stack(rows).contiguous())
        mode = "transparent_scalar_fallback"
    if not isinstance(result, SimulationResult):
        raise TensorContractError("batched simulator must return SimulationResult")
    if not torch.equal(theta, before):
        raise TensorContractError("simulator mutated its caller-owned theta batch")
    _validate_tensor_common(result.values, label="response batch", device=device, dtype=dtype)
    expected = (theta.shape[0], capabilities.response_spec.dimension)
    if result.values.shape != expected:
        raise TensorContractError(
            f"response batch shape {tuple(result.values.shape)} does not match {expected}"
        )
    if _storage_address(result.values) == _storage_address(theta):
        raise TensorContractError("response batch must not alias theta storage")
    metadata = {
        "batch_evaluation_policy": "per_logical_simulation_context_v1",
        "execution_mode": mode,
        "logical_call_count": theta.shape[0],
        "evaluation_seeds": list(seeds),
    }
    return SimulationResult(
        result.values,
        freeze_strict_metadata(metadata, label="batched simulator result metadata"),
    )
