"""Frozen random-feature GP-like landscapes for controlled diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch

from mutaco.identity import stable_hash
from mutaco.simulator import (
    SimulationContext,
    SimulationResult,
    SimulatorCapabilities,
    TensorContractError,
)
from mutaco.types import ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec


STATE_SCHEMA = "mutaco-frozen-gp-landscape-v1"


def _nested_tuple(matrix: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(matrix, list):
        raise ValueError("landscape matrix must be a list")
    return tuple(tuple(float(value) for value in row) for row in matrix)


@dataclass(frozen=True)
class FrozenGPLandscapeConfig:
    parameter_dim: int
    response_dim: int
    feature_count: int = 64
    latent_rank: int = 3
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()
    lengthscale: float = 0.65
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        for name in ("parameter_dim", "response_dim", "feature_count", "latent_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.lower:
            object.__setattr__(self, "lower", (-1.0,) * self.parameter_dim)
        if not self.upper:
            object.__setattr__(self, "upper", (1.0,) * self.parameter_dim)
        if len(self.lower) != self.parameter_dim or len(self.upper) != self.parameter_dim:
            raise ValueError("landscape bounds must match parameter_dim")
        if any(
            not math.isfinite(low)
            or not math.isfinite(high)
            or low >= high
            for low, high in zip(self.lower, self.upper)
        ):
            raise ValueError("landscape bounds must be finite and ordered")
        if not math.isfinite(self.lengthscale) or self.lengthscale <= 0:
            raise ValueError("lengthscale must be finite and positive")
        if not math.isfinite(self.amplitude) or self.amplitude <= 0:
            raise ValueError("amplitude must be finite and positive")


@dataclass(frozen=True)
class FrozenGPLandscapeState:
    """Complete device-independent state of one approximate GP prior draw."""

    config: FrozenGPLandscapeConfig
    landscape_seed: int
    frequencies: tuple[tuple[float, ...], ...]
    phases: tuple[float, ...]
    latent_weights: tuple[tuple[float, ...], ...]
    output_mixing: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]

    def __post_init__(self) -> None:
        config = self.config
        if isinstance(self.landscape_seed, bool) or not isinstance(self.landscape_seed, int):
            raise TypeError("landscape_seed must be an integer")
        if self.landscape_seed < 0:
            raise ValueError("landscape_seed must be nonnegative")
        expected_shapes = {
            "frequencies": (config.feature_count, config.parameter_dim),
            "latent_weights": (config.feature_count, config.latent_rank),
            "output_mixing": (config.latent_rank, config.response_dim),
        }
        for name, shape in expected_shapes.items():
            matrix = getattr(self, name)
            if len(matrix) != shape[0] or any(len(row) != shape[1] for row in matrix):
                raise ValueError(f"{name} must have shape {shape}")
            if any(not math.isfinite(value) for row in matrix for value in row):
                raise ValueError(f"{name} must contain only finite values")
        if len(self.phases) != config.feature_count or any(
            not math.isfinite(value) for value in self.phases
        ):
            raise ValueError("phases must be finite and match feature_count")
        if len(self.output_bias) != config.response_dim or any(
            not math.isfinite(value) for value in self.output_bias
        ):
            raise ValueError("output_bias must be finite and match response_dim")

    @classmethod
    def generate(
        cls,
        config: FrozenGPLandscapeConfig,
        *,
        landscape_seed: int,
    ) -> "FrozenGPLandscapeState":
        """Draw and persist one finite random-Fourier-feature landscape.

        A local CPU float64 generator fixes frequencies [F,P], phases [F], latent
        weights [F,L] and task mixing [L,M]. Shared latent features create correlated
        response channels. This approximate prior draw is generated once and stored
        as device-independent tuples; it is not a fitted GP or an exact kernel draw.
        """
        if isinstance(landscape_seed, bool) or not isinstance(landscape_seed, int):
            raise TypeError("landscape_seed must be an integer")
        if landscape_seed < 0:
            raise ValueError("landscape_seed must be nonnegative")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(landscape_seed)
        dtype = torch.float64
        frequencies = torch.randn(
            (config.feature_count, config.parameter_dim),
            generator=generator,
            dtype=dtype,
        ) / config.lengthscale
        phases = 2.0 * torch.pi * torch.rand(
            config.feature_count,
            generator=generator,
            dtype=dtype,
        )
        latent_weights = torch.randn(
            (config.feature_count, config.latent_rank),
            generator=generator,
            dtype=dtype,
        )
        output_mixing = torch.randn(
            (config.latent_rank, config.response_dim),
            generator=generator,
            dtype=dtype,
        ) / math.sqrt(config.latent_rank)
        output_bias = 0.1 * torch.randn(
            config.response_dim,
            generator=generator,
            dtype=dtype,
        )
        return cls(
            config=config,
            landscape_seed=landscape_seed,
            frequencies=tuple(tuple(float(value) for value in row) for row in frequencies.tolist()),
            phases=tuple(float(value) for value in phases.tolist()),
            latent_weights=tuple(
                tuple(float(value) for value in row) for row in latent_weights.tolist()
            ),
            output_mixing=tuple(
                tuple(float(value) for value in row) for row in output_mixing.tolist()
            ),
            output_bias=tuple(float(value) for value in output_bias.tolist()),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA,
            "config": {
                "parameter_dim": self.config.parameter_dim,
                "response_dim": self.config.response_dim,
                "feature_count": self.config.feature_count,
                "latent_rank": self.config.latent_rank,
                "lower": list(self.config.lower),
                "upper": list(self.config.upper),
                "lengthscale": self.config.lengthscale,
                "amplitude": self.config.amplitude,
            },
            "landscape_seed": self.landscape_seed,
            "frequencies": [list(row) for row in self.frequencies],
            "phases": list(self.phases),
            "latent_weights": [list(row) for row in self.latent_weights],
            "output_mixing": [list(row) for row in self.output_mixing],
            "output_bias": list(self.output_bias),
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash("gp-landscape-state-v1", self._payload())

    def to_document(self) -> dict[str, Any]:
        return {**self._payload(), "state_hash": self.fingerprint}

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "FrozenGPLandscapeState":
        if document.get("schema_version") != STATE_SCHEMA:
            raise ValueError("unsupported GP-landscape state schema")
        raw_config = document.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("GP-landscape state requires a config mapping")
        config = FrozenGPLandscapeConfig(
            parameter_dim=int(raw_config["parameter_dim"]),
            response_dim=int(raw_config["response_dim"]),
            feature_count=int(raw_config["feature_count"]),
            latent_rank=int(raw_config["latent_rank"]),
            lower=tuple(float(value) for value in raw_config["lower"]),
            upper=tuple(float(value) for value in raw_config["upper"]),
            lengthscale=float(raw_config["lengthscale"]),
            amplitude=float(raw_config["amplitude"]),
        )
        state = cls(
            config=config,
            landscape_seed=int(document["landscape_seed"]),
            frequencies=_nested_tuple(document["frequencies"]),
            phases=tuple(float(value) for value in document["phases"]),
            latent_weights=_nested_tuple(document["latent_weights"]),
            output_mixing=_nested_tuple(document["output_mixing"]),
            output_bias=tuple(float(value) for value in document["output_bias"]),
        )
        if document.get("state_hash") != state.fingerprint:
            raise ValueError("GP-landscape state hash mismatch")
        return state


@dataclass(frozen=True)
class FrozenGPLandscapeSimulator:
    """Evaluate a persisted correlated random-feature diagnostic landscape.

    Physical [P]/[H,P] parameters are affinely mapped to [-1,1]^P before cosine
    features and latent task mixing produce [M]/[H,M] responses. Fixed state is
    materialized on the configured float32/float64 device. Optional task-diagonal
    Gaussian observation noise uses context.evaluation_seed, not a new landscape;
    these controlled cases do not establish general benchmark performance.
    """

    state: FrozenGPLandscapeState
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64
    observation_noise_std: tuple[float, ...] | None = None
    _frequencies: torch.Tensor = field(init=False, repr=False, compare=False)
    _phases: torch.Tensor = field(init=False, repr=False, compare=False)
    _latent_weights: torch.Tensor = field(init=False, repr=False, compare=False)
    _output_mixing: torch.Tensor = field(init=False, repr=False, compare=False)
    _output_bias: torch.Tensor = field(init=False, repr=False, compare=False)
    _lower: torch.Tensor = field(init=False, repr=False, compare=False)
    _upper: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("GP landscape requires float32 or float64")
        if self.observation_noise_std is not None:
            if len(self.observation_noise_std) != self.state.config.response_dim:
                raise ValueError("observation_noise_std must match response_dim")
            if any(
                not math.isfinite(value) or value <= 0
                for value in self.observation_noise_std
            ):
                raise ValueError("observation noise scales must be finite and positive")
        options = {"dtype": self.dtype, "device": self.device}
        object.__setattr__(self, "_frequencies", torch.tensor(self.state.frequencies, **options))
        object.__setattr__(self, "_phases", torch.tensor(self.state.phases, **options))
        object.__setattr__(
            self,
            "_latent_weights",
            torch.tensor(self.state.latent_weights, **options),
        )
        object.__setattr__(
            self,
            "_output_mixing",
            torch.tensor(self.state.output_mixing, **options),
        )
        object.__setattr__(self, "_output_bias", torch.tensor(self.state.output_bias, **options))
        object.__setattr__(self, "_lower", torch.tensor(self.state.config.lower, **options))
        object.__setattr__(self, "_upper", torch.tensor(self.state.config.upper, **options))

    @property
    def capabilities(self) -> SimulatorCapabilities:
        config = self.state.config
        parameters = ParameterSpec(
            tuple(
                ParameterAxis(f"theta_{index + 1}", config.lower[index], config.upper[index])
                for index in range(config.parameter_dim)
            )
        )
        responses = ResponseSpec(
            tuple(ResponseAxis(f"g_{index + 1}") for index in range(config.response_dim))
        )
        noise_label = "deterministic" if self.observation_noise_std is None else "seeded-noise"
        return SimulatorCapabilities(
            parameter_spec=parameters,
            response_spec=responses,
            simulator_identity=f"frozen-gp-landscape.{self.state.fingerprint}.{noise_label}",
            deterministic=self.observation_noise_std is None,
            supports_batch=True,
            supports_context=True,
            definition_metadata={
                "simulator_name": "frozen_gp_landscape",
                "state_hash": self.state.fingerprint,
                "observation_noise_std": (
                    list(self.observation_noise_std)
                    if self.observation_noise_std is not None
                    else None
                ),
                "interpretation": (
                    "controlled_diagnostic_not_general_performance_evidence"
                ),
            },
        )

    def evaluate(
        self,
        theta: torch.Tensor,
        context: SimulationContext | None = None,
    ) -> SimulationResult:
        if theta.device != self.device or theta.dtype != self.dtype:
            raise TensorContractError("GP-landscape input does not match simulator device and dtype")
        if theta.ndim not in {1, 2} or theta.shape[-1] != self.state.config.parameter_dim:
            raise TensorContractError("GP-landscape theta has the wrong shape")
        normalized = 2.0 * (theta - self._lower) / (self._upper - self._lower) - 1.0
        features = math.sqrt(2.0 / self.state.config.feature_count) * torch.cos(
            normalized @ self._frequencies.transpose(-1, -2) + self._phases
        )
        response = self.state.config.amplitude * (
            features @ self._latent_weights @ self._output_mixing
        ) + self._output_bias
        seed: int | None = None
        if self.observation_noise_std is not None:
            if context is None:
                raise ValueError("noisy GP-landscape evaluation requires a SimulationContext")
            seed = context.evaluation_seed
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            scale = torch.tensor(
                self.observation_noise_std,
                dtype=self.dtype,
                device=self.device,
            )
            response = response + torch.randn(
                response.shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            ) * scale
        return SimulationResult(
            response.clone().contiguous(),
            {
                "simulator_name": "frozen_gp_landscape",
                "state_hash": self.state.fingerprint,
                "landscape_seed": self.state.landscape_seed,
                "observation_noise_seed": seed,
                "device": str(self.device),
                "dtype": str(self.dtype).removeprefix("torch."),
                "transport_call_count": 1,
                "interpretation": "controlled_diagnostic_not_general_performance_evidence",
            },
        )
