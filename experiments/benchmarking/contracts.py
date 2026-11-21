"""Typed experiment-owned contracts shared by benchmark adapters and methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch

from mutaco import ScientificIdentity
from mutaco.simulator import SimulatorProtocol


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class MutacoBudget:
    """Resolved total/initial budget with legacy explicit-round validation."""

    total_inference_calls: int
    initial_points: int
    sequential_rounds: int
    legacy_sequential_rounds_supplied: bool
    total_inference_calls_source: str

    @classmethod
    def resolve(cls, config: Mapping[str, Any]) -> "MutacoBudget":
        total = config.get("simulation_budget_per_target")
        runner = config.get("runner")
        if not isinstance(runner, Mapping):
            raise ValueError("MUTACO budget resolution requires runner configuration")
        initial = runner.get("initial_points")
        if isinstance(initial, bool) or not isinstance(initial, int) or initial < 2:
            raise ValueError("runner.initial_points must be an integer of at least 2")
        supplied = "sequential_rounds" in runner
        legacy = runner.get("sequential_rounds") if supplied else None
        if supplied and (
            isinstance(legacy, bool)
            or not isinstance(legacy, int)
            or legacy < 0
        ):
            raise ValueError(
                "legacy runner.sequential_rounds must be a nonnegative integer"
            )
        if total is None:
            if not supplied:
                raise ValueError(
                    "simulation_budget_per_target is required when the legacy "
                    "sequential-round field is absent"
                )
            assert isinstance(legacy, int)
            total = initial + legacy
            total_source = "legacy_initial_plus_sequential"
        elif isinstance(total, bool) or not isinstance(total, int) or total < 1:
            raise ValueError("simulation_budget_per_target must be a positive integer")
        else:
            total_source = "configured_total"
        if initial > total:
            raise ValueError(
                "runner.initial_points must not exceed simulation_budget_per_target"
            )
        derived = total - initial
        if supplied:
            if legacy != derived:
                raise ValueError(
                    "legacy runner.sequential_rounds must equal the value derived "
                    "from simulation_budget_per_target minus runner.initial_points"
                )
        return cls(total, initial, derived, supplied, total_source)

    def document(self) -> dict[str, Any]:
        return {
            "total_inference_calls": self.total_inference_calls,
            "initial_points": self.initial_points,
            "sequential_rounds": self.sequential_rounds,
            "sequential_rounds_source": "derived_total_minus_initial",
            "total_inference_calls_source": self.total_inference_calls_source,
            "legacy_sequential_rounds_supplied": (
                self.legacy_sequential_rounds_supplied
            ),
        }


@dataclass(frozen=True)
class TargetSelection:
    """Resolved, ordered target selection."""

    policy: str
    indices: tuple[int, ...]

    @classmethod
    def resolve(cls, document: Mapping[str, Any], bank_size: int) -> "TargetSelection":
        policy = str(document.get("policy", "full_bank"))
        if policy == "full_bank":
            indices = tuple(range(bank_size))
        elif policy == "prefix":
            count = document.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("prefix count must be a positive integer")
            if count > bank_size:
                raise ValueError("prefix count exceeds target bank")
            indices = tuple(range(count))
        elif policy == "explicit_indices":
            raw = document.get("indices")
            if not isinstance(raw, list) or not raw:
                raise ValueError("explicit_indices requires a nonempty list")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
                raise ValueError("target indices must be integers")
            indices = tuple(raw)
            if len(set(indices)) != len(indices):
                raise ValueError("target indices must be unique")
            if any(item < 0 or item >= bank_size for item in indices):
                raise ValueError("target index is out of range")
        else:
            raise ValueError(f"unknown target selection policy: {policy!r}")
        return cls(policy, indices)


@dataclass(frozen=True)
class Campaign:
    """Validated portable campaign configuration."""

    source_path: Path
    document: JsonObject
    simulator_identifier: str
    reference_method: str
    sbi_methods: tuple[str, ...]
    requested_device: str


@dataclass(frozen=True)
class AdapterTargetBank:
    theta: torch.Tensor
    seed: int
    science: ScientificIdentity
    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]]


class SimulatorAdapter(Protocol):
    identifier: str

    def target_bank(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AdapterTargetBank: ...

    def build_simulator(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SimulatorProtocol: ...

    def latent_response(self, theta: torch.Tensor) -> torch.Tensor | None: ...


@dataclass(frozen=True)
class CampaignExecution:
    output: Path
    manifest: JsonObject
    rows: tuple[JsonObject, ...]


REGISTERED_SBI_METHODS = (
    "sbi_npe",
    "sbi_snpe",
    "sbi_nle",
    "sbi_snle",
    "sbi_nre",
    "sbi_snre",
)


def validate_method_selection(
    reference_method: str, sbi_methods: Sequence[str]
) -> tuple[str, ...]:
    if reference_method != "mutaco":
        raise ValueError("reference_method must be 'mutaco'")
    methods = tuple(sbi_methods)
    if len(set(methods)) != len(methods):
        raise ValueError("sbi method identifiers must be unique")
    unknown = set(methods) - set(REGISTERED_SBI_METHODS)
    if unknown:
        raise ValueError(f"unknown SBI methods: {sorted(unknown)}")
    return methods
