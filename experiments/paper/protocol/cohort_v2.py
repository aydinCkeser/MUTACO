"""Generic, computation-free hidden-theta cohort declarations.

The declaration contains no simulator responses and does not authorize target
construction.  It is deliberately separate from the frozen Paper-100 v1
generator, whose bytes and historical identity must remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver, stable_hash

from .validation import PaperContractError


COHORT_SCHEMA = "mutaco-hidden-theta-cohort-declaration-v2"
TARGET_BANK_POLICY = "scrambled_sobol_explicit_index_v1"


def _strings(values: Sequence[str], field: str) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in values)
    if not resolved or any(not value.strip() for value in resolved):
        raise PaperContractError(f"{field} must contain nonempty strings")
    if len(set(resolved)) != len(resolved):
        raise PaperContractError(f"{field} must be unique")
    return resolved


def _bounds(values: Sequence[Sequence[float]], dimension: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(values) != 2:
        raise PaperContractError("bounds must contain lower and upper vectors")
    lower, upper = (tuple(float(value) for value in row) for row in values)
    if len(lower) != dimension or len(upper) != dimension:
        raise PaperContractError("bounds dimension mismatch")
    if not all(math.isfinite(lo) and math.isfinite(hi) and lo < hi for lo, hi in zip(lower, upper)):
        raise PaperContractError("bounds must be finite with lower < upper")
    return lower, upper


@dataclass(frozen=True)
class CohortDeclarationSpec:
    cohort_id: str
    simulator_adapter: str
    simulator_definition_id: str
    simulator_configuration_identity: str
    simulator_configuration_sha256: str
    parameter_profile: str
    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]]
    target_count: int
    campaign_seed: int
    target_seed_namespace: str
    sobol_skip: int = 0
    sobol_scramble: bool = True
    dtype: str = "float32"
    generation_device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "cohort_id", "simulator_adapter", "simulator_definition_id",
            "simulator_configuration_identity",
            "simulator_configuration_sha256",
            "parameter_profile", "target_seed_namespace",
        ):
            if not str(getattr(self, name)).strip():
                raise PaperContractError(f"{name} must be nonempty")
        names = _strings(self.parameter_names, "parameter_names")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "bounds", _bounds(self.bounds, len(names)))
        for name in ("target_count", "campaign_seed", "sobol_skip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PaperContractError(f"{name} must be a nonnegative integer")
        if self.target_count == 0:
            raise PaperContractError("target_count must be positive")
        if len(self.simulator_configuration_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.simulator_configuration_sha256
        ):
            raise PaperContractError("simulator_configuration_sha256 must be lowercase SHA-256")
        if self.sobol_scramble is not True:
            raise PaperContractError("v2 cohort declarations require scrambled Sobol")
        if self.dtype != "float32" or self.generation_device != "cpu":
            raise PaperContractError("v2 cohort generation is explicitly CPU float32")

    def family_definition(self) -> dict[str, Any]:
        return {
            "schema_version": COHORT_SCHEMA,
            "cohort_id": self.cohort_id,
            "simulator_adapter": self.simulator_adapter,
            "simulator_definition_id": self.simulator_definition_id,
            "simulator_configuration_identity": self.simulator_configuration_identity,
            "simulator_configuration_sha256": self.simulator_configuration_sha256,
            "parameter_profile": self.parameter_profile,
            "parameter_names": list(self.parameter_names),
            "bounds": [list(row) for row in self.bounds],
            "target_count": self.target_count,
            "campaign_seed": self.campaign_seed,
            "target_seed_namespace": self.target_seed_namespace,
            "target_bank": {
                "policy": TARGET_BANK_POLICY,
                "dimension": len(self.parameter_names),
                "scramble": self.sobol_scramble,
                "skip": self.sobol_skip,
                "indexing": "ordered_contiguous_zero_based_after_skip",
                "dtype": self.dtype,
                "device": self.generation_device,
                "mapping": "lower_plus_upper_minus_lower_times_unit_sobol",
            },
        }


def materialize_cohort_declaration(spec: CohortDeclarationSpec) -> dict[str, Any]:
    """Materialize theta only; no simulator is constructed or evaluated."""

    family_definition = spec.family_definition()
    family = ScientificIdentity.create(family_definition)
    sobol_coordinate = SeedCoordinate("target_bank", ("scrambled_sobol",))
    sobol_seed = SeedDeriver(family, spec.campaign_seed).seed_for(sobol_coordinate)
    engine = torch.quasirandom.SobolEngine(
        len(spec.parameter_names), scramble=True, seed=sobol_seed
    )
    if spec.sobol_skip:
        engine.fast_forward(spec.sobol_skip)
    unit = engine.draw(spec.target_count, dtype=torch.float32)
    lower = torch.tensor(spec.bounds[0], dtype=torch.float32)
    upper = torch.tensor(spec.bounds[1], dtype=torch.float32)
    theta = lower + (upper - lower) * unit
    target_deriver = SeedDeriver(family, spec.campaign_seed)
    records = []
    for offset, row in enumerate(theta):
        sobol_index = spec.sobol_skip + offset
        coordinate = SeedCoordinate(spec.target_seed_namespace, (offset,))
        content = {
            "target_index": offset,
            "sobol_index": sobol_index,
            "theta_star": [float(value) for value in row.tolist()],
            "target_seed_coordinate": {
                "namespace": coordinate.namespace,
                "coordinates": list(coordinate.coordinates),
            },
            "target_generation_seed": target_deriver.seed_for(coordinate),
        }
        records.append({
            **content,
            "target_identity": stable_hash(
                "hidden-theta-target-v2", {"target_family_id": family.value, **content}
            ),
        })
    body = {
        **family_definition,
        "classification": "frozen_hidden_theta_declaration_no_responses",
        "execution_authorized": False,
        "target_family_id": family.value,
        "sobol_seed_coordinate": {
            "namespace": sobol_coordinate.namespace,
            "coordinates": list(sobol_coordinate.coordinates),
        },
        "sobol_seed": sobol_seed,
        "records": records,
    }
    return {**body, "declaration_hash": stable_hash("hidden-theta-cohort-v2", body)}


def validate_cohort_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly regenerate and compare a v2 theta declaration."""

    if document.get("schema_version") != COHORT_SCHEMA:
        raise PaperContractError("unsupported cohort declaration schema")
    bank = document.get("target_bank")
    if not isinstance(bank, Mapping):
        raise PaperContractError("target_bank must be an object")
    spec = CohortDeclarationSpec(
        cohort_id=str(document.get("cohort_id", "")),
        simulator_adapter=str(document.get("simulator_adapter", "")),
        simulator_definition_id=str(document.get("simulator_definition_id", "")),
        simulator_configuration_identity=str(document.get("simulator_configuration_identity", "")),
        simulator_configuration_sha256=str(document.get("simulator_configuration_sha256", "")),
        parameter_profile=str(document.get("parameter_profile", "")),
        parameter_names=tuple(document.get("parameter_names", ())),
        bounds=tuple(tuple(row) for row in document.get("bounds", ())),
        target_count=document.get("target_count"),
        campaign_seed=document.get("campaign_seed"),
        target_seed_namespace=str(document.get("target_seed_namespace", "")),
        sobol_skip=bank.get("skip"),
        sobol_scramble=bank.get("scramble"),
        dtype=str(bank.get("dtype", "")),
        generation_device=str(bank.get("device", "")),
    )
    expected = materialize_cohort_declaration(spec)
    if dict(document) != expected:
        raise PaperContractError("cohort declaration does not match exact regeneration")
    indices = [record["target_index"] for record in expected["records"]]
    sobol = [record["sobol_index"] for record in expected["records"]]
    if indices != list(range(spec.target_count)) or sobol != list(
        range(spec.sobol_skip, spec.sobol_skip + spec.target_count)
    ):
        raise PaperContractError("cohort indices are missing, duplicated, or reordered")
    return expected

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.cohort_v2")
