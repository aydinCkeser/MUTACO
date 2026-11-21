"""Pre-execution scientific intent and sequential observation identities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from mutaco.identity.canonical import ScientificIdentity, stable_hash
from mutaco.identity.metadata import (
    StrictJsonValue,
    freeze_strict_metadata,
    strict_metadata_document,
)
from mutaco.randomness.seeds import SEED_POLICY
from mutaco.types import ParameterSpec, ResponseSpec


SEQUENTIAL_INTENT_SCHEMA = "mutaco-sequential-scientific-intent-v1"
SEQUENTIAL_OBSERVATION_SCHEMA = "mutaco-sequential-observation-v1"
EXACT_DUPLICATE_POLICY = "reject_exact_duplicate_v1"
BEST_RANKED_UNEVALUATED_RAW_POLICY = "best_ranked_unevaluated_raw_v1"
_DUPLICATE_POLICIES = {
    EXACT_DUPLICATE_POLICY,
    BEST_RANKED_UNEVALUATED_RAW_POLICY,
}

_DERIVED_OR_EXECUTION_KEYS = {
    "candidate_seed",
    "device",
    "execution",
    "fit_seed",
    "interpreter",
    "interpreter_path",
    "machine",
    "seed_value",
    "timings",
}
_INITIAL_DESIGN_POLICIES = {
    "scrambled_sobol",
    "caller_supplied",
    "iid_uniform",
}
_RECOMMENDATION_POLICIES = {
    "final_posterior_standardized_l1",
    "evaluated_posterior_expected_l2",
    "evaluated_posterior_mean_physical_l2",
    "continuous_posterior_mean_physical_l2",
    "continuous_posterior_mean_physical_l1_mae",
}

DEFAULT_EVALUATION_SEED_POLICY = "scientific_identity_coordinate_v1"
PAIRED_EVALUATION_SEED_POLICY = "paired_coordinate_diagnostic_v1"
FIXED_FULL_STREAM_CRN_POLICY = "fixed_full_stream_crn_diagnostic_v1"
_EVALUATION_SEED_POLICIES = {
    DEFAULT_EVALUATION_SEED_POLICY,
    PAIRED_EVALUATION_SEED_POLICY,
    FIXED_FULL_STREAM_CRN_POLICY,
}


def _reject_derived_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _DERIVED_OR_EXECUTION_KEYS:
                raise ValueError(f"{path} contains derived/execution field {key!r}")
            _reject_derived_keys(item, path=f"{path}.{key}")
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            _reject_derived_keys(item, path=f"{path}[{index}]")


def _spec_document(parameter_spec: ParameterSpec, response_spec: ResponseSpec) -> dict[str, Any]:
    return {
        "parameters": tuple(
            {
                "name": axis.name,
                "lower": axis.lower,
                "upper": axis.upper,
                "unit": axis.unit,
                "transformation": axis.transformation,
                "role": axis.role,
            }
            for axis in parameter_spec.axes
        ),
        "responses": tuple(
            {"name": axis.name, "unit": axis.unit} for axis in response_spec.axes
        ),
    }


@dataclass(frozen=True)
class SequentialScientificIntent:
    """Pre-execution scientific specification used to derive the parent run identity.

    The physical target, ordered parameter/response specifications, algorithm
    options, root seed and coordinate schema are identity-bearing. Derived random
    seeds, fitted transforms, observations and timing are not inputs to this
    identity. Execution placement is recorded separately. This avoids defining a
    run's random streams using results that those streams have yet to produce.
    """

    simulator_identity: str
    simulator_definition_metadata: Mapping[str, Any]
    parameter_spec: ParameterSpec
    response_spec: ResponseSpec
    physical_target: tuple[float, ...]
    initial_design_policy: str
    initial_point_count: int
    caller_initial_points: tuple[tuple[float, ...], ...] | None
    sequential_rounds: int
    transform_settings: Mapping[str, Any]
    gp_settings: Mapping[str, Any]
    acquisition_settings: Mapping[str, Any]
    recommendation_policy: str
    dtype: torch.dtype
    root_seed: int
    seed_coordinate_schema: Mapping[str, Any]
    duplicate_policy: str = EXACT_DUPLICATE_POLICY
    seed_policy: str = SEED_POLICY
    evaluation_seed_policy: str = DEFAULT_EVALUATION_SEED_POLICY
    evaluation_seed_family_id: str | None = None
    fixed_evaluation_seed: int | None = None
    diagnostic_internal_seed_family_id: str | None = None

    def __post_init__(self) -> None:
        if not self.simulator_identity.strip():
            raise ValueError("simulator_identity must be nonempty")
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("sequential scientific intent requires float32 or float64")
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("root_seed must be an integer")
        if self.root_seed < 0:
            raise ValueError("root_seed must be nonnegative")
        for name, value, minimum in (
            ("initial_point_count", self.initial_point_count, 2),
            ("sequential_rounds", self.sequential_rounds, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if any(axis.transformation != "identity" for axis in self.parameter_spec.axes):
            raise ValueError("sequential D supports identity parameter transformations only")
        if (
            len(self.physical_target) != self.response_spec.dimension
            or any(not math.isfinite(value) for value in self.physical_target)
        ):
            raise ValueError("physical_target must contain one finite value per response")
        if self.initial_design_policy not in _INITIAL_DESIGN_POLICIES:
            raise ValueError(
                f"unsupported initial-design policy: {self.initial_design_policy!r}"
            )
        if self.recommendation_policy not in _RECOMMENDATION_POLICIES:
            raise ValueError(
                f"unsupported recommendation policy: {self.recommendation_policy!r}"
            )
        if self.duplicate_policy not in _DUPLICATE_POLICIES:
            raise ValueError(f"unsupported duplicate policy: {self.duplicate_policy!r}")

        definition = freeze_strict_metadata(
            self.simulator_definition_metadata,
            label="simulator_definition_metadata",
        )
        object.__setattr__(self, "simulator_definition_metadata", definition)
        for name in (
            "transform_settings",
            "gp_settings",
            "acquisition_settings",
            "seed_coordinate_schema",
        ):
            frozen = freeze_strict_metadata(getattr(self, name), label=name)
            _reject_derived_keys(frozen, path=name)
            object.__setattr__(self, name, frozen)

        caller = self.caller_initial_points
        if self.initial_design_policy == "caller_supplied":
            if caller is None or len(caller) != self.initial_point_count:
                raise ValueError("caller-supplied intent requires every declared initial point")
            for row in caller:
                if len(row) != self.parameter_spec.dimension:
                    raise ValueError("caller initial points have the wrong parameter dimension")
                if any(not math.isfinite(value) for value in row):
                    raise ValueError("caller initial points must be finite")
            lower = self.parameter_spec.lower
            upper = self.parameter_spec.upper
            if any(
                value < lower[index] or value > upper[index]
                for row in caller
                for index, value in enumerate(row)
            ):
                raise ValueError("caller initial points lie outside declared bounds")
        elif caller is not None:
            raise ValueError("generated initial-design intent must not contain caller points")
        if self.seed_policy != SEED_POLICY:
            raise ValueError(f"unsupported seed policy: {self.seed_policy!r}")
        if self.evaluation_seed_policy not in _EVALUATION_SEED_POLICIES:
            raise ValueError(
                f"unsupported evaluation-seed policy: {self.evaluation_seed_policy!r}"
            )
        if self.evaluation_seed_policy == DEFAULT_EVALUATION_SEED_POLICY:
            if self.evaluation_seed_family_id is not None or self.fixed_evaluation_seed is not None:
                raise ValueError("default evaluation seeds accept no diagnostic controls")
        elif self.evaluation_seed_policy == PAIRED_EVALUATION_SEED_POLICY:
            if not self.evaluation_seed_family_id:
                raise ValueError("paired evaluation seeds require a family identity")
            if self.fixed_evaluation_seed is not None:
                raise ValueError("paired evaluation seeds do not accept a fixed seed")
        else:
            if self.evaluation_seed_family_id is not None:
                raise ValueError("fixed full-stream CRN accepts no family identity")
            if (
                isinstance(self.fixed_evaluation_seed, bool)
                or not isinstance(self.fixed_evaluation_seed, int)
                or self.fixed_evaluation_seed < 0
            ):
                raise ValueError("fixed full-stream CRN requires a nonnegative seed")
        if (
            self.diagnostic_internal_seed_family_id is not None
            and not self.diagnostic_internal_seed_family_id.strip()
        ):
            raise ValueError("diagnostic internal seed family must be nonempty")

    @property
    def caller_point_count(self) -> int:
        return 0 if self.caller_initial_points is None else len(self.caller_initial_points)

    def scientific_document(self) -> dict[str, Any]:
        document = {
            "schema": SEQUENTIAL_INTENT_SCHEMA,
            "simulator": {
                "identity": self.simulator_identity,
                "definition_metadata": strict_metadata_document(
                    self.simulator_definition_metadata
                ),
            },
            "specifications": _spec_document(self.parameter_spec, self.response_spec),
            "physical_target": self.physical_target,
            "initial_design": {
                "policy": self.initial_design_policy,
                "point_count": self.initial_point_count,
                "caller_point_count": self.caller_point_count,
                "caller_points": self.caller_initial_points,
            },
            "sequential_rounds": self.sequential_rounds,
            "transform_settings": strict_metadata_document(self.transform_settings),
            "gp_settings": strict_metadata_document(self.gp_settings),
            "acquisition_settings": strict_metadata_document(self.acquisition_settings),
            "recommendation_policy": self.recommendation_policy,
            "duplicate_policy": self.duplicate_policy,
            "dtype": str(self.dtype).removeprefix("torch."),
            "root_seed": self.root_seed,
            "seed_policy": self.seed_policy,
            "seed_coordinate_schema": strict_metadata_document(
                self.seed_coordinate_schema
            ),
        }
        # Preserve every pre-Prompt-8J default scientific identity exactly.
        if self.evaluation_seed_policy != DEFAULT_EVALUATION_SEED_POLICY:
            document["diagnostic_evaluation_seed_control"] = {
                "policy": self.evaluation_seed_policy,
                "classification": "diagnostic-only",
                "family_id": self.evaluation_seed_family_id,
                "fixed_seed": self.fixed_evaluation_seed,
            }
        if self.diagnostic_internal_seed_family_id is not None:
            document["diagnostic_internal_seed_control"] = {
                "policy": "paired_internal_coordinate_seeds_v1",
                "classification": "diagnostic-only",
                "family_id": self.diagnostic_internal_seed_family_id,
            }
        return document

    def identity(self) -> ScientificIdentity:
        return ScientificIdentity.create(self.scientific_document())


def sequential_observation_identity(
    *,
    scientific_identity: ScientificIdentity,
    global_logical_index: int,
    sequential_round_index: int,
    candidate_id: str,
    theta: torch.Tensor,
) -> str:
    """Identify a sequential evaluation address without result metadata or response."""

    for name, value in (
        ("global_logical_index", global_logical_index),
        ("sequential_round_index", sequential_round_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not candidate_id:
        raise ValueError("candidate_id must be nonempty")
    if (
        not isinstance(theta, torch.Tensor)
        or theta.ndim != 1
        or theta.dtype not in {torch.float32, torch.float64}
        or not theta.is_contiguous()
        or not bool(torch.isfinite(theta).all())
    ):
        raise ValueError("sequential observation theta must be finite contiguous [P]")
    return stable_hash(
        SEQUENTIAL_OBSERVATION_SCHEMA,
        {
            "scientific_identity": scientific_identity.value,
            "phase": "sequential",
            "global_logical_index": global_logical_index,
            "sequential_round_index": sequential_round_index,
            "candidate_id": candidate_id,
            "theta": theta.detach().cpu().tolist(),
            "dtype": str(theta.dtype).removeprefix("torch."),
        },
    )
