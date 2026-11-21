"""Manuscript Step 1: seeded designs in declared physical parameter bounds.

Generate coordinates only; simulator evaluation and response normalization are
owned by the runner. The configured paper design is scrambled Sobol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from mutaco.identity import stable_hash
from mutaco.policies import PolicyClassification, surface_record
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.types import ParameterSpec


class InitialDesignPolicy(str, Enum):
    SCRAMBLED_SOBOL = "scrambled_sobol"
    CALLER_SUPPLIED = "caller_supplied"
    IID_UNIFORM = "iid_uniform"


INITIAL_DESIGN_POLICIES: dict[InitialDesignPolicy, PolicyClassification] = {
    InitialDesignPolicy.SCRAMBLED_SOBOL: PolicyClassification.DEFAULT,
    InitialDesignPolicy.CALLER_SUPPLIED: PolicyClassification.APPROVED_OPTIONAL,
    InitialDesignPolicy.IID_UNIFORM: PolicyClassification.DIAGNOSTIC_ONLY,
}


@dataclass(frozen=True)
class InitialDesignResult:
    """Ordered physical design theta [N,P] and its pre-evaluation identities.

    Columns follow ParameterSpec axes; row i owns observation_ids[i]. The design
    identity includes the realized rows, so caller-supplied designs are identified
    without inventing a Sobol seed. Tensors retain the requested dtype/device;
    this record itself neither evaluates the simulator nor clones its input.
    """
    theta: torch.Tensor
    policy_id: str
    classification: str
    design_id: str
    observation_ids: tuple[str, ...]
    parameter_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    seed_coordinate: SeedCoordinate | None
    seed_value: int | None
    generation_device: str

    def provenance(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "classification": self.classification,
            "design_id": self.design_id,
            "observation_ids": self.observation_ids,
            "parameter_names": self.parameter_names,
            "lower": self.lower,
            "upper": self.upper,
            "point_count": self.theta.shape[0],
            "dtype": str(self.theta.dtype).removeprefix("torch."),
            "output_device": str(self.theta.device),
            "generation_device": self.generation_device,
            "seed_coordinate": (
                {
                    "namespace": self.seed_coordinate.namespace,
                    "coordinates": self.seed_coordinate.coordinates,
                }
                if self.seed_coordinate is not None
                else None
            ),
            "seed_value": self.seed_value,
        }


def _validate_common(
    theta: torch.Tensor,
    *,
    parameter_spec: ParameterSpec,
    point_count: int | None = None,
) -> None:
    if not isinstance(theta, torch.Tensor):
        raise TypeError("initial theta must be a torch.Tensor")
    if theta.ndim != 2 or theta.shape[1] != parameter_spec.dimension:
        raise ValueError(f"initial theta must have shape [N,{parameter_spec.dimension}]")
    if theta.shape[0] < 2:
        raise ValueError("initial design requires at least two points")
    if point_count is not None and theta.shape[0] != point_count:
        raise ValueError("initial theta point count does not match the request")
    if theta.dtype not in {torch.float32, torch.float64}:
        raise ValueError("initial theta requires float32 or float64")
    if not theta.is_contiguous() or not bool(torch.isfinite(theta).all()):
        raise ValueError("initial theta must be contiguous and finite")
    lower = torch.tensor(parameter_spec.lower, dtype=theta.dtype, device=theta.device)
    upper = torch.tensor(parameter_spec.upper, dtype=theta.dtype, device=theta.device)
    if bool(((theta < lower) | (theta > upper)).any()):
        raise ValueError("initial theta lies outside declared parameter bounds")


def _result(
    theta: torch.Tensor,
    *,
    policy: InitialDesignPolicy,
    parameter_spec: ParameterSpec,
    coordinate: SeedCoordinate | None,
    seed_value: int | None,
    generation_device: str,
) -> InitialDesignResult:
    names = tuple(axis.name for axis in parameter_spec.axes)
    semantic = {
        "policy_id": policy.value,
        "parameter_names": names,
        "lower": parameter_spec.lower,
        "upper": parameter_spec.upper,
        "dtype": str(theta.dtype).removeprefix("torch."),
        "theta": tuple(tuple(float(value) for value in row) for row in theta.detach().cpu().tolist()),
        "seed_coordinate": (
            {"namespace": coordinate.namespace, "coordinates": coordinate.coordinates}
            if coordinate is not None
            else None
        ),
        "seed_value": seed_value,
    }
    design_id = stable_hash("initial-design-v1", semantic)
    observations = tuple(f"{design_id}:observation:{index}" for index in range(theta.shape[0]))
    return InitialDesignResult(
        theta=theta,
        policy_id=policy.value,
        classification=INITIAL_DESIGN_POLICIES[policy].value,
        design_id=design_id,
        observation_ids=observations,
        parameter_names=names,
        lower=parameter_spec.lower,
        upper=parameter_spec.upper,
        seed_coordinate=coordinate,
        seed_value=seed_value,
        generation_device=generation_device,
    )


def generate_initial_design(
    *,
    parameter_spec: ParameterSpec,
    point_count: int,
    seed_deriver: SeedDeriver,
    seed_coordinate: SeedCoordinate,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    policy: InitialDesignPolicy | str = InitialDesignPolicy.SCRAMBLED_SOBOL,
) -> InitialDesignResult:
    """Generate ordered physical theta [N,P] and corresponding observation IDs.

    N is point_count (at least two), P is the parameter dimension. Use the named
    seed coordinate with a CPU Sobol engine or local IID generator, then affine
    bounds mapping and one transfer to device. Float32/float64 are supported;
    no global random stream is sampled and no simulator is called. The result
    records policy, seed and bounds so the design's identity is inspectable.
    """
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 2:
        raise ValueError("initial point_count must be an integer of at least two")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("initial design requires float32 or float64")
    device = torch.device(device)
    try:
        selected = InitialDesignPolicy(policy)
    except ValueError as exc:
        raise ValueError(f"unknown initial-design policy: {policy!r}") from exc
    if selected is InitialDesignPolicy.CALLER_SUPPLIED:
        raise ValueError("caller_supplied requires validate_caller_design")

    seed = seed_deriver.seed_for(seed_coordinate)
    # Torch Sobol is a CPU engine. Generating both policies on CPU makes the
    # design scientific identity device-independent; the one explicit transfer
    # is recorded as generation_device versus output_device.
    if selected is InitialDesignPolicy.SCRAMBLED_SOBOL:
        unit = torch.quasirandom.SobolEngine(
            parameter_spec.dimension, scramble=True, seed=seed
        ).draw(point_count, dtype=dtype)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        unit = torch.rand(
            (point_count, parameter_spec.dimension),
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
    lower = torch.tensor(parameter_spec.lower, dtype=dtype)
    upper = torch.tensor(parameter_spec.upper, dtype=dtype)
    theta = (lower + unit * (upper - lower)).to(device=device).contiguous()
    _validate_common(theta, parameter_spec=parameter_spec, point_count=point_count)
    return _result(
        theta,
        policy=selected,
        parameter_spec=parameter_spec,
        coordinate=seed_coordinate,
        seed_value=seed,
        generation_device="cpu",
    )


def validate_caller_design(
    theta: torch.Tensor,
    *,
    parameter_spec: ParameterSpec,
    dtype: torch.dtype,
    device: torch.device,
) -> InitialDesignResult:
    """Validate a caller's physical [N,P] design without reordering or cloning.

    The returned design retains the supplied tensor. Dtype, device, contiguity,
    finiteness and bounds must already satisfy the contract. Duplicate rows are
    not rejected at this boundary.
    """
    device = torch.device(device)
    if not isinstance(theta, torch.Tensor):
        raise TypeError("caller-supplied theta must be a torch.Tensor")
    if theta.dtype != dtype or theta.device != device:
        raise ValueError("caller-supplied theta must already match requested dtype and device")
    _validate_common(theta, parameter_spec=parameter_spec)
    return _result(
        theta,
        policy=InitialDesignPolicy.CALLER_SUPPLIED,
        parameter_spec=parameter_spec,
        coordinate=None,
        seed_value=None,
        generation_device=str(device),
    )


def initial_design_support_markdown() -> str:
    descriptions = {
        InitialDesignPolicy.SCRAMBLED_SOBOL: "Seeded scrambled Sobol points inside bounds",
        InitialDesignPolicy.CALLER_SUPPLIED: "Strictly validated caller points in preserved order",
        InitialDesignPolicy.IID_UNIFORM: "Seeded IID uniform points inside bounds",
    }
    lines = [
        "# Prompt 4A initial-design support",
        "",
        "This table is generated from `mutaco.design`; do not edit it independently.",
        "",
        "| Identifier | Classification | Surface | Behavior |",
        "|---|---|---|---|",
    ]
    for policy, classification in INITIAL_DESIGN_POLICIES.items():
        lines.append(
            f"| `{policy.value}` | {classification.value} | "
            f"`{surface_record(policy.value).surface.value}` | {descriptions[policy]} |"
        )
    lines.extend(
        [
            "",
            "Latin hypercube sampling is deferred and rejected as unsupported.",
        ]
    )
    return "\n".join(lines) + "\n"
