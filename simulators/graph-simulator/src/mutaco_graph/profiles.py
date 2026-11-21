"""Versioned graph parameter and response profiles."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from mutaco.types import ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec


RETAINED_ORACLE_IDENTITY = "graph-oscillator.retained-oracle.v1"
SCIENTIFIC_IDENTITY = "graph-oscillator.overdamped-scientific.v1"
RETAINED_P2_ORACLE_IDENTITY = "graph-oscillator.retained-p2-three-force-oracle.v1"

FULL_PARAMETER_ORDER = (
    "weight_mean",
    "pairwise_coupling_mean",
    "bias_scale_mean",
    "pairwise_coupling_CV",
    "weight_CV",
    "bias_scale_CV",
    "inverse_viscosity_CV",
)
DEFAULT_VALUES = MappingProxyType(
    {
        "weight_mean": 1.0,
        "pairwise_coupling_mean": 1.0,
        "bias_scale_mean": 1.0,
        "pairwise_coupling_CV": 0.25,
        "weight_CV": 0.25,
        "bias_scale_CV": 0.25,
        "inverse_viscosity_CV": 0.25,
    }
)
PARAMETER_BOUNDS = MappingProxyType(
    {
        "weight_mean": (0.5, 2.0),
        "pairwise_coupling_mean": (0.5, 2.0),
        "bias_scale_mean": (0.5, 2.0),
        "pairwise_coupling_CV": (0.0, 1.0),
        "weight_CV": (0.0, 0.5),
        "bias_scale_CV": (0.0, 0.5),
        "inverse_viscosity_CV": (0.0, 0.5),
    }
)


@dataclass(frozen=True)
class GraphParameterProfile:
    """Ordered inferred graph parameters plus defaults for all uninferred fields.

    parameter_names determines theta [P] order, not FULL_PARAMETER_ORDER. Bounds
    apply to means and coefficients of variation (CV=std/mean). expand_theta maps
    physical values directly, retaining the supplied tensor dtype/device.
    """
    identifier: str
    display_name: str
    parameter_names: tuple[str, ...]
    manuscript_profile: bool

    @property
    def dimension(self) -> int:
        return len(self.parameter_names)

    @property
    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(
            tuple(
                ParameterAxis(name, *PARAMETER_BOUNDS[name])
                for name in self.parameter_names
            )
        )

    def expand_theta(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Validate physical theta [P] and fill every fixed graph parameter by name.

        Return scalar tensors in theta's dtype/device. Inferred entries are views of
        theta.unbind(), while default entries are newly constructed tensors; this is
        not a normalized-coordinate transform or a fresh disorder realization.
        """
        if not isinstance(theta, torch.Tensor) or theta.ndim != 1:
            raise ValueError(f"{self.identifier} theta must have shape [{self.dimension}]")
        if theta.shape != (self.dimension,):
            raise ValueError(f"{self.identifier} theta must have shape [{self.dimension}]")
        if not theta.is_floating_point() or not bool(torch.isfinite(theta).all()):
            raise ValueError("graph theta must be finite and floating point")
        lower = torch.tensor(
            tuple(PARAMETER_BOUNDS[name][0] for name in self.parameter_names),
            device=theta.device,
            dtype=theta.dtype,
        )
        upper = torch.tensor(
            tuple(PARAMETER_BOUNDS[name][1] for name in self.parameter_names),
            device=theta.device,
            dtype=theta.dtype,
        )
        if bool((theta < lower).any()) or bool((theta > upper).any()):
            raise ValueError(f"{self.identifier} theta lies outside declared bounds")
        result = {
            name: torch.tensor(value, dtype=theta.dtype, device=theta.device)
            for name, value in DEFAULT_VALUES.items()
        }
        result.update(dict(zip(self.parameter_names, theta.unbind())))
        return result


PARAMETER_PROFILES: Mapping[str, GraphParameterProfile] = MappingProxyType(
    {
        "p2-reduced-coupling": GraphParameterProfile(
            "p2-reduced-coupling",
            "reduced coupling",
            ("pairwise_coupling_mean", "pairwise_coupling_CV"),
            True,
        ),
        "p3-weight-coupling": GraphParameterProfile(
            "p3-weight-coupling",
            "historical weight/coupling intermediate",
            ("weight_mean", "pairwise_coupling_mean", "pairwise_coupling_CV"),
            False,
        ),
        "p4-paper-baseline": GraphParameterProfile(
            "p4-paper-baseline",
            "scientific paper baseline",
            (
                "weight_mean",
                "pairwise_coupling_mean",
                "bias_scale_mean",
                "pairwise_coupling_CV",
            ),
            True,
        ),
        "p5-manuscript-scaling": GraphParameterProfile(
            "p5-manuscript-scaling",
            "manuscript dimension scaling",
            (
                "weight_mean",
                "pairwise_coupling_mean",
                "bias_scale_mean",
                "pairwise_coupling_CV",
                "weight_CV",
            ),
            True,
        ),
        "p6-bias-cv-extension": GraphParameterProfile(
            "p6-bias-cv-extension",
            "bias-CV extension",
            (
                "weight_mean",
                "pairwise_coupling_mean",
                "bias_scale_mean",
                "pairwise_coupling_CV",
                "weight_CV",
                "bias_scale_CV",
            ),
            False,
        ),
    }
)


def parameter_profile(identifier: str) -> GraphParameterProfile:
    try:
        return PARAMETER_PROFILES[identifier]
    except KeyError as exc:
        raise ValueError(f"unknown graph parameter profile: {identifier!r}") from exc


@dataclass(frozen=True)
class GraphResponseProfile:
    identifier: str
    forces: tuple[float, ...]
    gauges: tuple[float, ...] = (0.0, torch.pi)

    @property
    def dimension(self) -> int:
        return 2 * len(self.forces) * len(self.gauges)

    @property
    def names(self) -> tuple[str, ...]:
        conditions = tuple(
            (force, gauge_name)
            for force in self.forces
            for gauge_name in ("0", "pi")
        )
        return tuple(
            f"{quantity}(F={force}, gauge={gauge})"
            for quantity in ("omega", "Z_magnitude")
            for force, gauge in conditions
        )

    @property
    def response_spec(self) -> ResponseSpec:
        return ResponseSpec(tuple(ResponseAxis(name) for name in self.names))


SCIENTIFIC_M20 = GraphResponseProfile(
    "m20-five-force-scientific",
    (0.0, 1.25, 2.5, 3.75, 5.0),
)
RETAINED_M12 = GraphResponseProfile(
    "m12-three-force-retained-smoke",
    (0.0, 2.5, 5.0),
)


@dataclass(frozen=True)
class RetainedP2OracleProfile:
    """Compatibility facade for the audited historical P=2 adapter."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return PARAMETER_PROFILES["p2-reduced-coupling"].parameter_names

    @property
    def response_names(self) -> tuple[str, ...]:
        return RETAINED_M12.names

    def expand_theta(self, theta: torch.Tensor) -> torch.Tensor:
        profile = parameter_profile("p2-reduced-coupling")
        if theta.ndim == 1:
            expanded = profile.expand_theta(theta)
            return torch.stack(
                (
                    expanded["weight_mean"],
                    expanded["pairwise_coupling_mean"],
                    expanded["bias_scale_mean"],
                    expanded["pairwise_coupling_CV"],
                )
            ).contiguous()
        if theta.ndim == 2:
            return torch.stack(
                tuple(self.expand_theta(row) for row in theta),
                dim=0,
            ).contiguous()
        raise ValueError("retained P2 theta must have shape [2] or [H,2]")
