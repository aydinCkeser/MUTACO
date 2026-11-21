"""Frozen componentwise response coordinates for manuscript Steps 1--4.

Map physical y to (y-c)/s using initial-design statistics only. The same fixed
coordinates are used by training, the target and acquisition; terminal scoring
maps posterior means back to physical response units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from mutaco.identity import stable_hash
from mutaco.types import ResponseSpec


TRANSFORM_SCHEMA = "mutaco-frozen-response-transform-v1"
STANDARDIZE_INITIAL_DESIGN = "initial_design_zscore_v1"
IDENTITY_PHYSICAL = "identity_physical_v1"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _validate_response_tensor(
    value: torch.Tensor,
    *,
    label: str,
    response_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.ndim not in {1, 2} or value.shape[-1] != response_dim:
        raise ValueError(f"{label} must have shape [M] or [N,M] with M={response_dim}")
    if value.dtype != dtype:
        raise ValueError(f"{label} dtype {value.dtype} does not match transform dtype {dtype}")
    if value.device != device:
        raise ValueError(f"{label} device {value.device} does not match transform device {device}")
    if not value.is_contiguous():
        raise ValueError(f"{label} must be contiguous")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must contain only finite values")


@dataclass(frozen=True)
class ResponseTransformMetadata:
    schema_version: str
    transform_id: str
    response_names: tuple[str, ...]
    fitted_observation_ids: tuple[str, ...]
    mean: tuple[float, ...]
    empirical_scale: tuple[float, ...]
    effective_scale: tuple[float, ...]
    floor_activated: tuple[bool, ...]
    scale_floor: float
    degrees_of_freedom: int
    dtype: str
    fit_device: str
    transform_policy: str = STANDARDIZE_INITIAL_DESIGN

    def semantic_document(self) -> dict[str, Any]:
        """Scientific transform identity; execution device is deliberately absent."""

        document = {
            "schema_version": self.schema_version,
            "response_names": self.response_names,
            "fitted_observation_ids": self.fitted_observation_ids,
            "mean": self.mean,
            "empirical_scale": self.empirical_scale,
            "effective_scale": self.effective_scale,
            "floor_activated": self.floor_activated,
            "scale_floor": self.scale_floor,
            "degrees_of_freedom": self.degrees_of_freedom,
            "dtype": self.dtype,
        }
        # Preserve every historical fitted-transform identity exactly.
        if self.transform_policy != STANDARDIZE_INITIAL_DESIGN:
            document["transform_policy"] = self.transform_policy
        return document

    def provenance_document(self) -> dict[str, Any]:
        return {**self.semantic_document(), "transform_id": self.transform_id, "fit_device": self.fit_device}


@dataclass(frozen=True)
class StandardizedTarget:
    """Target vector [M] in one frozen response coordinate system.

    response_names fixes task ordering and transform_id prevents comparing a
    posterior with a target standardized by different initial observations.
    Validation retains values without cloning; callers must not mutate them.
    """
    values: torch.Tensor
    response_names: tuple[str, ...]
    transform_id: str

    def __post_init__(self) -> None:
        if not self.transform_id:
            raise ValueError("standardized target requires a transform_id")
        if self.values.ndim != 1 or self.values.shape != (len(self.response_names),):
            raise ValueError("standardized target values and names must have matching [M] shape")
        if not self.values.is_contiguous() or not bool(torch.isfinite(self.values).all()):
            raise ValueError("standardized target must be contiguous and finite")


@dataclass(frozen=True)
class FrozenResponseTransform:
    """Componentwise transform fitted once from ordered initial responses.

    M response components retain ResponseSpec order. Floating statistics are
    detached copies in the fit dtype/device; callers must not mutate them.
    Metadata binds the statistics to observation IDs and component ordering.
    """

    metadata: ResponseTransformMetadata
    mean: torch.Tensor
    empirical_scale: torch.Tensor
    scale: torch.Tensor
    floor_activated: torch.Tensor

    @classmethod
    def fit(
        cls,
        responses_physical: torch.Tensor,
        *,
        response_spec: ResponseSpec,
        observation_ids: tuple[str, ...],
        scale_floor: float = 1e-12,
    ) -> "FrozenResponseTransform":
        """Fit c=mean(y) and s=max(sample_std(y), scale_floor) from [N,M].

        Require N>=2 and contiguous finite float32/float64 responses. Reduction
        uses the input dtype/device and ddof=1, matching the manuscript. The floor
        protects nearly constant components; it does not add observation noise.
        This method does not mutate its input. Reuse the returned statistics for
        every subsequent response and target rather than fitting on new data.
        """
        if not isinstance(responses_physical, torch.Tensor):
            raise TypeError("responses_physical must be a torch.Tensor")
        if responses_physical.ndim != 2:
            raise ValueError("responses_physical must have shape [N,M]")
        if responses_physical.shape[0] < 2:
            raise ValueError("ddof=1 response transformation requires at least two observations")
        if responses_physical.shape[1] != response_spec.dimension:
            raise ValueError("response dimension does not match response_spec")
        if responses_physical.dtype not in {torch.float32, torch.float64}:
            raise ValueError("response transform requires float32 or float64")
        if not responses_physical.is_contiguous():
            raise ValueError("responses_physical must be contiguous")
        if not bool(torch.isfinite(responses_physical).all()):
            raise ValueError("responses_physical must contain only finite values")
        if len(observation_ids) != responses_physical.shape[0]:
            raise ValueError("observation_ids must identify every initial-design response")
        if len(set(observation_ids)) != len(observation_ids) or any(not item for item in observation_ids):
            raise ValueError("observation_ids must be unique and nonempty")
        if not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("scale_floor must be finite and positive")

        mean = responses_physical.mean(dim=0)
        empirical = responses_physical.std(dim=0, correction=1)
        floor = torch.as_tensor(
            scale_floor,
            dtype=responses_physical.dtype,
            device=responses_physical.device,
        )
        activated = empirical < floor
        effective = torch.maximum(empirical, floor)
        names = tuple(axis.name for axis in response_spec.axes)
        semantic = {
            "schema_version": TRANSFORM_SCHEMA,
            "response_names": names,
            "fitted_observation_ids": observation_ids,
            "mean": tuple(float(value) for value in mean.detach().cpu().tolist()),
            "empirical_scale": tuple(float(value) for value in empirical.detach().cpu().tolist()),
            "effective_scale": tuple(float(value) for value in effective.detach().cpu().tolist()),
            "floor_activated": tuple(bool(value) for value in activated.detach().cpu().tolist()),
            "scale_floor": float(scale_floor),
            "degrees_of_freedom": 1,
            "dtype": _dtype_name(responses_physical.dtype),
        }
        transform_id = stable_hash("response-transform-v1", semantic)
        metadata = ResponseTransformMetadata(
            **semantic,
            transform_id=transform_id,
            fit_device=str(responses_physical.device),
        )
        return cls(
            metadata=metadata,
            mean=mean.detach().clone().contiguous(),
            empirical_scale=empirical.detach().clone().contiguous(),
            scale=effective.detach().clone().contiguous(),
            floor_activated=activated.detach().clone().contiguous(),
        )

    @classmethod
    def identity(
        cls,
        responses_physical: torch.Tensor,
        *,
        response_spec: ResponseSpec,
        observation_ids: tuple[str, ...],
    ) -> "FrozenResponseTransform":
        """Create the explicit no-standardization ablation, not fitted moments.

        The physical [N,M] initial responses are validated and identified, but the
        center is zero and scale is one on their dtype/device. This distinct policy
        keeps physical response coordinates and must not be confused with the paper's
        initial-design sample-standard-deviation transform.
        """

        if not isinstance(responses_physical, torch.Tensor):
            raise TypeError("responses_physical must be a torch.Tensor")
        if responses_physical.ndim != 2 or responses_physical.shape[0] < 2:
            raise ValueError("identity response transform requires physical [N,M] with N>=2")
        if responses_physical.shape[1] != response_spec.dimension:
            raise ValueError("response dimension does not match response_spec")
        if responses_physical.dtype not in {torch.float32, torch.float64}:
            raise ValueError("response transform requires float32 or float64")
        if not responses_physical.is_contiguous() or not bool(torch.isfinite(responses_physical).all()):
            raise ValueError("responses_physical must be finite and contiguous")
        if len(observation_ids) != responses_physical.shape[0]:
            raise ValueError("observation_ids must identify every initial-design response")
        if len(set(observation_ids)) != len(observation_ids) or any(not item for item in observation_ids):
            raise ValueError("observation_ids must be unique and nonempty")

        dimension = response_spec.dimension
        zeros = torch.zeros(dimension, dtype=responses_physical.dtype, device=responses_physical.device)
        ones = torch.ones_like(zeros)
        activated = torch.zeros(dimension, dtype=torch.bool, device=responses_physical.device)
        names = tuple(axis.name for axis in response_spec.axes)
        semantic = {
            "schema_version": TRANSFORM_SCHEMA,
            "response_names": names,
            "fitted_observation_ids": observation_ids,
            "mean": tuple(0.0 for _ in range(dimension)),
            "empirical_scale": tuple(1.0 for _ in range(dimension)),
            "effective_scale": tuple(1.0 for _ in range(dimension)),
            "floor_activated": tuple(False for _ in range(dimension)),
            "scale_floor": 0.0,
            "degrees_of_freedom": 0,
            "dtype": _dtype_name(responses_physical.dtype),
            "transform_policy": IDENTITY_PHYSICAL,
        }
        transform_id = stable_hash("response-transform-v1", semantic)
        metadata = ResponseTransformMetadata(
            **semantic,
            transform_id=transform_id,
            fit_device=str(responses_physical.device),
        )
        return cls(
            metadata=metadata,
            mean=zeros.contiguous(),
            empirical_scale=ones.contiguous(),
            scale=ones.clone().contiguous(),
            floor_activated=activated.contiguous(),
        )

    @property
    def response_dim(self) -> int:
        return len(self.metadata.response_names)

    @property
    def dtype(self) -> torch.dtype:
        return self.mean.dtype

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def _validate_names(self, response_names: tuple[str, ...]) -> None:
        if response_names != self.metadata.response_names:
            raise ValueError("response component names/order do not match the fitted transform")

    def to_standardized(
        self,
        response_physical: torch.Tensor,
        *,
        response_names: tuple[str, ...],
    ) -> torch.Tensor:
        """Return (y-c)/s with shape [M] or [N,M] preserved.

        Names/order, dtype, device and contiguity must match the fitted transform;
        no implicit conversion or mutation of the physical input is performed.
        """
        self._validate_names(response_names)
        _validate_response_tensor(
            response_physical,
            label="response_physical",
            response_dim=self.response_dim,
            dtype=self.dtype,
            device=self.device,
        )
        return ((response_physical - self.mean) / self.scale).contiguous()

    def to_physical(
        self,
        response_standardized: torch.Tensor,
        *,
        response_names: tuple[str, ...],
    ) -> torch.Tensor:
        """Invert the frozen affine map, returning physical [M] or [N,M]."""
        self._validate_names(response_names)
        _validate_response_tensor(
            response_standardized,
            label="response_standardized",
            response_dim=self.response_dim,
            dtype=self.dtype,
            device=self.device,
        )
        return (response_standardized * self.scale + self.mean).contiguous()

    def standardize_target(
        self,
        target_physical: torch.Tensor,
        *,
        response_names: tuple[str, ...],
    ) -> StandardizedTarget:
        """Bind a physical target [M] to this transform's standardized coordinates."""
        values = self.to_standardized(target_physical, response_names=response_names)
        if values.ndim != 1:
            raise ValueError("target_physical must have shape [M]")
        return StandardizedTarget(values, response_names, self.metadata.transform_id)

    def _validate_covariance(
        self,
        covariance: torch.Tensor,
        *,
        response_names: tuple[str, ...],
        label: str,
    ) -> None:
        self._validate_names(response_names)
        if not isinstance(covariance, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if covariance.ndim not in {2, 3} or covariance.shape[-2:] != (
            self.response_dim,
            self.response_dim,
        ):
            raise ValueError(f"{label} must have shape [M,M] or [N,M,M]")
        if covariance.dtype != self.dtype or covariance.device != self.device:
            raise ValueError(f"{label} must match transform dtype and device")
        if not covariance.is_contiguous():
            raise ValueError(f"{label} must be contiguous")
        if not bool(torch.isfinite(covariance).all()):
            raise ValueError(f"{label} must contain only finite values")

    def covariance_to_standardized(
        self,
        covariance_physical: torch.Tensor,
        *,
        response_names: tuple[str, ...],
    ) -> torch.Tensor:
        """Map [M,M] or [N,M,M] covariance by dividing entry (p,q) by s[p]s[q]."""
        self._validate_covariance(
            covariance_physical,
            response_names=response_names,
            label="covariance_physical",
        )
        return (
            covariance_physical / self.scale.unsqueeze(-1) / self.scale.unsqueeze(-2)
        ).contiguous()

    def covariance_to_physical(
        self,
        covariance_standardized: torch.Tensor,
        *,
        response_names: tuple[str, ...],
    ) -> torch.Tensor:
        """Invert covariance scaling with the same frozen component order."""
        self._validate_covariance(
            covariance_standardized,
            response_names=response_names,
            label="covariance_standardized",
        )
        return (
            covariance_standardized * self.scale.unsqueeze(-1) * self.scale.unsqueeze(-2)
        ).contiguous()
