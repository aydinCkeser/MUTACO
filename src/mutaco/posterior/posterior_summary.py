"""Validated standardized latent posterior moments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import torch


class InvalidPosteriorError(ValueError):
    """Posterior moments violate D's latent covariance contract."""


class CovarianceValidationPolicy(str, Enum):
    """Typed numerical policies for validating latent covariance matrices."""

    ABSOLUTE_V1 = "absolute_v1"
    DTYPE_DIMENSION_SCALE_AWARE_V1 = "dtype_dimension_scale_aware_v1"
    STRUCTURAL_SYMMETRIC_MATERIALIZATION_V1 = (
        "structural_symmetric_materialization_v1"
    )


@dataclass(frozen=True)
class LatentPosteriorMeanSummary:
    """Latent standardized mean [M] or [H,M] without computing covariance.

    M follows response/task order; H follows query order. Finite contiguous
    float32/float64 values are retained without cloning. transform_id and model_id
    bind these means to the same scientific state as covariance-based interfaces.
    """

    mean: torch.Tensor
    transform_id: str
    model_id: str

    def __post_init__(self) -> None:
        if not self.transform_id or not self.model_id:
            raise InvalidPosteriorError("transform_id and model_id must be nonempty")
        if not isinstance(self.mean, torch.Tensor):
            raise InvalidPosteriorError("posterior mean must be a Torch tensor")
        if self.mean.ndim not in {1, 2}:
            raise InvalidPosteriorError("posterior mean must have shape [M] or [H,M]")
        if self.mean.dtype not in {torch.float32, torch.float64}:
            raise InvalidPosteriorError("posterior mean requires float32 or float64")
        if not self.mean.is_contiguous() or not bool(torch.isfinite(self.mean).all()):
            raise InvalidPosteriorError("posterior mean must be contiguous and finite")

    @property
    def response_dim(self) -> int:
        return self.mean.shape[-1]

    @property
    def is_batched(self) -> bool:
        return self.mean.ndim == 2


@dataclass(frozen=True)
class JointLatentPosteriorSummary:
    """Joint latent moments across query points, excluding observation noise.

    mean is [H,M] and covariance is [H*M,H*M], with the same dtype/device.
    Point-major index i*M + m denotes task m at point i. Cross-point blocks are
    retained for joint sampling; they are absent from LatentPosteriorSummary.
    Validation requires exact structural symmetry and permits only numerical-scale
    negative eigenvalues; it does not project the matrix onto the PSD cone.
    """

    mean: torch.Tensor
    covariance: torch.Tensor
    transform_id: str
    model_id: str
    materialization_diagnostics: dict[str, object] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.transform_id or not self.model_id:
            raise InvalidPosteriorError("joint posterior identities must be nonempty")
        if not isinstance(self.mean, torch.Tensor) or self.mean.ndim != 2:
            raise InvalidPosteriorError("joint posterior mean must have shape [H,M]")
        if self.mean.shape[0] < 1 or self.mean.shape[1] < 1:
            raise InvalidPosteriorError("joint posterior requires positive H and M")
        if self.mean.dtype not in {torch.float32, torch.float64}:
            raise InvalidPosteriorError("joint posterior requires float32 or float64")
        if not self.mean.is_contiguous() or not bool(torch.isfinite(self.mean).all()):
            raise InvalidPosteriorError("joint posterior mean must be finite and contiguous")
        dimension = self.mean.numel()
        if (
            not isinstance(self.covariance, torch.Tensor)
            or self.covariance.shape != (dimension, dimension)
        ):
            raise InvalidPosteriorError(
                "joint posterior covariance must have shape [H*M,H*M]"
            )
        if (
            self.covariance.dtype != self.mean.dtype
            or self.covariance.device != self.mean.device
            or not self.covariance.is_contiguous()
            or not bool(torch.isfinite(self.covariance).all())
        ):
            raise InvalidPosteriorError(
                "joint posterior covariance must match mean dtype/device and be finite/contiguous"
            )
        if not torch.equal(self.covariance, self.covariance.transpose(-1, -2)):
            raise InvalidPosteriorError(
                "joint posterior covariance must be structurally symmetric"
            )
        eigenvalues = torch.linalg.eigvalsh(self.covariance)
        eps = torch.finfo(self.covariance.dtype).eps
        scale = eigenvalues.abs().amax().clamp_min(1.0)
        tolerance = 8.0 * dimension * eps * scale
        if bool(eigenvalues[0] < -tolerance):
            raise InvalidPosteriorError(
                "joint posterior covariance is materially non-positive-semidefinite"
            )

    @property
    def point_count(self) -> int:
        return self.mean.shape[0]

    @property
    def response_dim(self) -> int:
        return self.mean.shape[1]

    @property
    def flattened_dimension(self) -> int:
        return self.mean.numel()


def structurally_symmetric_materialize(
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Materialize a self-adjoint GP covariance without hiding defects.

    The raw guard uses a first-order dense-linear-algebra envelope,
    ``8 M eps kappa``, capped at ``sqrt(eps)``. An ``8 eps`` absolute floor
    covers entries near zero. These controls are derived from precision and
    conditioning, not fitted to a campaign residual.
    """

    if not isinstance(covariance, torch.Tensor):
        raise InvalidPosteriorError("structural covariance must be a Torch tensor")
    if covariance.dtype not in {torch.float32, torch.float64}:
        raise InvalidPosteriorError("structural covariance requires float32 or float64")
    if covariance.ndim not in {2, 3} or covariance.shape[-1] != covariance.shape[-2]:
        raise InvalidPosteriorError(
            "structural covariance must have shape [M,M] or [H,M,M]"
        )
    if not bool(torch.isfinite(covariance).all()):
        raise InvalidPosteriorError("structural covariance must be finite")

    transpose = covariance.transpose(-1, -2)
    asymmetry = (covariance - transpose).abs().amax(dim=(-2, -1))
    scale = covariance.abs().amax(dim=(-2, -1))
    symmetric = (0.5 * (covariance + transpose)).contiguous()
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    spectral_scale = eigenvalues.abs().amax(dim=-1)
    minimum_absolute_eigenvalue = eigenvalues.abs().amin(dim=-1)
    condition = torch.where(
        minimum_absolute_eigenvalue > 0,
        spectral_scale / minimum_absolute_eigenvalue,
        torch.full_like(spectral_scale, torch.inf),
    )
    eps = torch.finfo(covariance.dtype).eps
    response_dim = covariance.shape[-1]
    relative_cap = math.sqrt(eps)
    first_order_relative = 8.0 * response_dim * eps * condition.clamp_min(1.0)
    relative_limit = torch.clamp(first_order_relative, max=relative_cap)
    absolute_floor = 8.0 * eps
    asymmetry_limit = absolute_floor + relative_limit * scale
    ratio = torch.where(
        asymmetry_limit > 0,
        asymmetry / asymmetry_limit,
        torch.where(
            asymmetry == 0,
            torch.zeros_like(asymmetry),
            torch.full_like(asymmetry, torch.inf),
        ),
    )
    relative_asymmetry = torch.where(
        scale > 0,
        asymmetry / scale,
        torch.where(
            asymmetry == 0,
            torch.zeros_like(asymmetry),
            torch.full_like(asymmetry, torch.inf),
        ),
    )
    psd_limit = absolute_floor + 4.0 * response_dim * eps * spectral_scale
    minimum_eigenvalue = eigenvalues[..., 0]
    diagnostics = {
        "policy_id": CovarianceValidationPolicy.STRUCTURAL_SYMMETRIC_MATERIALIZATION_V1.value,
        "source_stage": "botorch_latent_posterior_dense_materialization",
        "dtype": str(covariance.dtype),
        "response_dim": response_dim,
        "dtype_epsilon": eps,
        "raw_asymmetry_max": float(asymmetry.detach().max()),
        "raw_relative_asymmetry_max": float(relative_asymmetry.detach().max()),
        "raw_covariance_scale_max": float(scale.detach().max()),
        "relative_guard_formula": "min(sqrt(eps), 8*M*eps*kappa_sym)",
        "absolute_guard_formula": "8*eps",
        "relative_guard_cap": relative_cap,
        "relative_guard_max": float(relative_limit.detach().max()),
        "effective_guard_max": float(asymmetry_limit.detach().max()),
        "guard_exceedance_ratio_max": float(ratio.detach().max()),
        "condition_number_max": float(condition.detach().max()),
        "minimum_symmetric_eigenvalue": float(minimum_eigenvalue.detach().min()),
        "psd_limit_max": float(psd_limit.detach().max()),
        "exact_output_symmetry": bool(torch.equal(symmetric, symmetric.transpose(-1, -2))),
        "jitter_added": False,
        "diagonal_clamped": False,
        "eigenvalues_clipped": False,
    }
    if bool((asymmetry > asymmetry_limit).any()):
        worst = int(torch.argmax(ratio.reshape(-1)).detach())
        raise InvalidPosteriorError(
            "raw GP covariance is materially asymmetric before structural "
            "materialization "
            f"(worst_batch_flat_index={worst}, "
            f"asymmetry={float(asymmetry.reshape(-1)[worst].detach()):.9g}, "
            f"scale={float(scale.reshape(-1)[worst].detach()):.9g}, "
            f"condition={float(condition.reshape(-1)[worst].detach()):.9g}, "
            f"effective_guard={float(asymmetry_limit.reshape(-1)[worst].detach()):.9g})"
        )
    if bool((minimum_eigenvalue < -psd_limit).any()):
        worst = int(torch.argmin(minimum_eigenvalue.reshape(-1)).detach())
        raise InvalidPosteriorError(
            "structurally materialized covariance is not positive semidefinite "
            f"(worst_batch_flat_index={worst}, "
            f"minimum_eigenvalue={float(minimum_eigenvalue.reshape(-1)[worst].detach()):.9g}, "
            f"effective_limit={float(psd_limit.reshape(-1)[worst].detach()):.9g})"
        )
    return symmetric, diagnostics


@dataclass(frozen=True)
class LatentPosteriorSummary:
    """Per-point latent mean and task covariance in frozen standardized units.

    A single query has mean [M] and covariance [M,M]; H queries have [H,M] and
    [H,M,M]. These blocks describe task correlations at each point, not correlations
    between different query points, and exclude likelihood observation noise.
    Both tensors must be contiguous float32/float64 on the same device. Validation
    retains mean, replaces covariance by its symmetric average, and rejects
    asymmetry or negative eigenvalues outside the selected policy's guard. It adds
    no jitter and does not clip covariance eigenvalues; trace/largest-eigenvalue
    accessors clamp only their scalar summaries at zero.
    """
    mean: torch.Tensor
    covariance: torch.Tensor
    transform_id: str
    model_id: str
    symmetry_tolerance: float = 1e-10
    psd_tolerance: float = 1e-10
    covariance_validation_policy: CovarianceValidationPolicy | str = (
        CovarianceValidationPolicy.ABSOLUTE_V1
    )
    structural_materialization_diagnostics: dict[str, object] | None = field(
        default=None, repr=False, compare=False
    )
    posterior_numerics_diagnostics: dict[str, object] | None = field(
        default=None, repr=False, compare=False
    )
    _eigenvalues: torch.Tensor = field(init=False, repr=False, compare=False)
    _validation_diagnostics: dict[str, object] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.transform_id or not self.model_id:
            raise InvalidPosteriorError("transform_id and model_id must be nonempty")
        if (
            not math.isfinite(self.symmetry_tolerance)
            or self.symmetry_tolerance < 0
            or not math.isfinite(self.psd_tolerance)
            or self.psd_tolerance < 0
        ):
            raise InvalidPosteriorError("covariance tolerances must be finite and nonnegative")
        try:
            policy = CovarianceValidationPolicy(self.covariance_validation_policy)
        except ValueError as exc:
            raise InvalidPosteriorError(
                "unknown covariance validation policy: "
                f"{self.covariance_validation_policy!r}"
            ) from exc
        object.__setattr__(self, "covariance_validation_policy", policy)
        if (
            policy
            is CovarianceValidationPolicy.STRUCTURAL_SYMMETRIC_MATERIALIZATION_V1
            and self.structural_materialization_diagnostics is None
        ):
            raise InvalidPosteriorError(
                "structural covariance policy requires raw materialization diagnostics"
            )
        if not isinstance(self.mean, torch.Tensor) or not isinstance(self.covariance, torch.Tensor):
            raise InvalidPosteriorError("posterior mean and covariance must be Torch tensors")
        if self.mean.ndim not in {1, 2}:
            raise InvalidPosteriorError("posterior mean must have shape [M] or [H,M]")
        response_dim = self.mean.shape[-1]
        expected = (
            (response_dim, response_dim)
            if self.mean.ndim == 1
            else (self.mean.shape[0], response_dim, response_dim)
        )
        if self.covariance.shape != expected:
            raise InvalidPosteriorError(f"posterior covariance must have shape {expected}")
        if self.mean.dtype not in {torch.float32, torch.float64}:
            raise InvalidPosteriorError("posterior moments require float32 or float64")
        if self.covariance.dtype != self.mean.dtype or self.covariance.device != self.mean.device:
            raise InvalidPosteriorError("posterior mean and covariance must share dtype and device")
        if not self.mean.is_contiguous() or not self.covariance.is_contiguous():
            raise InvalidPosteriorError("posterior mean and covariance must be contiguous")
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.covariance).all()):
            raise InvalidPosteriorError("posterior moments must contain only finite values")

        transpose = self.covariance.transpose(-1, -2)
        asymmetry = torch.abs(self.covariance - transpose).amax(dim=(-2, -1))
        symmetric = 0.5 * (self.covariance + transpose)
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        covariance_scale = self.covariance.abs().amax(dim=(-2, -1))
        spectral_scale = eigenvalues.abs().amax(dim=-1)
        eps = torch.finfo(self.covariance.dtype).eps
        response_dim = self.mean.shape[-1]
        if policy is CovarianceValidationPolicy.ABSOLUTE_V1:
            symmetry_atol = self.symmetry_tolerance
            symmetry_rtol = 0.0
            symmetry_limit = torch.full_like(asymmetry, self.symmetry_tolerance)
            psd_atol = self.psd_tolerance
            psd_rtol = 0.0
            psd_limit = torch.full_like(spectral_scale, self.psd_tolerance)
            absolute_epsilon_multiplier = None
            relative_dimension_epsilon_multiplier = None
        else:
            absolute_epsilon_multiplier = 8
            relative_dimension_epsilon_multiplier = 4
            symmetry_atol = absolute_epsilon_multiplier * eps
            symmetry_rtol = (
                relative_dimension_epsilon_multiplier * response_dim * eps
            )
            symmetry_limit = symmetry_atol + symmetry_rtol * covariance_scale
            psd_atol = absolute_epsilon_multiplier * eps
            psd_rtol = relative_dimension_epsilon_multiplier * response_dim * eps
            psd_limit = psd_atol + psd_rtol * spectral_scale
        relative_asymmetry = torch.where(
            covariance_scale > 0,
            asymmetry / covariance_scale,
            torch.where(asymmetry == 0, torch.zeros_like(asymmetry), torch.full_like(asymmetry, torch.inf)),
        )
        symmetry_ratio = torch.where(
            symmetry_limit > 0,
            asymmetry / symmetry_limit,
            torch.where(
                asymmetry == 0,
                torch.zeros_like(asymmetry),
                torch.full_like(asymmetry, torch.inf),
            ),
        )
        minimum_eigenvalue = eigenvalues[..., 0]
        psd_ratio = torch.where(
            psd_limit > 0,
            (-minimum_eigenvalue).clamp_min(0) / psd_limit,
            torch.where(
                minimum_eigenvalue >= 0,
                torch.zeros_like(minimum_eigenvalue),
                torch.full_like(minimum_eigenvalue, torch.inf),
            ),
        )
        diagnostics = {
            "policy_id": policy.value,
            "dtype": str(self.covariance.dtype),
            "response_dim": response_dim,
            "dtype_epsilon": eps,
            "covariance_scale_max": float(covariance_scale.detach().max()),
            "raw_asymmetry_max": float(asymmetry.detach().max()),
            "relative_asymmetry_max": float(relative_asymmetry.detach().max()),
            "symmetry_absolute_epsilon_multiplier": absolute_epsilon_multiplier,
            "symmetry_relative_dimension_epsilon_multiplier": (
                relative_dimension_epsilon_multiplier
            ),
            "symmetry_atol": symmetry_atol,
            "symmetry_rtol": symmetry_rtol,
            "symmetry_limit_max": float(symmetry_limit.detach().max()),
            "symmetry_exceedance_ratio_max": float(
                symmetry_ratio.detach().max()
            ),
            "spectral_scale_max": float(spectral_scale.detach().max()),
            "minimum_symmetrized_eigenvalue": float(
                eigenvalues.detach()[..., 0].min()
            ),
            "psd_absolute_epsilon_multiplier": absolute_epsilon_multiplier,
            "psd_relative_dimension_epsilon_multiplier": (
                relative_dimension_epsilon_multiplier
            ),
            "psd_atol": psd_atol,
            "psd_rtol": psd_rtol,
            "psd_limit_max": float(psd_limit.detach().max()),
            "psd_exceedance_ratio_max": float(psd_ratio.detach().max()),
            "structural_materialization": self.structural_materialization_diagnostics,
            "posterior_numerics": self.posterior_numerics_diagnostics,
        }
        object.__setattr__(self, "_validation_diagnostics", diagnostics)
        if bool((asymmetry > symmetry_limit).any()):
            worst = int(torch.argmax(symmetry_ratio.reshape(-1)).detach())
            worst_asymmetry = asymmetry.reshape(-1)[worst]
            worst_scale = covariance_scale.reshape(-1)[worst]
            worst_relative = relative_asymmetry.reshape(-1)[worst]
            worst_limit = symmetry_limit.reshape(-1)[worst]
            worst_minimum_eigenvalue = minimum_eigenvalue.reshape(-1)[worst]
            raise InvalidPosteriorError(
                "posterior covariance is not symmetric within tolerance "
                f"(worst_batch_flat_index={worst}, "
                f"max_asymmetry={float(worst_asymmetry.detach()):.9g}, "
                f"max_abs_covariance={float(worst_scale.detach()):.9g}, "
                f"relative_asymmetry={float(worst_relative.detach()):.9g}, "
                f"minimum_symmetrized_eigenvalue="
                f"{float(worst_minimum_eigenvalue.detach()):.9g}, "
                f"effective_limit={float(worst_limit.detach()):.9g}, "
                f"exceedance_ratio={float(symmetry_ratio.reshape(-1)[worst].detach()):.9g}, "
                f"policy={policy.value}, "
                f"dtype={self.covariance.dtype})"
            )
        if bool((minimum_eigenvalue < -psd_limit).any()):
            worst = int(torch.argmax(psd_ratio.reshape(-1)).detach())
            raise InvalidPosteriorError(
                "posterior covariance is not positive semidefinite within tolerance "
                f"(worst_batch_flat_index={worst}, "
                f"minimum_eigenvalue="
                f"{float(minimum_eigenvalue.reshape(-1)[worst].detach()):.9g}, "
                f"effective_limit={float(psd_limit.reshape(-1)[worst].detach()):.9g}, "
                f"exceedance_ratio={float(psd_ratio.reshape(-1)[worst].detach()):.9g}, "
                f"policy={policy.value}, "
                f"dtype={self.covariance.dtype})"
            )
        object.__setattr__(self, "covariance", symmetric.contiguous())
        object.__setattr__(self, "_eigenvalues", eigenvalues)

    @property
    def response_dim(self) -> int:
        return self.mean.shape[-1]

    @property
    def is_batched(self) -> bool:
        return self.mean.ndim == 2

    @property
    def largest_eigenvalue(self) -> torch.Tensor:
        return self._eigenvalues[..., -1].clamp_min(0.0)

    @property
    def trace(self) -> torch.Tensor:
        return torch.diagonal(self.covariance, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(0.0)

    @property
    def validation_diagnostics(self) -> dict[str, object]:
        """Return JSON-compatible diagnostics for the raw validation boundary."""

        return dict(self._validation_diagnostics)
