"""Manuscript Step 3(a): residual, uncertainty radius and compatibility slack.

The paper radius is rho^2=beta^2*lambda_max of the latent task covariance.
Compatibility includes its boundary; the log-barrier objective needs a stricter
positive slack. All quantities live in the frozen standardized response space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from mutaco.posterior import LatentPosteriorSummary
from mutaco.policies import BetaPolicy, descriptor_for
from mutaco.response import StandardizedTarget


@dataclass(frozen=True)
class FixedBeta:
    """Positive multiplier beta for rho^2 = beta^2 * lambda_max(Sigma).

    This fixed setting carries no calibrated coverage or sequential confidence
    guarantee; beta=2 is the generic default, not the paper profile's joint quantile.
    """
    value: float = 2.0
    policy_id: str = "fixed_beta"
    guarantee: None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value <= 0:
            raise ValueError("beta must be finite and positive")
        if self.policy_id not in {"fixed_beta", BetaPolicy.FIXED_BETA_V1.value}:
            raise ValueError("unknown fixed-beta policy identifier")

    def provenance(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "value": self.value,
            "guarantee": self.guarantee,
            "interpretation": "confidence_radius_multiplier",
        }


_CHI_SQUARE_QUANTILE_ALGORITHM = "torch_gammainc_bisection_float64_v1"
_CHI_SQUARE_QUANTILE_ITERATIONS = 128


def chi_square_quantile(*, response_dimension: int, coverage: float) -> float:
    """Invert the chi-square CDF for M=response_dimension without SciPy.

    CPU float64 gammainc evaluation, a doubled upper bracket and 128 bisections
    define the recorded numerical algorithm. The returned quantile calibrates a
    Gaussian response-space radius; it is not a sequential coverage theorem.
    """

    if (
        isinstance(response_dimension, bool)
        or not isinstance(response_dimension, int)
        or response_dimension < 1
    ):
        raise ValueError("response_dimension must be a positive integer")
    if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be finite and strictly between zero and one")
    shape = torch.tensor(response_dimension / 2.0, dtype=torch.float64)
    lower, upper = 0.0, float(max(response_dimension, 1))
    while float(
        torch.special.gammainc(
            shape, torch.tensor(upper / 2.0, dtype=torch.float64)
        )
    ) < coverage:
        upper *= 2.0
    for _ in range(_CHI_SQUARE_QUANTILE_ITERATIONS):
        middle = (lower + upper) / 2.0
        probability = float(
            torch.special.gammainc(
                shape, torch.tensor(middle / 2.0, dtype=torch.float64)
            )
        )
        if probability < coverage:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


@dataclass(frozen=True)
class JointChiSquareResponseDimensionBeta:
    """Set beta to sqrt(chi_square_quantile(M, coverage)) for M response tasks.

    M is the response dimension, not the parameter dimension or number of training
    rows. With the largest covariance eigenvalue this sets a spherical radius for
    joint response-space screening. The policy records nominal coverage and its
    quantile calculation, but asserts no frequentist or time-uniform guarantee.
    """
    coverage: float
    response_dimension: int
    chi_square_quantile: float
    value: float
    policy_id: str = BetaPolicy.JOINT_CHI_SQUARE_RESPONSE_DIMENSION_V1.value
    guarantee: None = None

    @classmethod
    def resolve(
        cls, *, coverage: float, response_dimension: int
    ) -> "JointChiSquareResponseDimensionBeta":
        quantile = chi_square_quantile(
            response_dimension=response_dimension, coverage=coverage
        )
        return cls(
            coverage=coverage,
            response_dimension=response_dimension,
            chi_square_quantile=quantile,
            value=math.sqrt(quantile),
        )

    @classmethod
    def from_document(
        cls,
        document: dict[str, object],
        *,
        expected_response_dimension: int,
    ) -> "JointChiSquareResponseDimensionBeta":
        if document.get("id") != BetaPolicy.JOINT_CHI_SQUARE_RESPONSE_DIMENSION_V1.value:
            raise ValueError("unknown joint-response beta policy identifier")
        raw_coverage = document.get("coverage")
        raw_dimension = document.get("response_dimension")
        if isinstance(raw_coverage, bool) or not isinstance(raw_coverage, (int, float)):
            raise ValueError("joint-response beta coverage must be numeric")
        if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, int):
            raise ValueError("joint-response beta response_dimension must be an integer")
        try:
            resolved = cls.resolve(
                coverage=float(raw_coverage),
                response_dimension=raw_dimension,
            )
            recorded_quantile = float(document["chi_square_quantile"])
            recorded_beta = float(document["resolved_beta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("joint-response beta document is incomplete or invalid") from exc
        if resolved.response_dimension != expected_response_dimension:
            raise ValueError("beta response dimension does not match simulator response dimension")
        if not math.isclose(recorded_quantile, resolved.chi_square_quantile, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("recorded chi-square quantile is inconsistent")
        if not math.isclose(recorded_beta, resolved.value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("recorded resolved beta is inconsistent")
        provenance = document.get("quantile_provenance")
        if provenance != resolved.quantile_provenance():
            raise ValueError("chi-square quantile provenance is inconsistent")
        # Accepted documents pin the scientific scalar values.  Recompute only
        # to validate them: replacing them with a platform-local recomputation
        # can change canonical identities (and therefore every derived seed) by
        # one ULP.
        return cls(
            coverage=float(raw_coverage),
            response_dimension=raw_dimension,
            chi_square_quantile=recorded_quantile,
            value=recorded_beta,
        )

    def __post_init__(self) -> None:
        descriptor_for(self.policy_id)
        if self.policy_id != BetaPolicy.JOINT_CHI_SQUARE_RESPONSE_DIMENSION_V1.value:
            raise ValueError("unknown joint-response beta policy identifier")
        if self.guarantee is not None:
            raise ValueError("joint-response beta is heuristic and has no theorem guarantee")
        expected_quantile = chi_square_quantile(
            coverage=self.coverage, response_dimension=self.response_dimension
        )
        if not math.isclose(self.chi_square_quantile, expected_quantile, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("chi_square_quantile is inconsistent")
        if not math.isclose(self.value, math.sqrt(expected_quantile), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("resolved beta is inconsistent")

    def quantile_provenance(self) -> dict[str, object]:
        return {
            "distribution": "chi_square",
            "degrees_of_freedom": self.response_dimension,
            "probability": self.coverage,
            "algorithm": _CHI_SQUARE_QUANTILE_ALGORITHM,
            "iterations": _CHI_SQUARE_QUANTILE_ITERATIONS,
            "computation_dtype": "float64",
        }

    def provenance(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "coverage": self.coverage,
            "response_dimension": self.response_dimension,
            "chi_square_quantile": self.chi_square_quantile,
            "resolved_beta": self.value,
            "quantile_provenance": self.quantile_provenance(),
            "guarantee": self.guarantee,
            "interpretation": "conservative_spherical_joint_response_multiplier",
        }

    def document(self) -> dict[str, object]:
        provenance = self.provenance()
        return {
            "id": provenance["policy_id"],
            "coverage": provenance["coverage"],
            "response_dimension": provenance["response_dimension"],
            "chi_square_quantile": provenance["chi_square_quantile"],
            "resolved_beta": provenance["resolved_beta"],
            "quantile_provenance": provenance["quantile_provenance"],
        }


@dataclass(frozen=True)
class ConfidenceGeometryEvaluation:
    residual: torch.Tensor
    residual_squared: torch.Tensor
    largest_covariance_eigenvalue: torch.Tensor
    covariance_trace: torch.Tensor
    effective_dimension: torch.Tensor
    radius_squared: torch.Tensor
    feasibility_score: torch.Tensor
    margin: torch.Tensor
    feasible: torch.Tensor


def _validate_target(
    posterior: LatentPosteriorSummary,
    target: StandardizedTarget,
) -> None:
    if target.transform_id != posterior.transform_id:
        raise ValueError("posterior and target transform identities do not match")
    if target.values.shape != (posterior.response_dim,):
        raise ValueError("standardized target has the wrong response dimension")
    if target.values.dtype != posterior.mean.dtype or target.values.device != posterior.mean.device:
        raise ValueError("standardized target must match posterior dtype and device")


def evaluate_confidence_geometry(
    posterior: LatentPosteriorSummary,
    target: StandardizedTarget,
    *,
    beta: FixedBeta = FixedBeta(),
    radius_policy: str = "operator_norm_radius_v1",
) -> ConfidenceGeometryEvaluation:
    """Return residual r, squared radius rho^2 and manuscript slack Delta.

    For posterior means [M] or [C,M], residuals preserve that shape and scalar
    fields have shape [] or [C]. Target [M] must share transform, dtype/device
    and component order. The paper policy uses lambda_max; trace radius is a
    separately named diagnostic option.

    feasibility_score=||r||^2-rho^2, so feasible means score<=0. For positive
    radius, margin=(rho^2-||r||^2)/rho^2 is Delta. At zero radius, define margin
    as zero for exact target agreement and -inf otherwise: neither enters a
    positive strict interior, although exact agreement is still compatible.
    """

    _validate_target(posterior, target)
    residual = posterior.mean - target.values
    residual_squared = residual.square().sum(dim=-1)
    largest = posterior.largest_eigenvalue
    trace = posterior.trace
    if radius_policy == "operator_norm_radius_v1":
        radius_squared = (beta.value**2) * largest
    elif radius_policy == "trace_effective_dimension_radius_diagnostic_v1":
        radius_squared = (beta.value**2) * trace
    else:
        raise ValueError(f"unknown confidence-radius policy: {radius_policy!r}")
    effective_dimension = torch.where(
        largest > 0, trace / largest, torch.zeros_like(largest)
    )
    score = residual_squared - radius_squared
    feasible = score <= 0

    positive_radius = radius_squared > 0
    # Avoid division by zero before selecting the explicitly defined zero-radius
    # margin below; a compatible boundary point is not a barrier-interior point.
    arithmetic_denominator = torch.where(
        positive_radius,
        radius_squared,
        torch.ones_like(radius_squared),
    )
    regular_margin = (radius_squared - residual_squared) / arithmetic_denominator
    zero_radius_margin = torch.where(
        residual_squared == 0,
        torch.zeros_like(residual_squared),
        torch.full_like(residual_squared, -torch.inf),
    )
    margin = torch.where(positive_radius, regular_margin, zero_radius_margin)
    return ConfidenceGeometryEvaluation(
        residual=residual,
        residual_squared=residual_squared,
        largest_covariance_eigenvalue=largest,
        covariance_trace=trace,
        effective_dimension=effective_dimension,
        radius_squared=radius_squared,
        feasibility_score=score,
        margin=margin,
        feasible=feasible,
    )
