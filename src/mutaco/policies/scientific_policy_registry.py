"""Single source of truth for Prompt 3 public mathematical policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .surface import surface_record


class PolicyClassification(str, Enum):
    DEFAULT = "D default"
    APPROVED_OPTIONAL = "approved optional"
    DIAGNOSTIC_ONLY = "diagnostic-only"


class OptimizationDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    CONFIGURE = "configure"


class ExplorationPolicy(str, Enum):
    OPERATOR_NORM_UNCERTAINTY = "operator_norm_uncertainty"
    TRACE_MARGINAL_VARIANCE = "trace_marginal_variance"


class BetaPolicy(str, Enum):
    FIXED_BETA_V1 = "fixed_beta_v1"
    JOINT_CHI_SQUARE_RESPONSE_DIMENSION_V1 = (
        "joint_chi_square_response_dimension_v1"
    )


class RecommendationPolicy(str, Enum):
    FINAL_POSTERIOR_STANDARDIZED_L1 = "final_posterior_standardized_l1"
    EVALUATED_POSTERIOR_EXPECTED_L2 = "evaluated_posterior_expected_l2"
    EVALUATED_POSTERIOR_MEAN_PHYSICAL_L2 = "evaluated_posterior_mean_physical_l2"
    CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L2 = "continuous_posterior_mean_physical_l2"
    CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L1_MAE = (
        "continuous_posterior_mean_physical_l1_mae"
    )
    EVALUATED_BEST_OBSERVED_PHYSICAL_MAE = (
        "evaluated_best_observed_physical_mae_v1"
    )


@dataclass(frozen=True)
class PolicyDescriptor:
    family: str
    identifier: str
    classification: PolicyClassification
    direction: OptimizationDirection
    formula: str
    required_inputs: str
    simulator_calls: int


EXPLORATION_POLICIES: dict[ExplorationPolicy, PolicyDescriptor] = {
    ExplorationPolicy.OPERATOR_NORM_UNCERTAINTY: PolicyDescriptor(
        family="exploration",
        identifier=ExplorationPolicy.OPERATOR_NORM_UNCERTAINTY.value,
        classification=PolicyClassification.DEFAULT,
        direction=OptimizationDirection.MAXIMIZE,
        formula="lambda_max(Sigma_latent_std)",
        required_inputs="latent standardized posterior covariance",
        simulator_calls=0,
    ),
    ExplorationPolicy.TRACE_MARGINAL_VARIANCE: PolicyDescriptor(
        family="exploration",
        identifier=ExplorationPolicy.TRACE_MARGINAL_VARIANCE.value,
        classification=PolicyClassification.DIAGNOSTIC_ONLY,
        direction=OptimizationDirection.MAXIMIZE,
        formula="trace(Sigma_latent_std)",
        required_inputs="latent standardized posterior covariance",
        simulator_calls=0,
    ),
}

BETA_POLICIES: dict[BetaPolicy, PolicyDescriptor] = {
    BetaPolicy.FIXED_BETA_V1: PolicyDescriptor(
        family="beta",
        identifier=BetaPolicy.FIXED_BETA_V1.value,
        classification=PolicyClassification.DEFAULT,
        direction=OptimizationDirection.CONFIGURE,
        formula="beta = configured finite positive scalar (default 2.0)",
        required_inputs="finite positive beta",
        simulator_calls=0,
    ),
    BetaPolicy.JOINT_CHI_SQUARE_RESPONSE_DIMENSION_V1: PolicyDescriptor(
        family="beta",
        identifier=BetaPolicy.JOINT_CHI_SQUARE_RESPONSE_DIMENSION_V1.value,
        classification=PolicyClassification.APPROVED_OPTIONAL,
        direction=OptimizationDirection.CONFIGURE,
        formula="beta = sqrt(chi_square_quantile(M, coverage))",
        required_inputs="coverage in (0,1); declared response dimension M",
        simulator_calls=0,
    ),
}

RECOMMENDATION_POLICIES: dict[RecommendationPolicy, PolicyDescriptor] = {
    RecommendationPolicy.FINAL_POSTERIOR_STANDARDIZED_L1: PolicyDescriptor(
        family="recommendation",
        identifier=RecommendationPolicy.FINAL_POSTERIOR_STANDARDIZED_L1.value,
        classification=PolicyClassification.DEFAULT,
        direction=OptimizationDirection.MINIMIZE,
        formula="sum_j |mu_final_std,j(theta_i) - target_std,j|",
        required_inputs="evaluated theta; final latent posterior mean; standardized target",
        simulator_calls=0,
    ),
    RecommendationPolicy.EVALUATED_POSTERIOR_EXPECTED_L2: PolicyDescriptor(
        family="recommendation",
        identifier=RecommendationPolicy.EVALUATED_POSTERIOR_EXPECTED_L2.value,
        classification=PolicyClassification.APPROVED_OPTIONAL,
        direction=OptimizationDirection.MINIMIZE,
        formula="||mu_final_std(theta_i)-target_std||_2^2 + trace(Sigma_final_latent_std(theta_i))",
        required_inputs="evaluated theta; final latent posterior mean/covariance; standardized target",
        simulator_calls=0,
    ),
    RecommendationPolicy.EVALUATED_POSTERIOR_MEAN_PHYSICAL_L2: PolicyDescriptor(
        family="recommendation",
        identifier=RecommendationPolicy.EVALUATED_POSTERIOR_MEAN_PHYSICAL_L2.value,
        classification=PolicyClassification.APPROVED_OPTIONAL,
        direction=OptimizationDirection.MINIMIZE,
        formula="||mu_final_phys(theta_i)-target_phys||_2^2",
        required_inputs="evaluated theta; final latent posterior mean; frozen transform; physical target",
        simulator_calls=0,
    ),
    RecommendationPolicy.CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L2: PolicyDescriptor(
        family="recommendation",
        identifier=RecommendationPolicy.CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L2.value,
        classification=PolicyClassification.APPROVED_OPTIONAL,
        direction=OptimizationDirection.MINIMIZE,
        formula="min over bounded theta of ||mu_final_phys(theta)-target_phys||_2^2",
        required_inputs="bounded parameter specification; final latent posterior mean; frozen transform; physical target; coordinate seed",
        simulator_calls=0,
    ),
    RecommendationPolicy.CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L1_MAE: PolicyDescriptor(
        family="recommendation",
        identifier=RecommendationPolicy.CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L1_MAE.value,
        classification=PolicyClassification.APPROVED_OPTIONAL,
        direction=OptimizationDirection.MINIMIZE,
        formula="min over bounded theta of mean_m |mu_final_phys,m(theta)-target_phys,m|",
        required_inputs="bounded parameter specification; final latent posterior mean; frozen transform; physical target; coordinate seed",
        simulator_calls=0,
    ),
    RecommendationPolicy.EVALUATED_BEST_OBSERVED_PHYSICAL_MAE: PolicyDescriptor(
        family="recommendation",
        identifier=RecommendationPolicy.EVALUATED_BEST_OBSERVED_PHYSICAL_MAE.value,
        classification=PolicyClassification.APPROVED_OPTIONAL,
        direction=OptimizationDirection.MINIMIZE,
        formula="min over evaluated i of mean_m |y_observed_i,m-target_phys,m|",
        required_inputs="evaluated theta; observed physical responses; physical target",
        simulator_calls=0,
    ),
}


@dataclass(frozen=True)
class MathematicsPolicyConfig:
    exploration: ExplorationPolicy = ExplorationPolicy.OPERATOR_NORM_UNCERTAINTY
    recommendation: RecommendationPolicy = RecommendationPolicy.FINAL_POSTERIOR_STANDARDIZED_L1

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "exploration", ExplorationPolicy(self.exploration))
        except ValueError as exc:
            raise ValueError(f"unknown exploration policy: {self.exploration!r}") from exc
        try:
            object.__setattr__(self, "recommendation", RecommendationPolicy(self.recommendation))
        except ValueError as exc:
            raise ValueError(f"unknown recommendation policy: {self.recommendation!r}") from exc

    def provenance(self) -> dict[str, str]:
        return {
            "exploration": self.exploration.value,
            "recommendation": self.recommendation.value,
        }


def descriptor_for(identifier: str) -> PolicyDescriptor:
    for registry in (EXPLORATION_POLICIES, BETA_POLICIES, RECOMMENDATION_POLICIES):
        for descriptor in registry.values():
            if descriptor.identifier == identifier:
                return descriptor
    raise ValueError(f"unknown Prompt 3 policy identifier: {identifier!r}")


def render_policy_support_markdown() -> str:
    descriptors = [
        *EXPLORATION_POLICIES.values(),
        *BETA_POLICIES.values(),
        *RECOMMENDATION_POLICIES.values(),
    ]
    lines = [
        "# Prompt 3 policy support",
        "",
        "This table is generated from `mutaco.policies`; do not edit it independently.",
        "",
        "| Family | Identifier | Classification | Surface | Direction | Formula | Required inputs | Simulator calls |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for item in descriptors:
        lines.append(
            f"| {item.family} | `{item.identifier}` | {item.classification.value} | "
            f"`{surface_record(item.identifier).surface.value}` | "
            f"{item.direction.value} | `{item.formula}` | {item.required_inputs} | "
            f"{item.simulator_calls} |"
        )
    return "\n".join(lines) + "\n"
