"""Machine-readable ownership of supported, compatible, and diagnostic policy IDs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class PolicySurface(str, Enum):
    """The configuration surface on which an identifier is intentionally exposed."""

    ACTIVE_CORE = "active_core"
    ACTIVE_SPECIALIZED_CORE = "active_specialized_core"
    COMPATIBILITY = "compatibility"
    DIAGNOSTIC = "diagnostic"
    INTERNAL = "internal"


@dataclass(frozen=True)
class PolicySurfaceRecord:
    identifier: str
    family: str
    surface: PolicySurface
    ordinary_graph_configuration: bool
    retained_for_existing_evidence: bool
    owner: str

    def document(self) -> dict[str, object]:
        document = asdict(self)
        document["surface"] = self.surface.value
        return document


def _record(
    identifier: str,
    family: str,
    surface: PolicySurface,
    *,
    ordinary: bool = False,
    evidence: bool = False,
    owner: str = "mutaco",
) -> PolicySurfaceRecord:
    return PolicySurfaceRecord(identifier, family, surface, ordinary, evidence, owner)


# This inventory classifies selectable policies and identity-bearing mechanisms.
# It does not rename identifiers or participate in scientific identity creation.
POLICY_SURFACE: tuple[PolicySurfaceRecord, ...] = (
    _record("scrambled_sobol", "initial_and_candidate_design", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("caller_supplied", "initial_design", PolicySurface.ACTIVE_CORE),
    _record("task_diagonal_gaussian", "likelihood", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("shared_isotropic_gaussian", "likelihood", PolicySurface.COMPATIBILITY),
    _record("cold_only", "gp_fit", PolicySurface.ACTIVE_CORE),
    _record("graph_m20_sequential_recovery_v1", "gp_fit", PolicySurface.ACTIVE_SPECIALIZED_CORE, ordinary=True),
    _record("float32_fit_float64_covariance", "posterior_numerics", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("operator_norm_radius_v1", "confidence_geometry", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("operator_norm_uncertainty", "exploration", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("fixed_beta_v1", "beta", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("joint_chi_square_response_dimension_v1", "beta", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("posterior_uncertainty_v1", "infeasible_fallback", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("reject_exact_duplicate_v1", "duplicate", PolicySurface.ACTIVE_CORE),
    _record("best_ranked_unevaluated_raw_v1", "duplicate", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("final_posterior_standardized_l1", "recommendation", PolicySurface.ACTIVE_CORE),
    _record("continuous_posterior_mean_physical_l2", "recommendation", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("continuous_posterior_mean_physical_l1_mae", "recommendation", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("evaluated_best_observed_physical_mae_v1", "recommendation", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("composite_mc_nei_physical_mae_v1", "acquisition", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("best_ranked_finite_unevaluated_raw_v1", "acquisition_recovery", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("evaluated_posterior_expected_l2", "recommendation", PolicySurface.COMPATIBILITY),
    _record("evaluated_posterior_mean_physical_l2", "recommendation", PolicySurface.COMPATIBILITY),
    _record("independent", "evaluation_randomness", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("fresh_held_out", "assessment", PolicySurface.ACTIVE_CORE, ordinary=True),
    _record("high_compute", "proposal_effort_alias", PolicySurface.COMPATIBILITY, owner="graph_inversion"),
    _record("iid_uniform", "initial_and_candidate_design", PolicySurface.DIAGNOSTIC),
    _record("trace_marginal_variance", "exploration", PolicySurface.DIAGNOSTIC),
    _record("native_float32", "posterior_numerics", PolicySurface.DIAGNOSTIC, evidence=True),
    _record("full_float64_diagnostic", "posterior_numerics", PolicySurface.DIAGNOSTIC, evidence=True),
    _record("trace_effective_dimension_radius_diagnostic_v1", "confidence_geometry", PolicySurface.DIAGNOSTIC, evidence=True),
    _record("minimum_normalized_target_miss_diagnostic_v1", "infeasible_fallback", PolicySurface.DIAGNOSTIC, evidence=True),
    _record("paired_coordinate_diagnostic_v1", "evaluation_randomness", PolicySurface.DIAGNOSTIC, evidence=True),
    _record("fixed_full_stream_crn_diagnostic_v1", "evaluation_randomness", PolicySurface.DIAGNOSTIC, evidence=True),
    _record("same_target_realization_assessment", "assessment", PolicySurface.DIAGNOSTIC, evidence=True, owner="graph_inversion"),
    _record("continuous_expected_physical_l2_diagnostic", "recommendation", PolicySurface.DIAGNOSTIC, evidence=True, owner="graph_inversion"),
    _record("legacy_full_stream_crn_diagnostic", "search_randomness", PolicySurface.DIAGNOSTIC, evidence=True, owner="graph_inversion"),
    _record("legacy_scale_search_effort_diagnostic", "proposal_effort", PolicySurface.DIAGNOSTIC, evidence=True, owner="graph_inversion"),
    _record("continuous_posterior_mean_physical_l1_mae_diagnostic_v1", "recommendation_alias", PolicySurface.DIAGNOSTIC, evidence=True, owner="graph_policy_canary"),
    _record("gp_recovery_stages", "mechanism", PolicySurface.INTERNAL),
    _record("covariance_promotion_and_validation", "mechanism", PolicySurface.INTERNAL),
    _record("duplicate_detection_and_raw_ranking", "mechanism", PolicySurface.INTERNAL),
    _record("continuous_terminal_optimizer", "mechanism", PolicySurface.INTERNAL),
    _record("optimizer_numerical_controls", "mechanism", PolicySurface.INTERNAL),
    _record("execution_and_thread_controls", "mechanism", PolicySurface.INTERNAL),
)


def surface_record(identifier: str) -> PolicySurfaceRecord:
    matches = tuple(item for item in POLICY_SURFACE if item.identifier == identifier)
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous policy-surface identifier: {identifier!r}")
    return matches[0]


def policy_surface_document() -> dict[str, object]:
    return {
        "schema_version": "mutaco-policy-surface-v1",
        "classifications": [item.value for item in PolicySurface],
        "records": [item.document() for item in POLICY_SURFACE],
    }


def render_policy_surface_markdown() -> str:
    lines = [
        "# MUTACO policy surface and compatibility matrix",
        "",
        "This table is generated from `mutaco.policies.surface`; do not edit it independently.",
        "The surface label governs new configuration. `Evidence shim` marks identifiers required by accepted diagnostic or result evidence.",
        "",
        "| Identifier | Family | Surface | Ordinary graph | Evidence shim | Owner |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in POLICY_SURFACE:
        lines.append(
            f"| `{item.identifier}` | {item.family} | `{item.surface.value}` | "
            f"{'yes' if item.ordinary_graph_configuration else 'no'} | "
            f"{'yes' if item.retained_for_existing_evidence else 'no'} | `{item.owner}` |"
        )
    return "\n".join(lines) + "\n"
