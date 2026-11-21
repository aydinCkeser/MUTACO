"""Cross-experiment vocabulary, cohort, and randomness ownership contract.

This module is computation-free.  It validates scientific declarations and
describes random-stream ownership; it never derives a replacement seed for an
already frozen experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mutaco import stable_hash

from .validation import PaperContractError


CONTRACT_SCHEMA = "mutaco-cross-experiment-scientific-contract-v1"
CONTRACT_IDENTITY_DOMAIN = "cross-experiment-scientific-contract-v1"
PAPER120_COHORT_PATH = "data/cohorts/paper120_v3/manifest.json"
PAPER120_COHORT_SHA256 = (
    "359fc0bb59fc9a8a52f5da684e7a7eb352466bdc0d0a54fb8efa94131ce423bb"
)

ERROR_QUANTITIES = {
    "observed_search_error": {
        "symbol": "E_obs",
        "formula": "min_i<=B L(y_i,y_star)",
        "response_source": "noisy_in_search_simulator_observation",
        "selection_domain": "evaluated_designs_only",
    },
    "raw_terminal_surrogate_error": {
        "symbol": "E_raw_GP",
        "formula": "L(mu_B(theta_raw),y_star)",
        "response_source": "final_surrogate_posterior_mean",
        "selection_domain": "declared_raw_terminal_screen",
    },
    "optimized_terminal_surrogate_error": {
        "symbol": "E_term_GP",
        "formula": "L(mu_B(theta_hat),y_star)",
        "response_source": "final_surrogate_posterior_mean",
        "selection_domain": "declared_terminal_optimizer",
    },
    "heldout_assessment_error": {
        "symbol": "E_heldout",
        "formula": "L(y_assess(theta_hat),y_star)",
        "response_source": "fresh_heldout_simulator_response",
        "selection_domain": "frozen_recommendation_only",
    },
}

RECOMMENDATION_PROVENANCE = (
    "best_observed",
    "raw_terminal_screen",
    "historical_one_step_terminal",
    "corrected_refined_terminal",
    "heldout_assessment",
)

EVIDENCE_STATUSES = (
    "valid_current_policy_evidence",
    "valid_historical_policy_evidence",
    "protocol_noncompliant_for_current_claim",
    "qualification_evidence",
    "superseded_evidence",
)


@dataclass(frozen=True)
class RandomnessRole:
    role: str
    method_ownership: str
    index_required: bool
    pairing_scope: str

    def __post_init__(self) -> None:
        if self.method_ownership not in {"required", "forbidden"}:
            raise PaperContractError("randomness method ownership is unsupported")
        if not self.role or not self.pairing_scope:
            raise PaperContractError("randomness role fields must be nonempty")

    def document(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "method_ownership": self.method_ownership,
            "index_required": self.index_required,
            "pairing_scope": self.pairing_scope,
        }


RANDOMNESS_ROLES = {
    item.role: item
    for item in (
        RandomnessRole(
            "target_construction", "forbidden", True, "cohort_target_only"
        ),
        RandomnessRole(
            "shared_initial_design", "forbidden", True, "target_budget_comparison"
        ),
        RandomnessRole(
            "search_evaluation", "required", True, "method_specific"
        ),
        RandomnessRole(
            "acquisition_or_training", "required", True, "method_specific"
        ),
        RandomnessRole(
            "terminal_recommendation", "required", True, "method_specific"
        ),
        RandomnessRole(
            "paired_heldout_assessment",
            "forbidden",
            True,
            "target_budget_comparison",
        ),
    )
}


def contract_body() -> dict[str, Any]:
    """Return the canonical machine-readable contract body."""

    return {
        "schema_version": CONTRACT_SCHEMA,
        "scope": "paper_cross_experiment_scientific_semantics",
        "canonical_cohort": {
            "cohort_id": "paper120_v3",
            "manifest": PAPER120_COHORT_PATH,
            "manifest_sha256": PAPER120_COHORT_SHA256,
            "target_count": 120,
            "application": (
                "required_for_primary_population_repaired_sbi_and_"
                "paper120_mechanism_ablation; explicit_qualification_cohorts_exempt"
            ),
        },
        "error_quantities": ERROR_QUANTITIES,
        "recommendation_provenance": list(RECOMMENDATION_PROVENANCE),
        "evidence_statuses": list(EVIDENCE_STATUSES),
        "randomness_roles": {
            name: role.document() for name, role in RANDOMNESS_ROLES.items()
        },
        "randomness_invariants": {
            "one_global_seed_forbidden": True,
            "named_coordinate_addressing_required": True,
            "search_is_method_addressed": True,
            "paired_assessment_omits_method": True,
            "assessment_frozen_after_all_recommendations": True,
            "assessment_cannot_influence_search_or_recommendation": True,
            "initial_design_sharing_must_be_declared": True,
            "cross_budget_seed_reuse_forbidden_unless_explicitly_declared": True,
            "historical_seed_identities_immutable": True,
        },
        "simulator_backbone_projection": {
            "schema_version": "paper-simulator-scientific-backbone-v1",
            "included_fields": ["dtype", "simulator"],
            "excluded_fields": [
                "budget",
                "method_profile",
                "campaign",
                "runtime_limits",
                "artifact_policy",
            ],
            "reason": (
                "legacy simulator_configuration_identity fields hash complete "
                "budget-specific method configurations, not only simulator physics"
            ),
        },
        "study_families": {
            "scalar_primary": {
                "protocol_id": "mutaco-scalar-qlognei-sobol-primary-v1",
                "cohort_policy": "paper120_v3_required",
                "budgets": [40, 100],
                "budgets_are_independent_campaigns": True,
                "shared_initial_design": True,
                "paired_heldout_assessment": True,
            },
            "repaired_sbi_comparison": {
                "cohort_policy": "paper120_v3_required",
                "budgets": [40, 100],
                "historical_search_randomness": "immutable_if_reused",
                "new_paired_assessment_required": True,
            },
            "mechanism_ablation": {
                "cohort_policy": "paper120_v3_required",
                "randomness_policy": "paired_where_mechanisms_are_alignable",
                "declaration_required_before_execution": True,
            },
            "numerical_qualification": {
                "cohort_policy": "explicit_declared_cohort",
                "paper120_v3_required": False,
                "cannot_be_pooled_with_primary_population": True,
            },
        },
    }


def contract_document() -> dict[str, Any]:
    body = contract_body()
    return {
        **body,
        "contract_identity": stable_hash(CONTRACT_IDENTITY_DOMAIN, body),
    }


def load_cross_experiment_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    """Require exact canonical regeneration of the versioned contract."""

    expected = contract_document()
    if dict(document) != expected:
        raise PaperContractError(
            "cross-experiment scientific contract is not canonical"
        )
    return expected


def operation_owner(
    *,
    study_identity: str,
    target_identity: str,
    budget: int | None,
    role: str,
    method: str | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    """Describe semantic random-stream ownership without deriving a seed."""

    if not study_identity or not target_identity:
        raise PaperContractError("randomness ownership identities must be nonempty")
    if role not in RANDOMNESS_ROLES:
        raise PaperContractError(f"unsupported randomness role {role!r}")
    policy = RANDOMNESS_ROLES[role]
    if policy.method_ownership == "required" and not method:
        raise PaperContractError(f"randomness role {role!r} requires a method")
    if policy.method_ownership == "forbidden" and method is not None:
        raise PaperContractError(f"randomness role {role!r} must omit method")
    if policy.index_required and (
        isinstance(index, bool) or not isinstance(index, int) or index < 0
    ):
        raise PaperContractError(f"randomness role {role!r} requires a nonnegative index")
    if role != "target_construction" and budget not in {40, 100}:
        raise PaperContractError(f"randomness role {role!r} requires budget 40 or 100")
    if role == "target_construction" and budget is not None:
        raise PaperContractError("target construction ownership must omit budget")
    body = {
        "study_identity": study_identity,
        "target_identity": target_identity,
        "budget": budget,
        "role": role,
        "method": method,
        "index": index,
        "pairing_scope": policy.pairing_scope,
    }
    return {
        **body,
        "ownership_identity": stable_hash("scientific-randomness-owner-v1", body),
    }


def _manifest_target_identities(cohort_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    records = cohort_manifest.get("records")
    if not isinstance(records, Mapping):
        raise PaperContractError("cohort manifest records are missing")
    try:
        positions = sorted(int(key) for key in records)
    except (TypeError, ValueError) as exc:
        raise PaperContractError("cohort positions must be integer strings") from exc
    if positions != list(range(120)):
        raise PaperContractError("paper120_v3 cohort positions are incomplete")
    result = []
    for position in positions:
        row = records.get(str(position))
        if not isinstance(row, Mapping) or not row.get("target_record_id"):
            raise PaperContractError("cohort target record identity is missing")
        result.append(str(row["target_record_id"]))
    if len(set(result)) != 120:
        raise PaperContractError("cohort target identities are duplicated")
    return tuple(result)


def simulator_scientific_backbone(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a legacy complete method configuration onto simulator science."""

    simulator = configuration.get("simulator")
    dtype = configuration.get("dtype")
    if not isinstance(simulator, Mapping) or not isinstance(dtype, str) or not dtype:
        raise PaperContractError("simulator scientific backbone is incomplete")
    body = {
        "schema_version": "paper-simulator-scientific-backbone-v1",
        "dtype": dtype,
        "simulator": dict(simulator),
    }
    return {
        **body,
        "backbone_identity": stable_hash(
            "paper-simulator-scientific-backbone-v1", body
        ),
    }


def validate_primary_declaration_contract(
    declaration: Any,
    *,
    cohort_manifest: Mapping[str, Any],
    cohort_manifest_sha256: str,
    simulator_configuration: Mapping[str, Any],
    simulator_configuration_sha256: str,
) -> dict[str, Any]:
    """Validate a primary population declaration against the common backbone."""

    if cohort_manifest_sha256 != PAPER120_COHORT_SHA256:
        raise PaperContractError("primary declaration uses the wrong cohort manifest")
    expected_targets = _manifest_target_identities(cohort_manifest)
    if tuple(declaration.target_record_identities) != expected_targets:
        raise PaperContractError("primary declaration target order differs from paper120_v3")
    if declaration.target_ledger_hash != PAPER120_COHORT_SHA256:
        raise PaperContractError("primary declaration target ledger hash mismatch")
    comparison = declaration.comparison
    if simulator_configuration_sha256 != comparison.simulator_configuration_identity:
        raise PaperContractError(
            "primary complete simulator/method configuration hash mismatch"
        )
    simulator_backbone = simulator_scientific_backbone(simulator_configuration)
    if comparison.assessment_policy != "paired_held_out_assessment_v1":
        raise PaperContractError("primary assessment policy is not paired")
    if comparison.metric_schema != "primary_physical_response_metrics_v1":
        raise PaperContractError("primary physical metric schema mismatch")
    seeds = (
        comparison.design_campaign_seed,
        comparison.inference_campaign_seed,
        comparison.terminal_campaign_seed,
        comparison.assessment_campaign_seed,
    )
    if len(set(seeds)) != len(seeds):
        raise PaperContractError("primary randomness roles require distinct root seeds")
    return {
        "contract_identity": contract_document()["contract_identity"],
        "study_identity": declaration.study_identity,
        "campaign_id": declaration.campaign_id,
        "budget": declaration.budget,
        "cohort": "paper120_v3",
        "target_count": len(expected_targets),
        "simulator_backbone_identity": simulator_backbone["backbone_identity"],
        "assessment_pairing": "method_independent_within_target_budget",
        "search_randomness": "method_addressed",
        "status": "valid",
    }


def validate_primary_budget_pair(
    declarations: Sequence[Any],
    *,
    cohort_manifest: Mapping[str, Any],
    cohort_manifest_sha256: str,
    simulator_configurations: Mapping[int, Mapping[str, Any]],
    simulator_configuration_sha256: Mapping[int, str],
) -> dict[str, Any]:
    """Require independent B40/B100 campaigns on one common scientific backbone."""

    items = tuple(declarations)
    if len(items) != 2 or {item.budget for item in items} != {40, 100}:
        raise PaperContractError("primary budget pair must contain exactly B40 and B100")
    validations = tuple(
        validate_primary_declaration_contract(
            item,
            cohort_manifest=cohort_manifest,
            cohort_manifest_sha256=cohort_manifest_sha256,
            simulator_configuration=simulator_configurations[item.budget],
            simulator_configuration_sha256=simulator_configuration_sha256[
                item.budget
            ],
        )
        for item in items
    )
    left, right = items
    if left.study_identity != right.study_identity:
        raise PaperContractError("primary budgets must share one study identity")
    if left.campaign_id == right.campaign_id:
        raise PaperContractError("primary budgets require independent campaign identities")
    if tuple(item.method for item in left.methods) != tuple(
        item.method for item in right.methods
    ):
        raise PaperContractError("primary budgets use different method families")
    backbone_fields = (
        "simulator_definition_identity",
        "assessment_fidelity",
        "assessment_policy",
        "metric_schema",
    )
    if any(
        getattr(left.comparison, field) != getattr(right.comparison, field)
        for field in backbone_fields
    ):
        raise PaperContractError("primary budgets differ on the scientific backbone")
    if (
        validations[0]["simulator_backbone_identity"]
        != validations[1]["simulator_backbone_identity"]
    ):
        raise PaperContractError("primary budgets use different simulator physics")
    left_seeds = {
        left.comparison.design_campaign_seed,
        left.comparison.inference_campaign_seed,
        left.comparison.terminal_campaign_seed,
        left.comparison.assessment_campaign_seed,
    }
    right_seeds = {
        right.comparison.design_campaign_seed,
        right.comparison.inference_campaign_seed,
        right.comparison.terminal_campaign_seed,
        right.comparison.assessment_campaign_seed,
    }
    if left_seeds & right_seeds:
        raise PaperContractError("independent primary budgets reuse a root seed")
    return {
        "contract_identity": contract_document()["contract_identity"],
        "study_identity": left.study_identity,
        "budgets": [40, 100],
        "campaigns_independent": True,
        "common_cohort": "paper120_v3",
        "status": "valid",
        "declarations": list(validations),
    }
