"""Small graph-paper surface selecting core recommendation policies."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mutaco import (
    FixedBeta,
    FrozenResponseTransform,
    JointChiSquareResponseDimensionBeta,
    ParameterSpec,
    SeedDeriver,
)
from mutaco.identity import ScientificIdentity
from mutaco.recommendation import (
    CONTINUOUS_PHYSICAL_MAE,
    CONTINUOUS_PHYSICAL_MAE_SETTINGS,
    CONTINUOUS_PHYSICAL_L2,
    CONTINUOUS_PHYSICAL_L2_SETTINGS,
    EVALUATED_PHYSICAL_L2,
    PhysicalL2RecommendationError,
    PhysicalL2RecommendationResult,
    PhysicalMAERecommendationResult,
    physical_l2_scores,
    recommend_continuous_physical_l2,
    recommend_continuous_physical_mae,
    recommend_evaluated_physical_l2,
)


CANONICAL = "canonical"
HIGH_COMPUTE = "high_compute"
EXPLICIT_CONTROLS = "explicit_controls_v1"
INDEPENDENT = "independent"
FRESH_HELD_OUT = "fresh_held_out"
PROPOSAL_PROFILES: dict[str, dict[str, Any]] = {
    CANONICAL: {"raw_samples": 64, "restarts": 4, "max_iterations": 8, "learning_rate": 0.05},
}
COMPATIBILITY_PROPOSAL_PROFILES: dict[str, dict[str, Any]] = {
    HIGH_COMPUTE: {"raw_samples": 256, "restarts": 10, "max_iterations": 32, "learning_rate": 0.05},
}
TERMINAL_POLICIES = (
    CONTINUOUS_PHYSICAL_L2,
    CONTINUOUS_PHYSICAL_MAE,
)
COMPATIBILITY_TERMINAL_POLICIES = (EVALUATED_PHYSICAL_L2,)
CONTINUOUS_SETTINGS = CONTINUOUS_PHYSICAL_L2_SETTINGS
GRAPH_RESPONSE_DIMENSION = 20
CANONICAL_PROFILE = "graph_canonical_b40_v1"
PAPER120_PROFILE = "graph_paper_joint95_mae_initial16_v1"
PAPER120_NO_STANDARDIZATION_PROFILE = (
    "graph_paper_joint95_mae_initial16_no_standardization_v1"
)
STANDARDIZE_INITIAL_DESIGN = "initial_design_zscore_v1"
IDENTITY_PHYSICAL = "identity_physical_v1"
TerminalRecommendationError = PhysicalL2RecommendationError
GraphTerminalRecommendation = PhysicalL2RecommendationResult | PhysicalMAERecommendationResult

_FORBIDDEN_ORDINARY_VALUES = {
    "final_posterior_standardized_l1",
    "evaluated_posterior_expected_l2",
    "continuous_expected_physical_l2_diagnostic",
    "fixed_full_stream_crn_diagnostic_v1",
    "same_target_realization_assessment",
    "legacy_full_stream_crn_diagnostic",
}
_FORBIDDEN_ORDINARY_KEYS = {
    "recommendation_policy", "assessment_policy", "search_randomness", "evaluation_seed_policy"
}
ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIRECTORY = ROOT / "experiments" / "graph_inversion" / "configs" / "profiles"


def load_graph_policy_profile(profile_id: str) -> dict[str, Any]:
    """Load and validate one data-defined graph policy profile.

    A profile selects core options; it never selects a separate implementation.
    The sequential allocation is derived from budget minus initial points.
    """

    if not profile_id or Path(profile_id).name != profile_id:
        raise ValueError("graph profile_id must be a simple nonempty identifier")
    path = PROFILE_DIRECTORY / f"{profile_id}.json"
    if not path.is_file():
        raise ValueError(f"unknown graph policy profile: {profile_id!r}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("profile_id") != profile_id:
        raise ValueError("graph policy profile identity mismatch")
    budget = document.get("simulation_budget")
    initial = document.get("initial_points")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or isinstance(initial, bool)
        or not isinstance(initial, int)
        or initial < 2
        or initial > budget
    ):
        raise ValueError("graph profile requires a valid budget and initial allocation")
    if document.get("sequential_points") != budget - initial:
        raise ValueError("graph profile sequential_points must equal budget minus initial_points")
    beta = document.get("beta_policy")
    if not isinstance(beta, dict):
        raise ValueError("graph profile beta_policy must be a mapping")
    if beta.get("id") == "fixed_beta_v1":
        FixedBeta(float(beta.get("value")), policy_id="fixed_beta_v1")
    elif beta.get("id") == "joint_chi_square_response_dimension_v1":
        JointChiSquareResponseDimensionBeta.from_document(
            beta, expected_response_dimension=GRAPH_RESPONSE_DIMENSION
        )
    else:
        raise ValueError("graph profile selects an unknown beta policy")
    if document.get("terminal_recommendation") not in TERMINAL_POLICIES:
        raise ValueError("graph profile selects an unknown terminal policy")
    if document.get("evaluation_randomness") != INDEPENDENT:
        raise ValueError("ordinary graph profiles require independent evaluation randomness")
    if document.get("assessment") != FRESH_HELD_OUT:
        raise ValueError("ordinary graph profiles require fresh held-out assessment")
    standardization = document.get("standardization")
    if not isinstance(standardization, Mapping):
        raise ValueError("graph profile standardization must be a mapping")
    policy = standardization.get("policy", STANDARDIZE_INITIAL_DESIGN)
    if policy == STANDARDIZE_INITIAL_DESIGN:
        if (
            set(standardization) - {"policy", "ddof", "scale_floor"}
            or standardization.get("ddof") != 1
            or not isinstance(standardization.get("scale_floor"), (int, float))
            or isinstance(standardization.get("scale_floor"), bool)
            or not math.isfinite(float(standardization["scale_floor"]))
            or float(standardization["scale_floor"]) <= 0
        ):
            raise ValueError("standardized graph profiles require ddof=1 and a positive scale floor")
    elif policy == IDENTITY_PHYSICAL:
        if dict(standardization) != {"policy": IDENTITY_PHYSICAL}:
            raise ValueError("identity graph profile accepts no fitted-transform controls")
    else:
        raise ValueError("graph profile selects an unknown standardization policy")
    return document


def _policy_bundle(profile: Mapping[str, Any]) -> dict[str, Any]:
    bundle = {
        "profile_id": profile["profile_id"],
        "initial_design": {
            "policy": profile["initial_design_policy"],
            "point_count": profile["initial_points"],
        },
        "confidence_geometry": profile["confidence_geometry"],
        "beta_policy": dict(profile["beta_policy"]),
        "infeasible_fallback": profile["infeasible_fallback"],
        "terminal_recommendation": profile["terminal_recommendation"],
        "evaluation_randomness": profile["evaluation_randomness"],
        "assessment": profile["assessment"],
    }
    # Historical profiles omitted this redundant projection. Preserve their
    # accepted policy identities while making new selectable policies explicit.
    if "policy" in profile["standardization"]:
        bundle["standardization"] = dict(profile["standardization"])
    return bundle


def _resolved_beta(beta_document: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    beta = dict(beta_document)
    if beta.get("id") == "fixed_beta_v1":
        resolved = FixedBeta(float(beta.get("value")), policy_id="fixed_beta_v1")
        return resolved.value, beta
    resolved = JointChiSquareResponseDimensionBeta.from_document(
        beta, expected_response_dimension=GRAPH_RESPONSE_DIMENSION
    )
    return resolved.value, beta


def _walk(value: Any, *, path: str = "configuration") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_ORDINARY_KEYS:
                raise ValueError(f"ordinary graph-paper configuration forbids {path}.{key}")
            _walk(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and value in _FORBIDDEN_ORDINARY_VALUES:
        raise ValueError(f"ordinary graph-paper configuration forbids {value!r}")


def _proposal_selection(
    document: Mapping[str, Any], *, allow_compatibility: bool
) -> tuple[str, dict[str, Any], str]:
    explicit = document.get("proposal_controls")
    profile = document.get("proposal_effort")
    if explicit is not None:
        if profile is not None:
            raise ValueError("select proposal_controls or proposal_effort, not both")
        if not isinstance(explicit, Mapping):
            raise ValueError("proposal_controls must be a mapping")
        expected_keys = {"raw_samples", "restarts", "max_iterations", "learning_rate"}
        if set(explicit) != expected_keys:
            raise ValueError("proposal_controls must contain exactly the registered controls")
        controls = dict(explicit)
        for key in ("raw_samples", "restarts", "max_iterations"):
            value = controls[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"proposal_controls.{key} must be a positive integer")
        learning_rate = controls["learning_rate"]
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0:
            raise ValueError("proposal_controls.learning_rate must be positive")
        controls["learning_rate"] = float(learning_rate)
        return EXPLICIT_CONTROLS, controls, "active explicit computational controls"
    if profile in PROPOSAL_PROFILES:
        return str(profile), dict(PROPOSAL_PROFILES[str(profile)]), "frozen compatibility alias"
    if allow_compatibility and profile in COMPATIBILITY_PROPOSAL_PROFILES:
        return (
            str(profile),
            dict(COMPATIBILITY_PROPOSAL_PROFILES[str(profile)]),
            "quarantined compatibility alias",
        )
    raise ValueError(f"unknown ordinary graph-paper proposal selection: {profile!r}")


def validate_graph_paper_configuration(
    document: Mapping[str, Any], *, allow_compatibility: bool = False
) -> None:
    """Fail closed on ordinary graph-paper drift and diagnostic leakage."""

    _proposal_selection(document, allow_compatibility=allow_compatibility)
    allowed_terminal = TERMINAL_POLICIES + (
        COMPATIBILITY_TERMINAL_POLICIES if allow_compatibility else ()
    )
    if document.get("terminal_recommendation") not in allowed_terminal:
        raise ValueError(f"unknown graph-paper terminal_recommendation: {document.get('terminal_recommendation')!r}")
    if document.get("evaluation_randomness") != INDEPENDENT:
        raise ValueError("ordinary graph-paper evaluation_randomness must be independent")
    if document.get("assessment") != FRESH_HELD_OUT:
        raise ValueError("ordinary graph-paper assessment must be fresh_held_out")
    policy = document.get("graph_paper_policy")
    if policy is not None:
        if not isinstance(policy, Mapping):
            raise ValueError("graph_paper_policy must be a mapping")
        profile_id = policy.get("profile_id")
        if not isinstance(profile_id, str):
            raise ValueError("graph_paper_policy requires profile_id")
        profile = load_graph_policy_profile(profile_id)
        expected = _policy_bundle(profile)
        if dict(policy) != expected:
            raise ValueError("graph_paper_policy does not match the registered profile")
        runner = document.get("runner")
        if not isinstance(runner, Mapping):
            raise ValueError("registered graph-paper policy requires runner configuration")
        if int(runner.get("initial_points", -1)) != int(profile["initial_points"]):
            raise ValueError("runner initial_points differs from the selected graph profile")
        if int(runner.get("sequential_rounds", -1)) != int(profile["sequential_points"]):
            raise ValueError("runner sequential_rounds differs from the selected graph profile")
        if int(document.get("simulation_budget_per_target", -1)) != int(profile["simulation_budget"]):
            raise ValueError("simulation budget differs from the selected graph profile")
        if document.get("terminal_recommendation") != profile["terminal_recommendation"]:
            raise ValueError("terminal recommendation differs from the selected graph profile")
        transform = runner.get("transform")
        if not isinstance(transform, Mapping):
            raise ValueError("registered graph-paper profile requires runner transform settings")
        expected_transform = dict(profile["standardization"])
        if dict(transform) != expected_transform:
            raise ValueError("runner transform differs from the selected graph profile")
    _walk(document)
    declaration = document.get("target_declaration")
    if not isinstance(declaration, Mapping):
        raise ValueError("ordinary graph-paper configuration requires target_declaration")
    path = (ROOT / str(declaration.get("path", ""))).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("target declaration must remain inside the repository") from exc
    if not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise ValueError("target declaration and hash sidecar must exist")
    if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("target declaration hash mismatch")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    campaign = document.get("campaign", {})
    if ledger.get("schema_version") != declaration.get("schema_version"):
        raise ValueError("target declaration schema mismatch")
    if ledger.get("target_count") != campaign.get("target_count"):
        raise ValueError("target declaration count mismatch")
    if ledger.get("master_seed") != campaign.get("campaign_seed"):
        raise ValueError("target declaration master-seed mismatch")
    records = ledger.get("records")
    if not isinstance(records, list) or len(records) != campaign.get("target_count") or not records:
        raise ValueError("target declaration records must match the positive campaign target count")
    if any("target_response" in item for item in records):
        raise ValueError("graph-paper target declaration must not contain responses")


def resolved_proposal_settings(document: Mapping[str, Any]) -> dict[str, Any]:
    validate_graph_paper_configuration(document)
    profile, controls, classification = _proposal_selection(
        document, allow_compatibility=False
    )
    settings = {
        "profile_id": profile,
        "classification": classification,
        "raw_policy": "scrambled_sobol", **controls, "alpha": 0.1, "tau_interior": 1.0e-8,
    }
    policy = document.get("graph_paper_policy")
    if isinstance(policy, Mapping):
        beta_value, beta_document = _resolved_beta(policy["beta_policy"])
        settings.update(
            {
                "beta": beta_value,
                "beta_policy": beta_document,
                "confidence_radius_policy": str(policy["confidence_geometry"]),
                "infeasible_fallback_policy": str(policy["infeasible_fallback"]),
            }
        )
    return settings


def resolved_compatibility_proposal_settings(
    document: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve a retained alias only through an explicit compatibility API."""

    if document.get("proposal_effort") not in COMPATIBILITY_PROPOSAL_PROFILES:
        raise ValueError("compatibility resolver requires a registered compatibility alias")
    validate_graph_paper_configuration(document, allow_compatibility=True)
    profile, controls, classification = _proposal_selection(
        document, allow_compatibility=True
    )
    settings = {
        "profile_id": profile,
        "classification": classification,
        "raw_policy": "scrambled_sobol",
        **controls,
        "alpha": 0.1,
        "tau_interior": 1.0e-8,
    }
    policy = document.get("graph_paper_policy")
    if isinstance(policy, Mapping):
        beta_value, beta_document = _resolved_beta(policy["beta_policy"])
        settings.update(
            {
                "beta": beta_value,
                "beta_policy": beta_document,
                "confidence_radius_policy": str(policy["confidence_geometry"]),
                "infeasible_fallback_policy": str(policy["infeasible_fallback"]),
            }
        )
    return settings


def ordinary_graph_paper_policy_bundle(
    profile_id: str = PAPER120_PROFILE,
) -> dict[str, Any]:
    """Materialize one registered option profile for the shared core path."""

    return _policy_bundle(load_graph_policy_profile(profile_id))


def graph_paper_randomness_family(
    *,
    target_record_id: str | None,
    simulator_identity: str,
    policy_identity: Mapping[str, Any] | None = None,
) -> ScientificIdentity:
    document = {
        "schema": (
            "graph-paper-method-randomness-family-v1"
            if policy_identity is None
            else "graph-paper-method-randomness-family-v2"
        ),
        "target_record_id": target_record_id,
        "simulator_identity": simulator_identity,
        "method": "mutaco", "policy": "independent_coordinate_streams",
    }
    if policy_identity is not None:
        document["method_policy_identity"] = dict(policy_identity)
    return ScientificIdentity.create(document)


def recommend_graph_terminal(
    *, document: Mapping[str, Any], surrogate: Any,
    transform: FrozenResponseTransform, target_physical: torch.Tensor,
    target_id: str, theta: torch.Tensor, observation_ids: Sequence[str],
    parameter_spec: ParameterSpec, seed_deriver: SeedDeriver, target_index: int,
) -> GraphTerminalRecommendation:
    """Select an active or retained compatibility policy; no graph math lives here."""

    validate_graph_paper_configuration(document, allow_compatibility=True)
    if document["terminal_recommendation"] == EVALUATED_PHYSICAL_L2:
        return recommend_evaluated_physical_l2(
            surrogate=surrogate, transform=transform, target_physical=target_physical,
            target_id=target_id, theta=theta, observation_ids=observation_ids,
        )
    recommender = (
        recommend_continuous_physical_mae
        if document["terminal_recommendation"] == CONTINUOUS_PHYSICAL_MAE
        else recommend_continuous_physical_l2
    )
    return recommender(
        surrogate=surrogate, transform=transform, target_physical=target_physical,
        target_id=target_id, parameter_spec=parameter_spec, seed_deriver=seed_deriver,
        target_index=target_index,
    )


def support_document() -> dict[str, Any]:
    return {
        "schema_version": "graph-paper-option-support-v2",
        "ordinary_profiles": {
            profile_id: ordinary_graph_paper_policy_bundle(profile_id)
            for profile_id in (
                CANONICAL_PROFILE,
                PAPER120_PROFILE,
                PAPER120_NO_STANDARDIZATION_PROFILE,
            )
        },
        "proposal_profiles": {
            key: {
                "classification": "frozen compatibility alias",
                "resolved_controls": value,
            }
            for key, value in PROPOSAL_PROFILES.items()
        },
        "explicit_proposal_controls": {
            "classification": "active ordinary configuration",
            "required_keys": ["raw_samples", "restarts", "max_iterations", "learning_rate"],
        },
        "compatibility_proposal_profiles": {
            key: {
                "classification": "quarantined compatibility alias",
                "resolved_controls": value,
            }
            for key, value in COMPATIBILITY_PROPOSAL_PROFILES.items()
        },
        "terminal_policies": {
            CONTINUOUS_PHYSICAL_L2: {
                "classification": "active ordinary graph option", "formula": "||mu_phys(theta)-target_phys||_2^2",
                "domain": "continuous_bounded_theta", "posterior": "mean_only", "simulator_calls": 0,
                "gp_fits": 0, "optimizer": CONTINUOUS_SETTINGS,
            },
            CONTINUOUS_PHYSICAL_MAE: {
                "classification": "active ordinary graph-paper profile selection",
                "formula": "mean_m |mu_phys,m(theta)-target_phys,m|",
                "domain": "continuous_bounded_theta",
                "posterior": "mean_only",
                "simulator_calls": 0,
                "gp_fits": 0,
                "optimizer": CONTINUOUS_PHYSICAL_MAE_SETTINGS,
            },
        },
        "compatibility_terminal_policies": {
            EVALUATED_PHYSICAL_L2: {
                "classification": "generic compatibility API; not ordinary graph configuration",
                "formula": "||mu_phys(theta)-target_phys||_2^2",
                "domain": "evaluated_theta_only",
                "posterior": "mean_only",
                "simulator_calls": 0,
                "gp_fits": 0,
            },
        },
    }
