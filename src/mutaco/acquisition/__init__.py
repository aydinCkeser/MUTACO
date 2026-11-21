"""MUTACO Step 3 and the retained public imports of the composite NEI comparator."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from .confidence_geometry import (
    ConfidenceGeometryEvaluation,
    FixedBeta,
    JointChiSquareResponseDimensionBeta,
    chi_square_quantile,
    evaluate_confidence_geometry,
)
from .acquisition_objectives import (
    BarrierObjectiveResult,
    ExplorationResult,
    MatchedBarrierObjective,
    evaluate_exploration,
    standardized_l2_exploitation,
)
from .candidate_proposal import (
    RAW_CANDIDATE_POLICIES,
    AcquisitionBranch,
    BranchReason,
    CandidatePool,
    DuplicateDiagnostic,
    DuplicateRecoveryDiagnostic,
    FallbackReason,
    LatentPosteriorProvider,
    PosteriorCallCounts,
    ProposalConfig,
    ProposalFailureReason,
    ProposalResult,
    ProposalStatus,
    RawCandidatePolicy,
    RawRanking,
    RestartDiagnostic,
    RestartStatus,
    RestartStopReason,
    ReturnedCandidateSource,
    candidate_policy_support_markdown,
    generate_candidate_pool,
    propose_candidate,
    recover_best_ranked_unevaluated_raw,
)
from mutaco.comparators import composite_mc_nei as _composite_mc_nei
from mutaco.comparators.composite_mc_nei import (
    COMPOSITE_MC_NEI_PHYSICAL_MAE,
    NEI_BASELINE_FACTORIZATION,
    NEI_RAW_RECOVERY,
    JointLatentPosteriorProvider,
    NEIAcquisitionError,
    NEIBaselineFactorizationDiagnostic,
    NEICandidateEvaluation,
    NEIConfig,
    NEIProposalResult,
    evaluate_composite_mc_nei_candidates,
    NEIRestartDiagnostic,
    propose_composite_mc_nei,
)

from . import acquisition_objectives, candidate_proposal, confidence_geometry

# Historical imports, patches and pickle globals refer to these module names.
geometry = _bind_legacy_module(confidence_geometry, __name__ + ".geometry")
objectives = _bind_legacy_module(acquisition_objectives, __name__ + ".objectives")
proposal = _bind_legacy_module(candidate_proposal, __name__ + ".proposal")
nei = _bind_legacy_module(_composite_mc_nei, __name__ + ".nei")

__all__ = [
    "BarrierObjectiveResult",
    "ConfidenceGeometryEvaluation",
    "FixedBeta",
    "JointChiSquareResponseDimensionBeta",
    "chi_square_quantile",
    "ExplorationResult",
    "MatchedBarrierObjective",
    "evaluate_confidence_geometry",
    "evaluate_exploration",
    "standardized_l2_exploitation",
    "RAW_CANDIDATE_POLICIES",
    "AcquisitionBranch",
    "BranchReason",
    "CandidatePool",
    "DuplicateDiagnostic",
    "DuplicateRecoveryDiagnostic",
    "FallbackReason",
    "LatentPosteriorProvider",
    "PosteriorCallCounts",
    "ProposalConfig",
    "ProposalFailureReason",
    "ProposalResult",
    "ProposalStatus",
    "RawCandidatePolicy",
    "RawRanking",
    "RestartDiagnostic",
    "RestartStatus",
    "RestartStopReason",
    "ReturnedCandidateSource",
    "candidate_policy_support_markdown",
    "generate_candidate_pool",
    "propose_candidate",
    "recover_best_ranked_unevaluated_raw",
    "COMPOSITE_MC_NEI_PHYSICAL_MAE",
    "NEI_BASELINE_FACTORIZATION",
    "NEI_RAW_RECOVERY",
    "JointLatentPosteriorProvider",
    "NEIAcquisitionError",
    "NEIBaselineFactorizationDiagnostic",
    "NEICandidateEvaluation",
    "NEIConfig",
    "NEIProposalResult",
    "evaluate_composite_mc_nei_candidates",
    "NEIRestartDiagnostic",
    "propose_composite_mc_nei",
]
