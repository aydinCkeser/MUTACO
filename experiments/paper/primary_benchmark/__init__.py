"""Versioned primary MUTACO--NEI--Sobol benchmark orchestration."""

from .design import PrimarySobolDesign, generate_primary_sobol_design
from .protocol import (
    PRIMARY_METHODS,
    PRIMARY_PROTOCOL_ID,
    PRIMARY_RECORD_SCHEMA,
    PrimaryComparisonConfig,
    PrimaryDecisionOutcome,
    PrimaryMethodDeclaration,
    PrimaryPopulationDeclaration,
    PrimaryTrajectoryEntry,
    load_primary_population_declaration,
    primary_population_call_preview,
)
from .qualification import (
    NEIQualificationDeclaration,
    NEINumericalTier,
    load_nei_qualification_declaration,
    nei_qualification_call_preview,
    select_nei_numerical_tier,
)
from .qualification_execution import (
    QualificationCallCap,
    append_hashed_json,
    execute_qualification_worker,
    load_hashed_json,
    persist_qualification_attempt,
    qualification_worker_preview,
)
from .worker import execute_primary_worker, primary_worker_preview
from .execution import (
    PrimaryCallCap,
    PrimaryTargetExecution,
    execute_primary_target,
    primary_assessment_context,
    primary_method_context,
)
from .analysis import (
    endpoint_results,
    paired_endpoint_differences,
    runtime_results,
    trajectory_results,
)
from .assessment_rescue import (
    execute_primary_assessment_rescue,
    load_primary_assessment_rescue_declaration,
    primary_assessment_rescue_preview,
    validate_primary_assessment_rescue_freeze,
)
from .numerical_qualification import evaluate_qualification_snapshot
from .scalar_protocol import (
    SCALAR_PRIMARY_DECLARATION_SCHEMA,
    SCALAR_PRIMARY_METHODS,
    SCALAR_PRIMARY_PROTOCOL_ID,
    SCALAR_PRIMARY_RECORD_SCHEMA,
    ScalarPrimaryComparisonConfig,
    ScalarPrimaryPopulationDeclaration,
    load_scalar_primary_population_declaration,
    scalar_primary_call_preview,
)

__all__ = [
    "PRIMARY_METHODS",
    "PRIMARY_PROTOCOL_ID",
    "PRIMARY_RECORD_SCHEMA",
    "SCALAR_PRIMARY_DECLARATION_SCHEMA",
    "SCALAR_PRIMARY_METHODS",
    "SCALAR_PRIMARY_PROTOCOL_ID",
    "SCALAR_PRIMARY_RECORD_SCHEMA",
    "NEIQualificationDeclaration",
    "NEINumericalTier",
    "PrimaryCallCap",
    "PrimaryComparisonConfig",
    "PrimaryDecisionOutcome",
    "PrimaryMethodDeclaration",
    "PrimaryPopulationDeclaration",
    "PrimarySobolDesign",
    "PrimaryTargetExecution",
    "PrimaryTrajectoryEntry",
    "ScalarPrimaryComparisonConfig",
    "ScalarPrimaryPopulationDeclaration",
    "QualificationCallCap",
    "append_hashed_json",
    "execute_primary_worker",
    "execute_qualification_worker",
    "evaluate_qualification_snapshot",
    "execute_primary_target",
    "endpoint_results",
    "generate_primary_sobol_design",
    "load_nei_qualification_declaration",
    "load_primary_population_declaration",
    "load_hashed_json",
    "load_scalar_primary_population_declaration",
    "persist_qualification_attempt",
    "nei_qualification_call_preview",
    "primary_assessment_context",
    "primary_method_context",
    "primary_population_call_preview",
    "scalar_primary_call_preview",
    "primary_worker_preview",
    "qualification_worker_preview",
    "paired_endpoint_differences",
    "runtime_results",
    "execute_primary_assessment_rescue",
    "load_primary_assessment_rescue_declaration",
    "primary_assessment_rescue_preview",
    "validate_primary_assessment_rescue_freeze",
    "select_nei_numerical_tier",
    "trajectory_results",
]
