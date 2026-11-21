"""Versioned protocol for MUTACO--scalar qLogNEI--Sobol comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from mutaco import stable_hash
from experiments.paper.protocol.validation import PaperContractError
from .protocol import PrimaryMethodDeclaration


SCALAR_PRIMARY_PROTOCOL_ID = "mutaco-scalar-qlognei-sobol-primary-v1"
SCALAR_PRIMARY_RECORD_SCHEMA = "mutaco-scalar-primary-target-method-record-v1"
SCALAR_PRIMARY_DECLARATION_SCHEMA = "mutaco-scalar-primary-population-declaration-v1"
SCALAR_PRIMARY_METHODS = ("mutaco", "scalar_qlognei", "sobol")


def _text(value: object, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise PaperContractError(f"{label} must be nonempty")
    return result


def _seed(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PaperContractError(f"{label} must be a nonnegative integer")
    return value


def scalar_primary_call_preview(target_count: int, budget: int) -> dict[str, int]:
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        raise PaperContractError("scalar primary target count must be positive")
    if budget not in {40, 100}:
        raise PaperContractError("scalar primary budget must be 40 or 100")
    search = target_count * len(SCALAR_PRIMARY_METHODS) * budget
    assessment = target_count * len(SCALAR_PRIMARY_METHODS)
    return {
        "target_construction_calls": 0,
        "maximum_search_calls": search,
        "maximum_assessment_calls": assessment,
        "maximum_total_calls": search + assessment,
        "terminal_calls": 0,
        "scientific_calls_executed": 0,
    }


@dataclass(frozen=True)
class ScalarPrimaryComparisonConfig:
    simulator_definition_identity: str
    simulator_configuration_identity: str
    assessment_fidelity: str
    design_campaign_seed: int
    inference_campaign_seed: int
    terminal_campaign_seed: int
    assessment_campaign_seed: int
    protocol_id: str = SCALAR_PRIMARY_PROTOCOL_ID
    record_schema: str = SCALAR_PRIMARY_RECORD_SCHEMA
    assessment_policy: str = "paired_held_out_assessment_v1"
    metric_schema: str = "primary_physical_response_metrics_v1"

    def __post_init__(self) -> None:
        for name in (
            "simulator_definition_identity",
            "simulator_configuration_identity",
            "assessment_fidelity",
        ):
            _text(getattr(self, name), name)
        for name in (
            "design_campaign_seed",
            "inference_campaign_seed",
            "terminal_campaign_seed",
            "assessment_campaign_seed",
        ):
            _seed(getattr(self, name), name)
        if self.protocol_id != SCALAR_PRIMARY_PROTOCOL_ID:
            raise PaperContractError("unsupported scalar primary protocol")
        if self.record_schema != SCALAR_PRIMARY_RECORD_SCHEMA:
            raise PaperContractError("unsupported scalar primary record schema")
        if self.assessment_policy != "paired_held_out_assessment_v1":
            raise PaperContractError("unsupported scalar primary assessment policy")

    def document(self) -> dict[str, Any]:
        body = {
            "protocol_id": self.protocol_id,
            "record_schema": self.record_schema,
            "assessment_policy": self.assessment_policy,
            "metric_schema": self.metric_schema,
            "simulator_definition_identity": self.simulator_definition_identity,
            "simulator_configuration_identity": self.simulator_configuration_identity,
            "assessment_fidelity": self.assessment_fidelity,
            "design_campaign_seed": self.design_campaign_seed,
            "inference_campaign_seed": self.inference_campaign_seed,
            "terminal_campaign_seed": self.terminal_campaign_seed,
            "assessment_campaign_seed": self.assessment_campaign_seed,
        }
        return {
            **body,
            "configuration_identity": stable_hash(
                "scalar-primary-comparison-config-v1", body
            ),
        }

    @property
    def identity(self) -> str:
        return str(self.document()["configuration_identity"])


@dataclass(frozen=True)
class ScalarPrimaryPopulationDeclaration:
    campaign_id: str
    comparison: ScalarPrimaryComparisonConfig
    study_identity: str
    target_ledger_hash: str
    target_record_identities: tuple[str, ...]
    budget: int
    methods: tuple[PrimaryMethodDeclaration, ...]
    output_root: str

    decision_freeze_schema: ClassVar[str] = (
        "mutaco-scalar-primary-all-method-decision-freeze-v1"
    )
    decision_freeze_identity_domain: ClassVar[str] = (
        "scalar-primary-all-method-decision-freeze-v1"
    )
    record_identity_domain: ClassVar[str] = "scalar-primary-target-method-record-v1"
    target_execution_schema: ClassVar[str] = "mutaco-scalar-primary-target-execution-v1"
    collection_schema: ClassVar[str] = "mutaco-scalar-primary-global-collection-v1"
    collection_identity_domain: ClassVar[str] = "scalar-primary-global-collection-v1"
    compact_export_schema: ClassVar[str] = "mutaco-scalar-primary-compact-export-v1"

    def __post_init__(self) -> None:
        for name in ("campaign_id", "study_identity", "target_ledger_hash", "output_root"):
            _text(getattr(self, name), name)
        if self.budget not in {40, 100}:
            raise PaperContractError("scalar primary budget must be 40 or 100")
        if (
            not self.target_record_identities
            or len(set(self.target_record_identities)) != len(self.target_record_identities)
            or any(not item for item in self.target_record_identities)
        ):
            raise PaperContractError("scalar primary target identities are invalid")
        if tuple(item.method for item in self.methods) != SCALAR_PRIMARY_METHODS:
            raise PaperContractError(
                "scalar primary methods must be mutaco, scalar_qlognei, sobol"
            )

    def document(self) -> dict[str, Any]:
        body = {
            "schema_version": SCALAR_PRIMARY_DECLARATION_SCHEMA,
            "state": "unexecuted",
            "execution_requires_approval": True,
            "campaign_id": self.campaign_id,
            "study_identity": self.study_identity,
            "comparison": self.comparison.document(),
            "target_ledger_hash": self.target_ledger_hash,
            "target_record_identities": list(self.target_record_identities),
            "budget": self.budget,
            "methods": [item.document() for item in self.methods],
            "output_root": self.output_root,
            "call_preview": scalar_primary_call_preview(
                len(self.target_record_identities), self.budget
            ),
        }
        return {
            **body,
            "declaration_identity": stable_hash(
                "scalar-primary-population-declaration-v1", body
            ),
        }


def load_scalar_primary_population_declaration(
    document: Mapping[str, Any],
) -> ScalarPrimaryPopulationDeclaration:
    if document.get("schema_version") != SCALAR_PRIMARY_DECLARATION_SCHEMA:
        raise PaperContractError("scalar primary declaration schema is unsupported")
    if document.get("state") != "unexecuted" or document.get(
        "execution_requires_approval"
    ) is not True:
        raise PaperContractError("scalar primary declaration must remain approval-gated")
    raw_comparison = document.get("comparison")
    if not isinstance(raw_comparison, Mapping):
        raise PaperContractError("scalar primary comparison is missing")
    comparison = ScalarPrimaryComparisonConfig(
        simulator_definition_identity=str(
            raw_comparison.get("simulator_definition_identity", "")
        ),
        simulator_configuration_identity=str(
            raw_comparison.get("simulator_configuration_identity", "")
        ),
        assessment_fidelity=str(raw_comparison.get("assessment_fidelity", "")),
        design_campaign_seed=raw_comparison.get("design_campaign_seed"),
        inference_campaign_seed=raw_comparison.get("inference_campaign_seed"),
        terminal_campaign_seed=raw_comparison.get("terminal_campaign_seed"),
        assessment_campaign_seed=raw_comparison.get("assessment_campaign_seed"),
        protocol_id=str(raw_comparison.get("protocol_id", "")),
        record_schema=str(raw_comparison.get("record_schema", "")),
        assessment_policy=str(raw_comparison.get("assessment_policy", "")),
        metric_schema=str(raw_comparison.get("metric_schema", "")),
    )
    if raw_comparison.get("configuration_identity") != comparison.identity:
        raise PaperContractError("scalar primary comparison identity mismatch")
    raw_methods = document.get("methods")
    if not isinstance(raw_methods, list):
        raise PaperContractError("scalar primary methods must be a list")
    declaration = ScalarPrimaryPopulationDeclaration(
        campaign_id=str(document.get("campaign_id", "")),
        comparison=comparison,
        study_identity=str(document.get("study_identity", "")),
        target_ledger_hash=str(document.get("target_ledger_hash", "")),
        target_record_identities=tuple(
            str(item) for item in document.get("target_record_identities", ())
        ),
        budget=document.get("budget"),
        methods=tuple(
            PrimaryMethodDeclaration(
                method=str(item.get("method", "")),
                configuration_identity=str(item.get("configuration_identity", "")),
                search_policy_identifier=str(item.get("search_policy_identifier", "")),
                terminal_policy_identifier=str(item.get("terminal_policy_identifier", "")),
            )
            for item in raw_methods
            if isinstance(item, Mapping)
        ),
        output_root=str(document.get("output_root", "")),
    )
    if declaration.document() != dict(document):
        raise PaperContractError("scalar primary declaration is not canonical")
    return declaration
