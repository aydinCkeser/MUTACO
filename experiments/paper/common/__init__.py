"""Paper-specific protocols and artifact operations, resolved on first use."""

from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module
from importlib.util import spec_from_loader as _spec_from_loader
from typing import Any

from ..analysis import timing_report

# Preserve the timing pickle/import name without loading scientific or plotting libraries.
_sys.modules[__name__ + ".timing_report"] = timing_report


# These historic names occur in pickles and in direct imports. Look them up only
# when requested: importing the parent alone must work without Torch or SBI.
_IMPORT_OWNERS = {
    "cross_budget_synthesis_cli": "experiments.paper.analysis.cross_budget_synthesis_cli",
    "evidence_analysis_cli": "experiments.paper.analysis.evidence_analysis_cli",
    "repaired_sbi_assessment_preparation_cli": "experiments.paper.historical_workflows.repaired_sbi_assessment_preparation_cli",
    "repaired_sbi_recovery_cli": "experiments.paper.historical_workflows.repaired_sbi_recovery_cli",
    "repaired_sbi_salvage_cli": "experiments.paper.historical_workflows.repaired_sbi_salvage_cli",
    "repaired_sbi_terminal_archive_cli": "experiments.paper.historical_workflows.repaired_sbi_terminal_archive_cli",
    "repaired_sbi_terminal_preparation_cli": "experiments.paper.historical_workflows.repaired_sbi_terminal_preparation_cli",
    "scalar_primary_b100_preparation_audit_cli": "experiments.paper.historical_workflows.scalar_primary_b100_preparation_audit_cli",
    "terminal_lineage_forensics_cli": "experiments.paper.historical_workflows.terminal_lineage_forensics_cli",
    "aggregate": "experiments.paper.analysis.aggregate",
    "campaign": "experiments.paper.protocol.campaign",
    "cli": "experiments.paper.protocol.cli",
    "cohort_v2": "experiments.paper.protocol.cohort_v2",
    "compact_append": "experiments.paper.protocol.compact_append",
    "cross_budget_synthesis": "experiments.paper.analysis.cross_budget_synthesis",
    "integrity": "experiments.paper.protocol.integrity",
    "method_append": "experiments.paper.protocol.method_append",
    "method_specs": "experiments.paper.protocol.method_specs",
    "repaired_sbi_assessment_execution": "experiments.paper.historical_workflows.repaired_sbi_assessment_execution",
    "repaired_sbi_assessment_preparation": "experiments.paper.historical_workflows.repaired_sbi_assessment_preparation",
    "supplied_targets": "experiments.paper.protocol.supplied_targets",
    "target_freeze_v2": "experiments.paper.protocol.target_freeze_v2",
}


class _CommonImportAliases:
    """Resolve historical experiment imports and CLI names in this package alone."""

    def find_spec(self, fullname, target=None):
        name = fullname.removeprefix(__name__ + ".")
        if name in _IMPORT_OWNERS:
            return _spec_from_loader(fullname, self)
        return None

    def get_code(self, fullname):
        # python -m historically dispatched the CLI names to their canonical main.
        name = fullname.rsplit(".", 1)[1]
        owner = _IMPORT_OWNERS[name]
        if name.endswith("_cli"):
            body = f"from {owner} import main\nif __name__ == '__main__':\n    main()"
        else:
            body = f"from importlib import import_module\nimport_module({owner!r})"
        return compile(body, f"<{fullname}>", "exec")

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # The loader returns the actual owner, including its original spec and globals.
        old = module.__name__
        _sys.modules[old] = _import_module(_IMPORT_OWNERS[old.rsplit(".", 1)[1]])


# PathFinder consults this entry only for children of this package. Real CLI
# files take precedence and keep their original python -m and script behavior.
_alias_path = __path__[0] + "/__historical_imports__"
_sys.path_importer_cache[_alias_path] = _CommonImportAliases()
__path__.append(_alias_path)


_EXPORTS = {
    'CohortDeclarationSpec': ('cohort_v2', 'CohortDeclarationSpec'),
    'materialize_cohort_declaration': ('cohort_v2', 'materialize_cohort_declaration'),
    'validate_cohort_declaration': ('cohort_v2', 'validate_cohort_declaration'),
    'MethodDeclarationV3': ('population_v3', 'MethodDeclarationV3'),
    'PopulationDeclarationV3': ('population_v3', 'PopulationDeclarationV3'),
    'collect_population_v3': ('population_v3', 'collect_population_v3'),
    'execute_population_v3': ('population_v3', 'execute_population_v3'),
    'population_call_preview': ('population_v3', 'population_call_preview'),
    'load_population_declaration_v3': ('population_v3', 'load_population_declaration_v3'),
    'shard_indices': ('population_v3', 'shard_indices'),
    'compact_method_record': ('compact_append', 'compact_method_record'),
    'join_compact_method_records': ('compact_append', 'join_compact_method_records'),
    'write_consolidated_compact_evidence': ('compact_append', 'write_consolidated_compact_evidence'),
    'DEFERRED_SBI_COMPARISON_GROUP': ('method_specs', 'DEFERRED_SBI_COMPARISON_GROUP'),
    'PRIMARY_COMPARISON_GROUP': ('method_specs', 'PRIMARY_COMPARISON_GROUP'),
    'MethodExecutionSpec': ('method_specs', 'MethodExecutionSpec'),
    'load_method_execution_specs': ('method_specs', 'load_method_execution_specs'),
    'selected_method_configuration_paths': ('method_specs', 'selected_method_configuration_paths'),
    'collect_target_freeze': ('target_freeze_v2', 'collect_target_freeze'),
    'execute_target_freeze': ('target_freeze_v2', 'execute_target_freeze'),
    'preview_target_freeze': ('target_freeze_v2', 'preview_target_freeze'),
    'load_supplied_target_set': ('supplied_targets', 'load_supplied_target_set'),
    'contract_document': ('cross_experiment_contract', 'contract_document'),
    'load_cross_experiment_contract': ('cross_experiment_contract', 'load_cross_experiment_contract'),
    'operation_owner': ('cross_experiment_contract', 'operation_owner'),
    'simulator_scientific_backbone': ('cross_experiment_contract', 'simulator_scientific_backbone'),
    'validate_primary_budget_pair': ('cross_experiment_contract', 'validate_primary_budget_pair'),
    'validate_primary_declaration_contract': ('cross_experiment_contract', 'validate_primary_declaration_contract'),
    'ERROR_KEYS': ('evidence_analysis', 'ERROR_KEYS'),
    'CROSS_EXPERIMENT_REPORT_SCHEMA': ('evidence_analysis', 'REPORT_SCHEMA'),
    'CROSS_EXPERIMENT_ROW_SCHEMA': ('evidence_analysis', 'ROW_SCHEMA'),
    'CROSS_EXPERIMENT_SPEC_SCHEMA': ('evidence_analysis', 'SPEC_SCHEMA'),
    'adapt_mechanism_ablation_records': ('evidence_analysis', 'adapt_mechanism_ablation_records'),
    'adapt_paper_v3_bundle': ('evidence_analysis', 'adapt_paper_v3_bundle'),
    'adapt_scalar_primary_export': ('evidence_analysis', 'adapt_scalar_primary_export'),
    'analysis_specification': ('evidence_analysis', 'analysis_specification'),
    'canonical_json_text': ('evidence_analysis', 'canonical_json_text'),
    'deterministic_analysis': ('evidence_analysis', 'deterministic_analysis'),
    'validate_analysis_rows': ('evidence_analysis', 'validate_analysis_rows'),
    'CROSS_BUDGET_REPORT_SCHEMA': ('cross_budget_synthesis', 'REPORT_SCHEMA'),
    'ROW_SCHEMA_CROSS_BUDGET': ('cross_budget_synthesis', 'ROW_SCHEMA_CROSS_BUDGET'),
    'CROSS_BUDGET_SPEC_SCHEMA': ('cross_budget_synthesis', 'SPEC_SCHEMA'),
    'build_cross_budget_rows': ('cross_budget_synthesis', 'build_cross_budget_rows'),
    'deterministic_cross_budget_synthesis': ('cross_budget_synthesis', 'deterministic_cross_budget_synthesis'),
    'validate_synthesis_specification': ('cross_budget_synthesis', 'validate_synthesis_specification'),
    'build_b100_rows': ('terminal_lineage_forensics', 'build_b100_rows'),
    'build_b40_rows': ('terminal_lineage_forensics', 'build_b40_rows'),
    'forensic_audit': ('terminal_lineage_forensics', 'forensic_audit'),
    'forensic_report': ('terminal_lineage_forensics', 'forensic_report'),
    'forensic_specification': ('terminal_lineage_forensics', 'forensic_specification'),
    'validate_forensic_rows': ('terminal_lineage_forensics', 'validate_forensic_rows'),
    'verify_exact_source_git': ('terminal_lineage_forensics', 'verify_exact_source_git'),
    'inventory_salvage': ('repaired_sbi_salvage', 'inventory_salvage'),
    'salvage_design': ('repaired_sbi_salvage', 'salvage_design'),
    'validate_salvage_design': ('repaired_sbi_salvage', 'validate_salvage_design'),
    'exact_state_qualification': ('repaired_sbi_recovery', 'exact_state_qualification'),
    'recover_b100_inventory': ('repaired_sbi_recovery', 'recover_b100_inventory'),
    'corrected_terminal_declaration': ('repaired_sbi_terminal_preparation', 'corrected_terminal_declaration'),
    'preparation_inventory': ('repaired_sbi_terminal_preparation', 'preparation_inventory'),
    'validate_corrected_terminal_declaration': ('repaired_sbi_terminal_preparation', 'validate_corrected_terminal_declaration'),
    'collect_terminal_archive': ('repaired_sbi_terminal_archive', 'collect_terminal_archive'),
    'repaired_sbi_assessment_context': ('repaired_sbi_assessment_preparation', 'assessment_context'),
    'repaired_sbi_assessment_declaration': ('repaired_sbi_assessment_preparation', 'assessment_declaration'),
    'repaired_sbi_assessment_inventory': ('repaired_sbi_assessment_preparation', 'preparation_inventory'),
}
_SUBMODULES = ('cohort_v2', 'population_v3', 'compact_append', 'method_specs', 'target_freeze_v2', 'supplied_targets', 'cross_experiment_contract', 'evidence_analysis', 'cross_budget_synthesis', 'terminal_lineage_forensics', 'repaired_sbi_salvage', 'repaired_sbi_recovery', 'repaired_sbi_terminal_preparation', 'repaired_sbi_terminal_archive', 'repaired_sbi_assessment_preparation')

__all__ = (
    "CohortDeclarationSpec",
    "MethodDeclarationV3",
    "PopulationDeclarationV3",
    "collect_population_v3",
    "collect_target_freeze",
    "execute_population_v3",
    "execute_target_freeze",
    "materialize_cohort_declaration",
    "load_population_declaration_v3",
    "population_call_preview",
    "compact_method_record",
    "join_compact_method_records",
    "write_consolidated_compact_evidence",
    "MethodExecutionSpec",
    "PRIMARY_COMPARISON_GROUP",
    "DEFERRED_SBI_COMPARISON_GROUP",
    "load_method_execution_specs",
    "selected_method_configuration_paths",
    "load_supplied_target_set",
    "preview_target_freeze",
    "shard_indices",
    "validate_cohort_declaration",
    "contract_document",
    "load_cross_experiment_contract",
    "operation_owner",
    "simulator_scientific_backbone",
    "validate_primary_budget_pair",
    "validate_primary_declaration_contract",
    "ERROR_KEYS",
    "CROSS_EXPERIMENT_REPORT_SCHEMA",
    "CROSS_EXPERIMENT_ROW_SCHEMA",
    "CROSS_EXPERIMENT_SPEC_SCHEMA",
    "adapt_mechanism_ablation_records",
    "adapt_paper_v3_bundle",
    "adapt_scalar_primary_export",
    "analysis_specification",
    "canonical_json_text",
    "deterministic_analysis",
    "validate_analysis_rows",
    "CROSS_BUDGET_REPORT_SCHEMA",
    "ROW_SCHEMA_CROSS_BUDGET",
    "CROSS_BUDGET_SPEC_SCHEMA",
    "build_cross_budget_rows",
    "deterministic_cross_budget_synthesis",
    "validate_synthesis_specification",
    "build_b100_rows",
    "build_b40_rows",
    "forensic_audit",
    "forensic_report",
    "forensic_specification",
    "validate_forensic_rows",
    "verify_exact_source_git",
    "inventory_salvage",
    "salvage_design",
    "validate_salvage_design",
    "exact_state_qualification",
    "recover_b100_inventory",
    "corrected_terminal_declaration",
    "preparation_inventory",
    "validate_corrected_terminal_declaration",
    "collect_terminal_archive",
    "repaired_sbi_assessment_context",
    "repaired_sbi_assessment_declaration",
    "repaired_sbi_assessment_inventory",
)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module, symbol = _EXPORTS[name]
        value = getattr(_import_module("." + module, __name__), symbol)
    elif name in _SUBMODULES:
        value = _import_module("." + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Cache the actual object, preserving ordinary package monkeypatch behavior.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
