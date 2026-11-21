"""Versioned method configuration set for the unified Paper-v3 runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.benchmarking.artifacts import file_hash

from .population_v3 import MethodDeclarationV3
from .validation import PaperContractError, validate_method_ids


PRIMARY_COMPARISON_GROUP = "mutaco_npe_snpe_primary"
DEFERRED_SBI_COMPARISON_GROUP = "conditional_likelihood_ratio_sbi"

_METHOD_CLASSIFICATION = {
    "mutaco": ("mutaco", PRIMARY_COMPARISON_GROUP, "mutaco"),
    "mutaco_standardized": (
        "mutaco", "mutaco_standardization_ablation", "mutaco"
    ),
    "mutaco_unstandardized": (
        "mutaco", "mutaco_standardization_ablation", "mutaco"
    ),
    "sbi_npe": ("direct_posterior", PRIMARY_COMPARISON_GROUP, "direct_posterior"),
    "sbi_snpe": ("direct_posterior", PRIMARY_COMPARISON_GROUP, "direct_posterior"),
    "sbi_nle": ("likelihood", DEFERRED_SBI_COMPARISON_GROUP, "deferred"),
    "sbi_snle": ("likelihood", DEFERRED_SBI_COMPARISON_GROUP, "deferred"),
    "sbi_nre": ("ratio", DEFERRED_SBI_COMPARISON_GROUP, "deferred"),
    "sbi_snre": ("ratio", DEFERRED_SBI_COMPARISON_GROUP, "deferred"),
}


@dataclass(frozen=True)
class MethodExecutionSpec:
    method: str
    method_family: str
    comparison_group: str
    adapter: str
    configuration_path: Path
    configuration_sha256: str
    inference_budget: int
    decision_policy_identifier: str

    def validate_against(
        self, declaration: MethodDeclarationV3, *, root: Path
    ) -> Path:
        if declaration.method != self.method:
            raise PaperContractError("method spec and declaration methods differ")
        if declaration.inference_budget != self.inference_budget:
            raise PaperContractError("method spec and declaration budgets differ")
        if declaration.decision_policy_identifier != self.decision_policy_identifier:
            raise PaperContractError("method spec and declaration decisions differ")
        path = (root / self.configuration_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise PaperContractError("method configuration escapes the repository") from exc
        if not path.is_file() or file_hash(path) != self.configuration_sha256:
            raise PaperContractError("method configuration hash mismatch")
        return path


def load_method_execution_specs(
    document: Mapping[str, Any], *, source: Path
) -> tuple[MethodExecutionSpec, ...]:
    """Load one canonical, approval-gated method set without executing methods."""

    if document.get("schema_version") != "mutaco-paper-method-spec-set-v1":
        raise PaperContractError("unified runner requires a v1 method-spec set")
    if document.get("state") != "unexecuted" or document.get("execution_requires_approval") is not True:
        raise PaperContractError("method-spec set must remain unexecuted and approval-gated")
    raw = document.get("methods")
    if not isinstance(raw, list) or not raw:
        raise PaperContractError("method-spec set requires methods")
    specs = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise PaperContractError("method specs must be objects")
        method = str(item.get("method", ""))
        validate_method_ids((method,))
        expected = _METHOD_CLASSIFICATION[method]
        observed = (
            str(item.get("method_family", "")),
            str(item.get("comparison_group", "")),
            str(item.get("adapter", "")),
        )
        if observed != expected:
            raise PaperContractError(f"method classification drifted for {method}")
        budget = item.get("inference_budget")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise PaperContractError("method-spec budgets must be positive integers")
        sha = str(item.get("configuration_sha256", ""))
        if len(sha) != 64:
            raise PaperContractError("method configuration SHA-256 is malformed")
        path = Path(str(item.get("configuration_path", "")))
        if not str(path) or path.is_absolute():
            raise PaperContractError("method configuration path must be repository-relative")
        specs.append(MethodExecutionSpec(
            method=method,
            method_family=observed[0],
            comparison_group=observed[1],
            adapter=observed[2],
            configuration_path=path,
            configuration_sha256=sha,
            inference_budget=budget,
            decision_policy_identifier=str(item.get("decision_policy_identifier", "")),
        ))
    validate_method_ids(tuple(spec.method for spec in specs))
    if len({spec.method for spec in specs}) != len(specs):
        raise PaperContractError("method-spec set contains duplicate methods")
    del source
    return tuple(specs)


def selected_method_configuration_paths(
    specs: Sequence[MethodExecutionSpec],
    declarations: Sequence[MethodDeclarationV3],
    *,
    root: Path,
) -> dict[str, Path]:
    """Resolve exactly the declared subset and reject unavailable adapters."""

    by_method = {spec.method: spec for spec in specs}
    selected = {}
    for declaration in declarations:
        spec = by_method.get(declaration.method)
        if spec is None:
            raise PaperContractError(f"declared method lacks a method spec: {declaration.method}")
        if spec.adapter == "deferred":
            raise PaperContractError(
                f"deferred SBI method lacks an approved Paper-v3 adapter: {spec.method}"
            )
        selected[spec.method] = spec.validate_against(declaration, root=root)
    return selected

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.method_specs")
