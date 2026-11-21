"""Strict validation shared by paper execution and reload-only analysis."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.contracts import PaperContractError
from reproducibility.recorded_paths import resolve_recorded_path

# Keep the historical pickle module-name object as well as its spelling.
PaperContractError.__module__ = __name__


REGISTERED_METHODS = (
    "mutaco",
    "mutaco_standardized",
    "mutaco_unstandardized",
    "sbi_npe",
    "sbi_snpe",
    "sbi_nle",
    "sbi_snle",
    "sbi_nre",
    "sbi_snre",
    "sbi_nle_single_round",
    "sbi_nre_single_round",
)


FROZEN_PAPER100_CONFIGURATION = (
    "experiments/graph_inversion/configs/d4_100_targets.json"
)
FROZEN_PAPER100_SHA256 = (
    "11c0494594007f7c09161ba52e175a764aae2f0ef9ca23eaab196d7737f8ac4e"
)
FROZEN_PAPER100_DUPLICATE_KEYS = frozenset({"artifact_policy", "storage_plan"})


def read_json_object(
    path: str | Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Read one object while rejecting duplicates except the frozen declaration."""

    source = Path(path).resolve()
    duplicates: list[tuple[str, Any, Any]] = []

    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append((key, result[key], value))
            result[key] = value
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=object_hook
        )
    except json.JSONDecodeError as exc:
        raise PaperContractError(f"invalid JSON document: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperContractError(f"expected JSON object: {source}")
    if not duplicates:
        return value

    root = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    frozen = (root / FROZEN_PAPER100_CONFIGURATION).resolve()
    digest = sha256(source.read_bytes()).hexdigest()
    duplicate_keys = frozenset(key for key, _, _ in duplicates)
    identical_values = all(first == second for _, first, second in duplicates)
    if not (
        source == frozen
        and digest == FROZEN_PAPER100_SHA256
        and duplicate_keys == FROZEN_PAPER100_DUPLICATE_KEYS
        and identical_values
    ):
        raise PaperContractError(
            "duplicate JSON keys are forbidden: "
            + ", ".join(sorted(duplicate_keys))
        )
    return value


def validate_method_ids(methods: Sequence[str], *, declared: Sequence[str] | None = None) -> tuple[str, ...]:
    resolved = tuple(str(method) for method in methods)
    if len(set(resolved)) != len(resolved):
        raise PaperContractError("method identifiers must be unique")
    unknown = set(resolved) - set(REGISTERED_METHODS)
    if unknown:
        raise PaperContractError(f"unknown method identifiers: {sorted(unknown)}")
    if declared is not None:
        undeclared = set(resolved) - set(declared)
        if undeclared:
            raise PaperContractError(f"undeclared method identifiers: {sorted(undeclared)}")
    return resolved


def repository_path(root: Path, value: str, *, field: str) -> Path:
    path = resolve_recorded_path(root, value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PaperContractError(f"{field} must remain inside the repository") from exc
    return path


def require_unexecuted(document: Mapping[str, Any]) -> None:
    if document.get("state") != "unexecuted" or document.get("execution_requires_approval") is not True:
        raise PaperContractError("prepared paper declarations must be unexecuted and approval-gated")


REUSE_IDENTITY_FIELDS = (
    "campaign_identity",
    "simulator_identity",
    "simulator_configuration_hash",
    "target_bank_identity",
    "target_record_identity",
    "hidden_target_ledger_hash",
    "target_response_identity",
    "target_response_seed",
    "method",
    "method_configuration_hash",
    "effective_benchmark_configuration_hash",
    "inference_budget",
    "budget_interpretation",
    "evaluation_randomness_policy",
    "assessment_policy",
    "assessment_identity",
    "parameter_dimension",
    "response_dimension",
    "dtype_policy",
    "device_policy",
    "resolved_device",
    "software_schema",
    "software_compatibility_hash",
)


def assert_record_compatible(record: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    mismatches = [
        field for field in REUSE_IDENTITY_FIELDS if record.get(field) != expected.get(field)
    ]
    if mismatches:
        raise PaperContractError(
            "paper method record identity mismatch: " + ", ".join(mismatches)
        )
    if record.get("status") not in {"completed", "failed"}:
        raise PaperContractError("paper method record is incomplete")
