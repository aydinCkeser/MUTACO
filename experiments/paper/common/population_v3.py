"""Prospective append-only population orchestration for v3 paper records.

Scientific method implementations remain in their experiment-owned adapters.
This module owns ordering, caps, freeze-before-assessment, persistence, and
shard-safe collection through explicit callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Callable, Mapping, Sequence

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, write_pretty_json

from .method_append import append_method_record, load_method_record, record_path
from .protocol_v3 import (
    DecisionOutcome,
    PaperComparisonConfigV3,
    PhysicalCallCounts,
    assess_frozen_decisions,
    freeze_decisions,
    method_inference_context,
)
from .validation import PaperContractError, validate_method_ids


@dataclass(frozen=True)
class MethodDeclarationV3:
    method: str
    configuration_identity: str
    decision_policy_identifier: str
    inference_budget: int

    def __post_init__(self) -> None:
        validate_method_ids((self.method,))
        if not self.configuration_identity or not self.decision_policy_identifier:
            raise PaperContractError("v3 method identities must be nonempty")
        if isinstance(self.inference_budget, bool) or not isinstance(self.inference_budget, int) or self.inference_budget < 0:
            raise PaperContractError("v3 inference budget must be nonnegative")


@dataclass(frozen=True)
class PopulationDeclarationV3:
    campaign_id: str
    comparison: PaperComparisonConfigV3
    target_ledger_hash: str
    target_record_identities: tuple[str, ...]
    methods: tuple[MethodDeclarationV3, ...]
    output_root: str

    def __post_init__(self) -> None:
        for field in ("campaign_id", "target_ledger_hash", "output_root"):
            if not str(getattr(self, field)).strip():
                raise PaperContractError(f"{field} must be nonempty")
        if not self.target_record_identities or len(set(self.target_record_identities)) != len(self.target_record_identities):
            raise PaperContractError("population target identities must be nonempty and unique")
        if not self.methods:
            raise PaperContractError("population declaration requires methods")
        validate_method_ids(tuple(method.method for method in self.methods))

    def document(self) -> dict[str, Any]:
        body = {
            "schema_version": "mutaco-paper-population-declaration-v3",
            "state": "unexecuted",
            "execution_requires_approval": True,
            "campaign_id": self.campaign_id,
            "paper_comparison": self.comparison.document(),
            "target_ledger_hash": self.target_ledger_hash,
            "target_record_identities": list(self.target_record_identities),
            "methods": [
                {
                    "method": item.method,
                    "configuration_identity": item.configuration_identity,
                    "decision_policy_identifier": item.decision_policy_identifier,
                    "inference_budget": item.inference_budget,
                }
                for item in self.methods
            ],
            "output_root": self.output_root,
            "call_preview": population_call_preview(len(self.target_record_identities), self.methods),
        }
        return {**body, "declaration_identity": stable_hash("paper-population-declaration-v3", body)}


def load_population_declaration_v3(document: Mapping[str, Any]) -> PopulationDeclarationV3:
    """Load only the explicit v3 schema; v2 fallback is forbidden."""

    if document.get("schema_version") != "mutaco-paper-population-declaration-v3":
        raise PaperContractError("population runner requires explicit v3 declaration")
    if document.get("state") != "unexecuted" or document.get("execution_requires_approval") is not True:
        raise PaperContractError("population v3 declaration must remain approval-gated")
    raw_comparison = document.get("paper_comparison")
    if not isinstance(raw_comparison, Mapping):
        raise PaperContractError("population v3 comparison is missing")
    comparison = PaperComparisonConfigV3(
        simulator_definition_identity=str(raw_comparison.get("simulator_definition_identity", "")),
        simulator_configuration_identity=str(raw_comparison.get("simulator_configuration_identity", "")),
        assessment_fidelity=str(raw_comparison.get("assessment_fidelity", "")),
        assessment_campaign_seed=raw_comparison.get("assessment_campaign_seed"),
        inference_campaign_seed=raw_comparison.get("inference_campaign_seed"),
        protocol_id=str(raw_comparison.get("protocol_id", "")),
        record_schema=str(raw_comparison.get("record_schema", "")),
        assessment_policy=str(raw_comparison.get("assessment_policy", "")),
        metric_schema=str(raw_comparison.get("metric_schema", "")),
    )
    if raw_comparison.get("configuration_identity") != comparison.identity:
        raise PaperContractError("population v3 comparison identity mismatch")
    raw_methods = document.get("methods")
    if not isinstance(raw_methods, list):
        raise PaperContractError("population v3 methods must be a list")
    declaration = PopulationDeclarationV3(
        campaign_id=str(document.get("campaign_id", "")),
        comparison=comparison,
        target_ledger_hash=str(document.get("target_ledger_hash", "")),
        target_record_identities=tuple(str(value) for value in document.get("target_record_identities", ())),
        methods=tuple(
            MethodDeclarationV3(
                method=str(item.get("method", "")),
                configuration_identity=str(item.get("configuration_identity", "")),
                decision_policy_identifier=str(item.get("decision_policy_identifier", "")),
                inference_budget=item.get("inference_budget"),
            )
            for item in raw_methods if isinstance(item, Mapping)
        ),
        output_root=str(document.get("output_root", "")),
    )
    if declaration.document() != dict(document):
        raise PaperContractError("population v3 declaration is not canonical")
    return declaration


@dataclass
class PopulationCallCap:
    maximum: int
    attempted: int = 0
    successful: int = 0
    failed: int = 0
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.maximum, bool) or not isinstance(self.maximum, int) or self.maximum < 0:
            raise PaperContractError("physical-call maximum must be nonnegative")

    def invoke(self, category: str, operation: Callable[[], Any]) -> Any:
        if category not in {"method_inference_or_training", "paired_assessment_and_metric"}:
            raise PaperContractError("population v3 cannot charge target or recommendation calls")
        if self.attempted >= self.maximum:
            raise PaperContractError("physical-call cap exhausted before attempted call")
        counts = self.by_category.setdefault(
            category, {"attempted_calls": 0, "successful_calls": 0, "failed_calls": 0}
        )
        self.attempted += 1
        counts["attempted_calls"] += 1
        try:
            result = operation()
        except Exception:
            self.failed += 1
            counts["failed_calls"] += 1
            raise
        self.successful += 1
        counts["successful_calls"] += 1
        return result

    def document(self) -> dict[str, int]:
        return {
            "maximum_calls": self.maximum,
            "attempted_calls": self.attempted,
            "successful_calls": self.successful,
            "failed_calls": self.failed,
        }

    def category_document(self) -> dict[str, dict[str, int]]:
        return {
            category: dict(self.by_category.get(category, {
                "attempted_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
            }))
            for category in (
                "method_inference_or_training",
                "paired_assessment_and_metric",
            )
        }


def population_call_preview(
    target_count: int, methods: Sequence[MethodDeclarationV3]
) -> dict[str, int]:
    if target_count <= 0:
        raise PaperContractError("target_count must be positive")
    if not methods:
        raise PaperContractError("population preview requires methods")
    validate_method_ids(tuple(method.method for method in methods))
    inference = target_count * sum(method.inference_budget for method in methods)
    assessments = target_count * len(methods)
    return {
        "target_construction_calls": 0,
        "maximum_inference_calls": inference,
        "maximum_assessment_calls": assessments,
        "maximum_total_calls": inference + assessments,
        "recommendation_calls": 0,
        "scientific_calls_executed": 0,
    }


def shard_indices(target_count: int, *, rank: int, workers: int) -> tuple[int, ...]:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (target_count, rank, workers)):
        raise PaperContractError("shard values must be integers")
    if target_count < 0 or workers <= 0 or rank < 0 or rank >= workers:
        raise PaperContractError("invalid shard geometry")
    return tuple(range(rank, target_count, workers))


def _write_hashed(path: Path, document: Mapping[str, Any]) -> None:
    write_pretty_json(path, document)
    path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")


def _attempt_path(output: Path, target_id: str, method: str) -> Path:
    token = sha256(target_id.encode("utf-8")).hexdigest()[:16]
    return output / "attempts_v3" / f"target-{token}" / f"{method}.json"


def _persist_assessment_randomness_raw(
    output: Path, record: dict[str, Any]
) -> None:
    raw = record.pop("_raw_assessment_randomness_evidence", None)
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise PaperContractError("raw assessment randomness evidence is malformed")
    compact = record.get("assessment_randomness_evidence")
    manifests = raw.get("namespace_seed_manifests")
    full = raw.get("full_seed_manifest")
    if not isinstance(compact, Mapping) or not isinstance(manifests, Mapping) or not isinstance(full, Mapping):
        raise PaperContractError("assessment randomness raw/compact evidence is incomplete")
    merged = {}
    for namespace in ("quenched_disorder", "initial_state", "thermal_noise"):
        manifest = manifests.get(namespace)
        if not isinstance(manifest, Mapping) or not manifest:
            raise PaperContractError(f"raw assessment randomness namespace is missing: {namespace}")
        if set(merged).intersection(manifest):
            raise PaperContractError("raw assessment randomness namespaces overlap")
        merged.update(manifest)
        expected = stable_hash(
            "paper-assessment-seed-manifest-namespace-v1",
            {"namespace": namespace, "seeds": dict(manifest)},
        )
        if compact["namespace_seed_manifest_hashes"].get(namespace) != expected:
            raise PaperContractError("raw assessment namespace hash differs from compact record")
    if dict(full) != merged:
        raise PaperContractError("raw full assessment seed manifest differs from namespace union")
    full_hash = stable_hash("paper-assessment-seed-manifest-v1", dict(full))
    if compact.get("full_seed_manifest_hash") != full_hash:
        raise PaperContractError("raw full assessment seed hash differs from compact record")
    target_id = str(record["target_record_identity"])
    method = str(record["method"])
    token = sha256(target_id.encode("utf-8")).hexdigest()[:16]
    path = output / "assessment_randomness" / f"target-{token}" / f"{method}.json"
    if path.exists() or path.with_suffix(".sha256").exists():
        raise PaperContractError("append-only raw assessment randomness evidence exists")
    document = {
        **dict(raw),
        "target_record_identity": target_id,
        "method": method,
        "assessment_context": record["assessment_context"],
        "namespace_seed_manifest_hashes": compact["namespace_seed_manifest_hashes"],
        "full_seed_manifest_hash": full_hash,
    }
    _write_hashed(path, document)
    record["assessment_randomness_raw_evidence"] = {
        "path": path.relative_to(output).as_posix(),
        "sha256": file_hash(path),
        "schema_version": raw.get("schema_version"),
    }


def execute_population_v3(
    *, comparison: PaperComparisonConfigV3, targets: Sequence[Mapping[str, Any]],
    methods: Sequence[MethodDeclarationV3], output: Path, maximum_calls: int,
    run_method: Callable[[MethodDeclarationV3, Mapping[str, Any], Mapping[str, Any], Callable[[Sequence[float], Mapping[str, Any]], Sequence[float]]], DecisionOutcome],
    evaluate_inference: Callable[[str, Mapping[str, Any], Sequence[float], Mapping[str, Any]], Sequence[float]],
    evaluate_assessment: Callable[[Mapping[str, Any], Sequence[float], Mapping[str, Any]], Sequence[float]],
    execution_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run v3 population work and always retain terminal call evidence."""

    attempt_path = output / "population_execution_attempt.json"
    if attempt_path.exists() or attempt_path.with_suffix(".sha256").exists():
        raise PaperContractError("append-only population execution attempt already exists")
    cap = PopulationCallCap(maximum_calls)
    state: dict[str, Any] = {
        "last_completed_phase": "not_started",
        "completed_slots": 0,
        "failed_slots": 0,
    }
    try:
        result = _execute_population_v3_body(
            comparison=comparison,
            targets=targets,
            methods=methods,
            output=output,
            maximum_calls=maximum_calls,
            run_method=run_method,
            evaluate_inference=evaluate_inference,
            evaluate_assessment=evaluate_assessment,
            cap=cap,
            state=state,
        )
    except Exception as exc:
        attempt = {
            "schema_version": "mutaco-paper-v3-population-execution-attempt-v1",
            "status": "failed",
            "declared_maximum_calls": maximum_calls,
            "physical_calls": cap.document(),
            "physical_calls_by_category": cap.category_document(),
            "completed_slots": state["completed_slots"],
            "failed_slots": state["failed_slots"],
            "last_completed_phase": state["last_completed_phase"],
            "terminal_failure_type": type(exc).__name__,
            "terminal_failure_explanation": str(exc),
            "execution_provenance": dict(execution_provenance or {}),
            "no_automatic_retry": True,
        }
        _write_hashed(attempt_path, attempt)
        raise
    attempt = {
        "schema_version": "mutaco-paper-v3-population-execution-attempt-v1",
        "status": "completed",
        "declared_maximum_calls": maximum_calls,
        "physical_calls": cap.document(),
        "physical_calls_by_category": cap.category_document(),
        "completed_slots": state["completed_slots"],
        "failed_slots": state["failed_slots"],
        "last_completed_phase": state["last_completed_phase"],
        "terminal_failure_type": None,
        "terminal_failure_explanation": None,
        "execution_provenance": dict(execution_provenance or {}),
        "no_automatic_retry": True,
    }
    _write_hashed(attempt_path, attempt)
    return result


def _execute_population_v3_body(
    *, comparison: PaperComparisonConfigV3, targets: Sequence[Mapping[str, Any]],
    methods: Sequence[MethodDeclarationV3], output: Path, maximum_calls: int,
    run_method: Callable[[MethodDeclarationV3, Mapping[str, Any], Mapping[str, Any], Callable[[Sequence[float], Mapping[str, Any]], Sequence[float]]], DecisionOutcome],
    evaluate_inference: Callable[[str, Mapping[str, Any], Sequence[float], Mapping[str, Any]], Sequence[float]],
    evaluate_assessment: Callable[[Mapping[str, Any], Sequence[float], Mapping[str, Any]], Sequence[float]],
    cap: PopulationCallCap,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Run selected frozen targets under the accepted v3 causal sequence."""

    declared_methods = tuple(methods)
    validate_method_ids(tuple(item.method for item in declared_methods))
    if not targets:
        raise PaperContractError("v3 population execution requires frozen targets")
    target_ids = tuple(str(target["target_record_id"]) for target in targets)
    if len(set(target_ids)) != len(target_ids):
        raise PaperContractError("frozen target identities must be unique")
    preview = population_call_preview(len(targets), declared_methods)
    if maximum_calls != preview["maximum_total_calls"]:
        raise PaperContractError("v3 execution requires the exact declared maximum call cap")
    state["last_completed_phase"] = "input_validation"
    completed_slots = 0
    failed_slots = 0
    timing_events: list[dict[str, Any]] = []
    for target in targets:
        target_id = str(target["target_record_id"])
        existing = {
            method.method: load_method_record(record_path(output, target_id, method.method))
            for method in declared_methods
        }
        if all(record is not None for record in existing.values()):
            completed_slots += sum(record["status"] == "completed" for record in existing.values() if record is not None)
            failed_slots += sum(record["status"] == "failed" for record in existing.values() if record is not None)
            state["completed_slots"] = completed_slots
            state["failed_slots"] = failed_slots
            state["last_completed_phase"] = "existing_record_validation"
            continue
        if any(record is not None for record in existing.values()):
            raise PaperContractError("partial target-method record set requires explicit recovery review")
        decisions = []
        for declaration in declared_methods:
            attempt_path = _attempt_path(output, target_id, declaration.method)
            if attempt_path.exists() or attempt_path.with_suffix(".sha256").exists():
                raise PaperContractError("append-only v3 method attempt already exists")
            context = method_inference_context(
                comparison, target_id, declaration.method, declaration.configuration_identity
            )
            before_attempted, before_successful, before_failed = cap.attempted, cap.successful, cap.failed
            method_wall, method_cpu = perf_counter(), process_time()

            def evaluate(theta: Sequence[float], simulator_context: Mapping[str, Any]) -> Sequence[float]:
                return cap.invoke(
                    "method_inference_or_training",
                    lambda: evaluate_inference(declaration.method, target, theta, simulator_context),
                )

            try:
                outcome = run_method(declaration, target, context, evaluate)
                if not isinstance(outcome, DecisionOutcome):
                    raise PaperContractError("method runner must return DecisionOutcome")
                if outcome.method != declaration.method or outcome.inference_budget != declaration.inference_budget:
                    raise PaperContractError("method outcome differs from v3 declaration")
                if outcome.method_configuration_identity != declaration.configuration_identity:
                    raise PaperContractError("method outcome configuration differs from v3 declaration")
            except Exception as exc:
                attempted = cap.attempted - before_attempted
                successful = cap.successful - before_successful
                failed = cap.failed - before_failed
                outcome = DecisionOutcome(
                    method=declaration.method,
                    status="failed",
                    decision_policy_identifier=declaration.decision_policy_identifier,
                    method_configuration_identity=declaration.configuration_identity,
                    inference_budget=declaration.inference_budget,
                    inference_calls=PhysicalCallCounts(attempted, successful, failed),
                    failure_stage=str(getattr(exc, "failure_stage", "method_dispatch")),
                    failure_type=type(exc).__name__,
                    failure_explanation=str(exc),
                )
            finally:
                timing_events.append({
                    "target_record_identity": target_id,
                    "method": declaration.method,
                    "phase": "method_inference_and_zero_call_decision_dispatch",
                    "wall_seconds": perf_counter() - method_wall,
                    "process_cpu_seconds": process_time() - method_cpu,
                })
            charged = cap.attempted - before_attempted
            if outcome.inference_calls.attempted != charged:
                raise PaperContractError("method inference ledger differs from capped physical calls")
            attempt = {
                "schema_version": "mutaco-paper-v3-method-attempt-v1",
                "target_record_identity": target_id,
                "method": declaration.method,
                "status": outcome.status,
                "failure_stage": outcome.failure_stage,
                "failure_type": outcome.failure_type,
                "failure_explanation": outcome.failure_explanation,
                "inference_context": context,
                "inference_calls": outcome.inference_calls.document(),
                "no_automatic_retry": True,
            }
            _write_hashed(attempt_path, attempt)
            decisions.append(outcome)
            state["last_completed_phase"] = "method_attempt_persistence"
        freeze_token = sha256(target_id.encode("utf-8")).hexdigest()[:16]
        freeze_path = output / "decision_freezes" / f"target-{freeze_token}.json"
        if freeze_path.exists() or freeze_path.with_suffix(".sha256").exists():
            raise PaperContractError("append-only decision freeze already exists")
        batch = freeze_decisions(
            comparison, target_id, decisions, persist=lambda document: _write_hashed(freeze_path, document)
        )
        state["last_completed_phase"] = "decision_freeze"

        assessment_position = 0
        assessment_methods = [
            decision.method for decision in decisions if decision.status == "recommended"
        ]

        def assess(theta: tuple[float, ...], context: Mapping[str, Any]) -> Sequence[float]:
            nonlocal assessment_position
            if not freeze_path.is_file() or not freeze_path.with_suffix(".sha256").is_file():
                raise PaperContractError("paired assessment requires persisted all-method freeze")
            method = assessment_methods[assessment_position]
            assessment_position += 1
            wall, cpu = perf_counter(), process_time()
            try:
                return cap.invoke(
                    "paired_assessment_and_metric",
                    lambda: evaluate_assessment(target, theta, context),
                )
            finally:
                timing_events.append({
                    "target_record_identity": target_id,
                    "method": method,
                    "phase": "paired_held_out_assessment",
                    "wall_seconds": perf_counter() - wall,
                    "process_cpu_seconds": process_time() - cpu,
                })

        records = assess_frozen_decisions(
            comparison, batch,
            target_response=target["target_response"], theta_star=target.get("theta_star"),
            target_construction_counts=PhysicalCallCounts(0, 0),
            target_disposition="reused", evaluate=assess,
        )
        state["last_completed_phase"] = "assessment_and_record_construction"
        for record in records:
            record["population_timing_events"] = [
                event for event in timing_events
                if event["target_record_identity"] == target_id
                and event["method"] == record["method"]
            ]
            _persist_assessment_randomness_raw(output, record)
            persisted = append_method_record(record_path(output, target_id, record["method"]), record)
            completed_slots += persisted["status"] == "completed"
            failed_slots += persisted["status"] == "failed"
            state["completed_slots"] = completed_slots
            state["failed_slots"] = failed_slots
            state["last_completed_phase"] = "method_record_persistence"
    result = {
        "schema_version": "mutaco-paper-v3-population-execution-v1",
        "protocol_id": comparison.protocol_id,
        "configuration_identity": comparison.identity,
        "target_count": len(targets),
        "methods": [method.method for method in declared_methods],
        "completed_slots": completed_slots,
        "failed_slots": failed_slots,
        "call_preview": preview,
        "physical_calls": cap.document(),
        "physical_calls_by_category": cap.category_document(),
        "target_construction_calls": 0,
        "recommendation_calls": 0,
    }
    _write_hashed(
        output / "population_timing.json",
        {
            "schema_version": "mutaco-paper-v3-population-timing-v1",
            "events": timing_events,
            "scientific_calls_added_by_timing": 0,
        },
    )
    _write_hashed(output / "population_execution.json", result)
    state["last_completed_phase"] = "population_execution_persistence"
    return result


def collect_population_v3(
    output: Path, *, target_identities: Sequence[str], methods: Sequence[str]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Strictly collect every declared completed-or-failed v3 slot."""

    selected_methods = validate_method_ids(methods)
    targets = tuple(str(value) for value in target_identities)
    if len(set(targets)) != len(targets):
        raise PaperContractError("collection target identities must be unique")
    rows, missing = [], []
    for target in targets:
        for method in selected_methods:
            record = load_method_record(record_path(output, target, method))
            if record is None:
                missing.append({"target_record_identity": target, "method": method})
            else:
                rows.append(record)
    manifest = {
        "schema_version": "mutaco-paper-v3-population-collection-v1",
        "declared_target_count": len(targets),
        "declared_method_count": len(selected_methods),
        "declared_slot_count": len(targets) * len(selected_methods),
        "collected_slot_count": len(rows),
        "missing_slots": missing,
        "complete": not missing,
        "scientific_calls": 0,
    }
    _write_hashed(output / "population_collection.json", manifest)
    return manifest, tuple(rows)
