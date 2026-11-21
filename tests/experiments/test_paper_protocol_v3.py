from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from experiments.paper.common.aggregate import summarize_paired
from experiments.paper.common.method_append import append_method_record, load_method_record
from experiments.paper.common.protocol_v3 import (
    ASSESSMENT_POLICY_ID,
    PROTOCOL_ID,
    RECORD_SCHEMA_V3,
    DecisionOutcome,
    PaperComparisonConfigV3,
    PhysicalCallCounts,
    assess_frozen_decisions,
    freeze_decisions,
    method_inference_context,
    paired_assessment_context,
    validate_v3_record,
)
from experiments.paper.common.validation import PaperContractError


def _config() -> PaperComparisonConfigV3:
    return PaperComparisonConfigV3(
        simulator_definition_identity="toy-linear-definition-v1",
        simulator_configuration_identity="toy-linear-config-v1",
        assessment_fidelity="toy-physical-m2-v1",
        assessment_campaign_seed=20260809,
        inference_campaign_seed=20260810,
    )


def _decision(
    method: str,
    theta: tuple[float, ...] | None,
    *,
    status: str = "recommended",
) -> DecisionOutcome:
    return DecisionOutcome(
        method=method,
        status=status,
        decision_policy_identifier=(
            "continuous_posterior_mean_physical_l1_mae"
            if method == "mutaco"
            else "test_zero_call_decision"
        ),
        method_configuration_identity=f"config-{method}",
        inference_budget=4,
        inference_calls=PhysicalCallCounts(4, 4),
        recommended_theta=theta,
        failure_type="ToyDecisionFailure" if status == "failed" else None,
        failure_explanation="deliberate toy failure" if status == "failed" else None,
    )


def _batch(events: list[str], *decisions: DecisionOutcome):
    def persist(document):
        assert document["status"] == "frozen_before_assessment"
        assert len(document["decisions"]) == len(decisions)
        events.append("persisted_all_decisions")

    return freeze_decisions(_config(), "target-A", decisions, persist=persist)


def test_assessment_is_method_independent_while_inference_is_method_addressed():
    config = _config()
    assessment = paired_assessment_context(config, "target-A")
    repeated = paired_assessment_context(config, "target-A")
    other_target = paired_assessment_context(config, "target-B")
    other_replicate = paired_assessment_context(config, "target-A", replicate=1)
    assert assessment == repeated
    assert assessment["policy_identifier"] == ASSESSMENT_POLICY_ID
    assert assessment["coordinate"] == {
        "namespace": "paired_held_out_assessment",
        "coordinates": ["target-A", 0],
    }
    assert assessment["evaluation_seed"] != other_target["evaluation_seed"]
    assert assessment["evaluation_seed"] != other_replicate["evaluation_seed"]

    mutaco = method_inference_context(config, "target-A", "mutaco", "mutaco-cfg")
    npe = method_inference_context(config, "target-A", "sbi_npe", "npe-cfg")
    assert mutaco["family_identity"] != npe["family_identity"]
    assert mutaco["evaluation_seed"] != npe["evaluation_seed"]
    assert mutaco["coordinate"]["coordinates"] == ["target-A", "mutaco"]
    assert npe["coordinate"]["coordinates"] == ["target-A", "sbi_npe"]


def test_freeze_precedes_all_assessment_and_metrics_reuse_one_call():
    events: list[str] = []
    batch = _batch(
        events,
        _decision("mutaco", (1.0, 2.0)),
        _decision("sbi_npe", (2.0, 1.0)),
    )
    contexts = []

    def evaluate(theta, context):
        assert events == ["persisted_all_decisions"] or events[-1] == "assessed"
        events.append("assessed")
        contexts.append(dict(context))
        return (theta[0] + 1.0, theta[1] + 2.0)

    records = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(1.0, 1.0),
        theta_star=(0.0, 0.0),
        target_construction_counts=PhysicalCallCounts(0, 0),
        target_disposition="reused",
        evaluate=evaluate,
    )
    assert events == ["persisted_all_decisions", "assessed", "assessed"]
    assert contexts[0] == contexts[1]
    assert len({record["decision_freeze_identity"] for record in records}) == 1
    assert len({record["assessment_context"]["evaluation_seed"] for record in records}) == 1
    for record in records:
        assert record["status"] == "completed"
        assert record["decision"]["recommendation_simulator_calls"] == 0
        assert record["call_ledger"]["terminal_recommendation"]["attempted_calls"] == 0
        assert record["call_ledger"]["paired_assessment_and_metric"]["successful_calls"] == 1
        assert record["call_ledger"]["method_inference_or_training"]["nominal_calls"] == 4
    mutaco = records[0]
    assert mutaco["response_error_l1_sum"] == pytest.approx(4.0)
    assert mutaco["response_error_mae"] == pytest.approx(2.0)
    assert mutaco["response_error_l2"] == pytest.approx(math.sqrt(10.0))
    assert mutaco["parameter_error_l2"] == pytest.approx(math.sqrt(5.0))


def test_failed_decision_is_retained_and_never_assessed():
    events: list[str] = []
    batch = _batch(
        events,
        _decision("mutaco", (1.0, 2.0)),
        _decision("sbi_npe", None, status="failed"),
    )
    calls = []

    def evaluate(theta, context):
        calls.append((theta, context["evaluation_seed"]))
        return (1.0, 1.0)

    records = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(1.0, 1.0),
        theta_star=None,
        target_construction_counts=PhysicalCallCounts(1, 1),
        target_disposition="constructed",
        evaluate=evaluate,
    )
    assert len(calls) == 1
    failed = records[1]
    assert failed["method"] == "sbi_npe"
    assert failed["status"] == "failed"
    assert failed["failure_stage"] == "final_decision"
    assert failed["call_ledger"]["paired_assessment_and_metric"]["attempted_calls"] == 0
    assert all(failed[field] is None for field in (
        "response_error_l1_sum", "response_error_mae", "response_error_l2",
        "parameter_error_l2",
    ))


def test_failed_sample_map_does_not_require_success_metadata():
    failed = DecisionOutcome(
        method="sbi_npe",
        status="failed",
        decision_policy_identifier="posterior_sample_map_physical_v1",
        method_configuration_identity="config-sbi_npe",
        inference_budget=40,
        inference_calls=PhysicalCallCounts(40, 40),
        failure_type="StrictMetadataError",
        failure_explanation="post-training persistence failed",
    )
    batch = _batch([], failed)
    records = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(1.0, 1.0),
        theta_star=(0.0, 0.0),
        target_construction_counts=PhysicalCallCounts(0, 0),
        target_disposition="reused",
        evaluate=lambda *args: pytest.fail("failed decision must not be assessed"),
    )
    assert records[0]["status"] == "failed"
    assert records[0]["decision"]["decision_metadata"] is None
    validate_v3_record(records[0])


def test_assessment_failure_is_typed_and_does_not_stop_another_method():
    events: list[str] = []
    batch = _batch(
        events,
        _decision("mutaco", (1.0, 2.0)),
        _decision("sbi_npe", (2.0, 1.0)),
    )

    def evaluate(theta, context):
        if theta[0] == 1.0:
            raise RuntimeError("toy assessment failure")
        return (1.0, 1.0)

    records = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(1.0, 1.0),
        theta_star=(0.0, 0.0),
        target_construction_counts=PhysicalCallCounts(0, 0),
        target_disposition="reused",
        evaluate=evaluate,
    )
    assert records[0]["status"] == "failed"
    assert records[0]["failure_stage"] == "paired_held_out_assessment"
    assert records[0]["call_ledger"]["paired_assessment_and_metric"] == {
        "category": "paired_assessment_and_metric",
        "scope": "target_method_record",
        "nominal_calls": 1,
        "budget_treatment": "outside_method_inference_budget",
        "attempted_calls": 1,
        "successful_calls": 0,
        "failed_calls": 1,
        "retry_calls": 0,
        "ledger_identity": records[0]["call_ledger"]["paired_assessment_and_metric"]["ledger_identity"],
    }
    assert records[1]["status"] == "completed"


def test_v3_reader_is_strict_and_v2_reader_remains_unchanged(tmp_path: Path):
    events: list[str] = []
    batch = _batch(events, _decision("mutaco", (1.0, 2.0)))
    record = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(1.0, 1.0),
        theta_star=(0.0, 0.0),
        target_construction_counts=PhysicalCallCounts(0, 0),
        target_disposition="reused",
        evaluate=lambda theta, context: (2.0, 4.0),
    )[0]
    path = tmp_path / "v3.json"
    append_method_record(path, record)
    loaded = load_method_record(path)
    assert loaded is not None
    assert loaded["schema_version"] == RECORD_SCHEMA_V3
    assert loaded["paper_comparison_protocol"] == PROTOCOL_ID
    assert "target_error_l1" not in loaded

    leaked = copy.deepcopy(record)
    leaked["target_error_l1"] = leaked["response_error_l1_sum"]
    with pytest.raises(PaperContractError, match="legacy metric"):
        validate_v3_record(leaked)
    malformed = copy.deepcopy(record)
    del malformed["call_ledger"]["terminal_recommendation"]
    with pytest.raises(PaperContractError, match="four-category"):
        validate_v3_record(malformed)
    wrong_family = copy.deepcopy(record)
    wrong_family["assessment_context"]["family_definition"]["method"] = "mutaco"
    with pytest.raises(PaperContractError, match="forbidden fields"):
        validate_v3_record(wrong_family)
    wrong_ledger = copy.deepcopy(record)
    wrong_ledger["call_ledger"]["paired_assessment_and_metric"][
        "ledger_identity"
    ] = "paper-call-ledger-entry-v1-corrupt"
    with pytest.raises(PaperContractError, match="ledger identity"):
        validate_v3_record(wrong_ledger)
    with pytest.raises(PaperContractError, match="legacy metric"):
        append_method_record(tmp_path / "invalid-v3.json", leaked)

    legacy = {
        "schema_version": "mutaco-paper-target-method-record-v2",
        "status": "completed",
        "method": "sbi_npe",
        "target_error_l1": 3.0,
        "target_error_l2": 2.0,
        "design_error_l2": 1.0,
    }
    legacy_path = tmp_path / "v2.json"
    append_method_record(legacy_path, legacy)
    assert load_method_record(legacy_path) == {
        **legacy,
        "record_content_hash": load_method_record(legacy_path)["record_content_hash"],
    }


def test_v3_aggregation_uses_explicit_metrics_without_scientific_calls():
    events: list[str] = []
    batch = _batch(
        events,
        _decision("mutaco", (1.0, 2.0)),
        _decision("sbi_npe", (2.0, 1.0)),
    )
    records = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(1.0, 1.0),
        theta_star=(0.0, 0.0),
        target_construction_counts=PhysicalCallCounts(0, 0),
        target_disposition="reused",
        evaluate=lambda theta, context: (theta[0], theta[1]),
    )
    before = list(events)
    summary = summarize_paired(records)
    assert events == before
    assert summary["record_schema"] == RECORD_SCHEMA_V3
    assert summary["metrics"] == (
        "response_error_l1_sum",
        "response_error_mae",
        "response_error_l2",
        "parameter_error_l2",
    )
    assert "mean_left_minus_right_response_l2" in summary["paired"][0]
    assert "mean_left_minus_right_target_l2" not in summary["paired"][0]
