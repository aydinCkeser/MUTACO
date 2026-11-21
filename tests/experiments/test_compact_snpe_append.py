from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.paper.common.compact_append import (
    join_compact_method_records,
    write_consolidated_compact_evidence,
)
from experiments.paper.common.validation import PaperContractError


def _existing():
    return [{
        "target_position": 0,
        "target_record_identity": "target-a",
        "methods": {"mutaco": {"status": "completed"}, "sbi_npe": {"status": "completed"}},
    }]


def _snpe():
    return {
        "target_record_identity": "target-a",
        "method": "sbi_snpe",
        "status": "completed",
        "recommended_theta": [0.1, 0.2],
        "response_error_l1_sum": 0.4,
        "response_error_mae": 0.2,
        "response_error_l2": 0.3,
        "parameter_error_l2": 0.1,
        "call_ledger": {"method_inference_or_training": {"successful_calls": 40}},
        "method_configuration_identity": "snpe-config",
        "record_content_hash": "snpe-record",
        "decision": {"decision_policy_identifier": "posterior_sample_map_physical_v1"},
        "population_timing_events": [{"phase": "paired_held_out_assessment"}],
    }


def test_standalone_snpe_join_is_computation_free_and_preserves_existing_rows():
    existing = _existing()
    before = deepcopy(existing)
    joined = join_compact_method_records(existing, [_snpe()])
    assert existing == before
    assert set(joined[0]["methods"]) == {"mutaco", "sbi_npe", "sbi_snpe"}
    compact = joined[0]["methods"]["sbi_snpe"]
    assert compact["response_error_mae"] == 0.2
    assert compact["decision_policy_identifier"] == "posterior_sample_map_physical_v1"
    assert compact["timing_events"] == [{"phase": "paired_held_out_assessment"}]


def test_standalone_snpe_join_rejects_unknown_duplicate_and_overwrite():
    unknown = _snpe()
    unknown["target_record_identity"] = "target-b"
    with pytest.raises(PaperContractError, match="absent"):
        join_compact_method_records(_existing(), [unknown])
    with pytest.raises(PaperContractError, match="duplicate"):
        join_compact_method_records(_existing(), [_snpe(), _snpe()])
    existing = _existing()
    existing[0]["methods"]["sbi_snpe"] = {"status": "completed"}
    with pytest.raises(PaperContractError, match="already contains"):
        join_compact_method_records(existing, [_snpe()])


def test_compact_join_accepts_multiple_new_methods_without_special_cases():
    snpe = _snpe()
    other = deepcopy(snpe)
    other["method"] = "mutaco"
    other["decision"]["decision_policy_identifier"] = (
        "continuous_posterior_mean_physical_l1_mae"
    )
    existing = _existing()
    del existing[0]["methods"]["mutaco"]
    joined = join_compact_method_records(existing, [snpe, other])
    assert set(joined[0]["methods"]) == {"mutaco", "sbi_npe", "sbi_snpe"}


def test_consolidation_writes_a_new_zero_call_bundle_and_never_overwrites(tmp_path):
    joined = join_compact_method_records(_existing(), [_snpe()])
    output = tmp_path / "consolidated"
    manifest = write_consolidated_compact_evidence(
        output, joined, source_identities=("paper120-baseline", "snpe-append")
    )
    assert manifest["methods"] == ["mutaco", "sbi_npe", "sbi_snpe"]
    assert manifest["scientific_calls_added_by_consolidation"] == 0
    assert (output / "paired_records.jsonl").is_file()
    with pytest.raises(PaperContractError, match="new append-only path"):
        write_consolidated_compact_evidence(
            output, joined, source_identities=("paper120-baseline", "snpe-append")
        )
