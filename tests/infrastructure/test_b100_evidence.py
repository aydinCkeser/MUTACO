from pathlib import Path
import pytest
from infrastructure.truba import b100_evidence as evidence
from infrastructure.truba import b100_regression as gate
from infrastructure.truba.shareable_evidence import load, write


def failed_reference(tmp_path: Path):
    root = tmp_path / "raw"
    workers = root / "workers"
    workers.mkdir(parents=True)
    report = {"status": "diverged", "budget": 100, "target": 0, "record_pointer": "/records/0",
              "archived_observations_consumed": 16, "ambient_rng_preserved": True,
              "physical_simulator_calls": 0,
              "first_divergence": {"kind": "candidate_vector", "evaluation_index": 17,
                                   "expected": [0.1], "actual": [0.2]}}
    write(workers / "baseline-history.json.gz", report)
    gate.finish(workers, "baseline_historical_replay", 2, 2)
    return root


def test_failed_historical_reference_is_not_misreported_as_refactor_failure(tmp_path):
    raw = failed_reference(tmp_path)
    output = tmp_path / "portable"
    manifest = evidence.export(raw, output)
    assert not manifest["reference_established"]
    assert not manifest["baseline_refactor_exact_equivalence_established"]
    assert manifest["histories"]["baseline"]["first_divergence"]["evaluation_index"] == 17
    assert "refactor" not in manifest["histories"]
    assert manifest["mathematical_derivatives"] == {}
    assert evidence.verify(output, raw)["private_original_bytes_verified"]


def test_private_original_mutation_is_rejected(tmp_path):
    raw = failed_reference(tmp_path)
    output = tmp_path / "portable"
    evidence.export(raw, output)
    source = raw / "workers/baseline-history.json.gz"
    report = load(source)
    report["archived_observations_consumed"] = 17
    source.unlink()
    write(source, report)
    with pytest.raises(AssertionError):
        evidence.verify(output, raw)


def test_reported_memory_above_request_is_preserved_and_flagged(tmp_path):
    execution = tmp_path / "execution"
    execution.mkdir()
    (execution / "scheduler-accounting.txt").write_text(
        "JobIDRaw|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS\n"
        "1|debug|FAILED|1:0|598|20|8G|\n"
        "1.batch||FAILED|1:0|598|20||13927916K\n"
    )
    result = evidence.resource_summary(tmp_path)
    assert result["requested_memory"] == "8G"
    assert result["reported_peak_rss_kib"] == 13927916
    assert result["reported_peak_exceeds_request"]
    assert result["future_memory_allocation_requires_review"]
