from pathlib import Path
import json
import shutil
import pytest
from infrastructure.truba import shareable_evidence as evidence

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "results/reproducibility/shareable_milestone_v1"


def test_derived_evidence_retains_exact_scientific_hashes_and_scope():
    result = evidence.verify(DERIVED)
    assert result["derived_integrity_verified"]
    assert not result["raw_original_checks_verified"]
    assert result["scientific_payload_sha256"] == "12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c"
    assert result["replay_mathematics_sha256"] == "687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782"
    assert evidence.load(DERIVED / "manifest.json")["b40"]["physical_simulator_calls"] == 0


def test_rehashing_a_changed_mathematical_value_does_not_pass(tmp_path):
    altered = tmp_path / "derived"
    shutil.copytree(DERIVED, altered)
    file = altered / "b40-scientific-payload.json.gz"
    document = evidence.load(file)
    document["logical_accounting"]["committed_observations"] -= 1
    file.unlink()
    evidence.write(file, document)
    manifest = evidence.load(altered / "manifest.json")
    manifest["derived_files"][file.name]["derived_sha256"] = evidence.sha(file)
    (altered / "manifest.json").write_text(json.dumps(manifest))
    evidence.seal(altered)
    with pytest.raises(AssertionError):
        evidence.verify(altered)


def test_sanitized_reports_cannot_claim_original_packet_identity():
    for name in ("baseline-summary.json", "refactor-summary.json", "macos-diagnostic-summary.json"):
        document = evidence.load(DERIVED / name)
        assert document["derived"] and document["not_a_raw_acceptance_packet"]
        assert document["byte_identical_to_raw"] is False
        assert not evidence.PRIVATE_PATH.search(json.dumps(document))
