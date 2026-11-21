"""Keep original B100 execution provenance separate from current preparation bytes."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.paper.common import scalar_primary_b100_archive_cli as archive
from experiments.paper.historical_workflows import scalar_primary_b100_preparation_audit_cli as prep
from experiments.paper.protocol.validation import PaperContractError


ROOT = Path(__file__).resolve().parents[2]


def test_b100_receipt_and_both_sealed_manifests_bind_the_original_profile() -> None:
    receipt = json.loads((ROOT / prep.SOURCE_RECEIPT).read_bytes())
    prepared = json.loads((ROOT / prep.OUTPUT / "manifest.json").read_bytes())
    archived = json.loads((ROOT / archive.OUTPUT / "manifest.json").read_bytes())["scientific_source"]
    assert archived["commit"] == receipt["source_commit"]
    assert archived["profile"] == receipt["profile"] == prepared["profile"]
    current = (ROOT / prep.PROFILE).read_bytes()
    original = prep.verified_historical_profile(ROOT, current)
    assert archive._archived_profile_sha256(ROOT, prepared) == sha256(original).hexdigest()
    assert sha256(original).hexdigest() == receipt["profile_sha256"] == archived["profile_sha256"]
    assert original != current
    # A new preparation must stamp its own actual profile and resulting identity.
    live = json.loads(prep._content(ROOT, historical=False)["manifest.json"])
    assert live["profile_sha256"] == sha256(current).hexdigest()
    assert live["package_identity"] != prepared["package_identity"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prep.build(repository_root=ROOT, check=False)


def test_b100_scheduler_and_preparation_tampering_are_rejected() -> None:
    current = (ROOT / prep.PROFILE).read_bytes()
    wrong_workers = current.replace(b"MUTACO_SCALAR_PRIMARY_WORKERS=120", b"MUTACO_SCALAR_PRIMARY_WORKERS=119")
    assert wrong_workers != current
    with pytest.raises(PaperContractError, match="differs beyond the approved cohort path"):
        prep.verified_historical_profile(ROOT, wrong_workers)
    with pytest.raises(PaperContractError, match="differs from the preparation freeze"):
        archive._archived_profile_sha256(ROOT, {"profile": prep.PROFILE, "profile_sha256": sha256(current).hexdigest()})
