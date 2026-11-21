from __future__ import annotations

import torch

from mutaco import CpuParallelismProvenance, ScientificIdentity


def test_parallelism_inspection_preserves_settings_and_scientific_identity() -> None:
    identity = ScientificIdentity.create({"experiment": "parallelism-neutrality"})
    before = (torch.get_num_threads(), torch.get_num_interop_threads())
    inspected = CpuParallelismProvenance.inspect()
    after = (torch.get_num_threads(), torch.get_num_interop_threads())
    assert after == before
    assert inspected.torch_intraop_threads == before[0]
    assert inspected.torch_interop_threads == before[1]
    assert inspected.physical_cores is None or inspected.physical_cores > 0
    assert inspected.logical_processors is None or inspected.logical_processors > 0
    assert inspected.provenance()["selection_policy"] == "environment_resolved_preserved"
    assert ScientificIdentity.create({"experiment": "parallelism-neutrality"}) == identity
