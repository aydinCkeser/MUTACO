"""Import, serialization and patch compatibility across the acquisition/comparator boundary."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys

import pytest


MODULE_PAIRS = (
    ("mutaco.acquisition.geometry", "mutaco.acquisition.confidence_geometry"),
    ("mutaco.acquisition.objectives", "mutaco.acquisition.acquisition_objectives"),
    ("mutaco.acquisition.proposal", "mutaco.acquisition.candidate_proposal"),
    ("mutaco.acquisition.nei", "mutaco.comparators.composite_mc_nei"),
    ("mutaco.diagnostics.policies", "mutaco.comparators.diagnostic_policy_aliases"),
)


@pytest.mark.parametrize("canonical_first", [False, True])
def test_old_and_new_paths_work_in_fresh_interpreters(canonical_first):
    # Import the comparator package first when exercising the new public route.
    pairs = tuple((b, a) for a, b in reversed(MODULE_PAIRS)) if canonical_first else MODULE_PAIRS
    script = """
import importlib
import sys
for first, second in pairs:
    module = importlib.import_module(first)
    assert importlib.import_module(second) is module
    for path in (first, second):
        parent, name = path.rsplit(".", 1)
        assert getattr(sys.modules[parent], name) is module
from mutaco import api
from mutaco import acquisition, NEIConfig
from mutaco.comparators import composite_mc_nei
assert NEIConfig is api.NEIConfig is acquisition.NEIConfig is composite_mc_nei.NEIConfig
"""
    subprocess.run(
        [sys.executable, "-c", f"pairs = {pairs!r}\n{script}"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=True,
    )


@pytest.mark.parametrize(("old_module", "new_module", "symbol"), [
    (*MODULE_PAIRS[0], "FixedBeta"),
    (*MODULE_PAIRS[1], "MatchedBarrierObjective"),
    (*MODULE_PAIRS[2], "propose_candidate"),
    (*MODULE_PAIRS[3], "NEIAcquisitionError"),
])
def test_historical_pickle_globals_and_candidate_patch_points(old_module, new_module, symbol, monkeypatch):
    authoritative = importlib.import_module(new_module)
    old = importlib.import_module(old_module)
    value = getattr(authoritative, symbol)
    assert value.__module__ == old_module
    assert pickle.loads(f"c{old_module}\n{symbol}\n.".encode("ascii")) is value
    assert pickle.loads(pickle.dumps(value)) is value
    monkeypatch.setattr(old, symbol, object())
    assert getattr(old, symbol) is getattr(authoritative, symbol)


def test_diagnostic_aliases_retain_enum_objects():
    from mutaco.acquisition.candidate_proposal import RawCandidatePolicy
    from mutaco.comparators import diagnostic_policy_aliases
    from mutaco.diagnostics import IID_UNIFORM_CANDIDATES

    assert diagnostic_policy_aliases.IID_UNIFORM_CANDIDATES is IID_UNIFORM_CANDIDATES
    assert IID_UNIFORM_CANDIDATES is RawCandidatePolicy.IID_UNIFORM
