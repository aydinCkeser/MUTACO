"""Verify recommendation ownership across public imports and historical pickle globals."""

import importlib
import os
import pickle
import subprocess
import sys

import pytest


LEGACY_OBSERVED = "mutaco.recommendation.observed_physical_mae"
CANONICAL_OBSERVED = "mutaco.comparators.observed_physical_mae_recommendation"


@pytest.mark.parametrize("first", [LEGACY_OBSERVED, CANONICAL_OBSERVED])
def test_observed_comparator_import_order_and_patch_binding(first, tmp_path):
    script = """
import importlib, sys
from unittest.mock import patch
first, legacy_name, canonical_name = sys.argv[1:]
first_module = importlib.import_module(first)
legacy = importlib.import_module(legacy_name)
canonical = importlib.import_module(canonical_name)
from mutaco import api, recommendation
assert legacy is canonical is first_module is recommendation.observed_physical_mae
assert canonical.recommend_best_observed_physical_mae is api.recommend_best_observed_physical_mae
with patch.object(legacy, 'descriptor_for') as patched:
    assert canonical.descriptor_for is patched
"""
    subprocess.run(
        [sys.executable, "-c", script, first, LEGACY_OBSERVED, CANONICAL_OBSERVED],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(("legacy", "name"), [
    ("mutaco.recommendation.policies", "EvaluatedDesigns"),
    ("mutaco.recommendation.physical_l2", "MeanOnlyPosterior"),
    ("mutaco.recommendation.physical_l2", "PhysicalL2RecommendationError"),
])
def test_shared_contract_keeps_legacy_pickle_global(legacy, name):
    from mutaco import recommendation
    from mutaco.recommendation import recommendation_contracts

    value = getattr(recommendation_contracts, name)
    assert getattr(recommendation, name) is getattr(importlib.import_module(legacy), name) is value
    assert value.__module__ == legacy
    assert pickle.loads(f"c{legacy}\n{name}\n.".encode("ascii")) is value
    assert pickle.loads(pickle.dumps(value)) is value
    assert recommendation.PhysicalMAERecommendationError.__bases__ == (
        recommendation_contracts.PhysicalL2RecommendationError,
    )


@pytest.mark.parametrize("first", ["old", "new"])
def test_terminal_module_aliases_pickle_globals_and_overlay(first, tmp_path):
    """Check both import orders and the optimizer binding used by terminal overlays."""
    script = """
import importlib, pickle, sys
from types import ModuleType
from unittest.mock import patch

prefix = 'mutaco.recommendation.'
modules = [
    ('_continuous', '_continuous_terminal_optimizer', 'optimize_continuous_physical'),
    ('physical_l2', 'comparators.physical_squared_l2_recommendation', 'PhysicalL2RecommendationResult'),
    ('physical_mae', 'physical_mae_terminal_recommendation', 'PhysicalMAERecommendationError'),
    ('policies', 'evaluated_design_recommendation', 'RecommendationResult'),
]
for old_name, new_name, name in modules:
    first_name = old_name if sys.argv[1] == 'old' else new_name
    initial = importlib.import_module(prefix + first_name)
    old = importlib.import_module(prefix + old_name)
    new = importlib.import_module(prefix + new_name)
    assert old is new is initial
    package = importlib.import_module('mutaco.recommendation')
    assert getattr(package, old_name) is old
    value = getattr(new, name)
    assert value.__module__ == prefix + old_name
    assert pickle.loads(pickle.dumps(value)) is value
    if old_name != '_continuous':
        assert getattr(package, name) is value
    with patch.object(old, name) as patched:
        assert getattr(new, name) is patched

# The exact-source bootstrap inserts the optimizer at its old module name,
# then reloads physical_mae to bind its score callbacks to that optimizer.
overlay = ModuleType(prefix + '_continuous')
overlay.optimize_continuous_physical = object()
sys.modules[overlay.__name__] = overlay
mae = importlib.reload(sys.modules[prefix + 'physical_mae'])
assert mae.optimize_continuous_physical is overlay.optimize_continuous_physical
"""
    subprocess.run(
        [sys.executable, "-c", script, first],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        cwd=tmp_path,
        check=True,
    )
