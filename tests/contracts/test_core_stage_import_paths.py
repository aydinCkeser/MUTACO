"""Public stage names retain one module, shared patch state and legacy pickle globals."""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import pickle

import pytest


ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    ("response.transform", "response.response_standardization", "FrozenResponseTransform"),
    ("surrogate.exact_multitask", "surrogate.exact_multitask_gp", "GPFitConfig"),
    ("posterior.summary", "posterior.posterior_summary", "LatentPosteriorSummary"),
    ("simulator.protocol", "simulator.simulator_adapter_contracts", "SimulationContext"),
    ("orchestration.runner", "sequential_search.sequential_runner", "run_sequential"),
)

IDENTITY_PATHS = (
    ("identity.canonical", "identity.canonical_identity_hashing", "ScientificIdentity"),
    ("identity.metadata", "identity.strict_metadata_validation", "StrictMetadataError"),
    ("identity.sequential", "identity.sequential_scientific_identity", "SequentialScientificIntent"),
    ("randomness.seeds", "randomness.deterministic_seed_derivation", "SeedDeriver"),
)


@pytest.mark.parametrize(("legacy", "current", "public"), STAGES + IDENTITY_PATHS)
def test_stage_module_global_and_patch_identity(legacy, current, public, monkeypatch):
    old_name, new_name = "mutaco." + legacy, "mutaco." + current
    old = importlib.import_module(old_name)
    new = importlib.import_module(new_name)
    symbol = getattr(old, public)
    assert new is old
    assert new.__name__ == new_name
    assert Path(new.__file__).name == current.rsplit(".", 1)[1] + ".py"
    assert getattr(importlib.import_module(old_name.rsplit(".", 1)[0]), public) is symbol
    root_api = importlib.import_module("mutaco.api")
    assert public in root_api.__all__
    assert getattr(root_api, public) is symbol
    assert getattr(importlib.import_module("mutaco"), public) is symbol
    assert symbol.__module__ == old_name
    # Protocol 0 resolves the recorded original module and global directly.
    assert pickle.loads(f"c{old_name}\n{public}\n.".encode("ascii")) is symbol
    assert pickle.loads(pickle.dumps(symbol)) is symbol
    with monkeypatch.context() as patch:
        patch.setattr(old, public, object())
        assert getattr(new, public) is getattr(old, public)


@pytest.mark.parametrize(("legacy", "current", "public"), STAGES)
def test_legacy_source_readers_still_find_the_same_definitions(legacy, current, public):
    def source(module):
        return ROOT / "src/mutaco" / Path(*module.split(".")).with_suffix(".py")

    # Old import names resolve to the canonical Python source even in a wheel.
    # The two held-out declarations additionally hash the literal protocol.py path.
    module = importlib.import_module("mutaco." + legacy)
    b = source(current)
    assert Path(inspect.getsourcefile(module)).read_bytes() == b.read_bytes()
    if legacy == "simulator.protocol":
        assert source(legacy).read_bytes() == b.read_bytes()
    else:
        assert not source(legacy).exists()
    assert public in {
        n.name for n in ast.parse(b.read_bytes()).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    }


def test_identity_and_seeds_match_audited_source_values():
    # Captured from 9f3ce189, before moving the modules. Order and signed zero matter.
    import torch

    from mutaco.identity import ExecutionIdentity, ScientificIdentity, sequential_observation_identity
    from mutaco.randomness import SeedCoordinate, SeedDeriver

    before = torch.random.get_rng_state().clone()
    scientific = ScientificIdentity.create({"simulator": "analytic", "target_id": "target-7", "root_seed": 91})
    execution = ExecutionIdentity.create(scientific, {"device": "cpu", "dtype": "float64"})
    assert scientific.value == "scientific-v1-f4ffe5ce6e945e3fa58a76c0d3df91953e4494c85aded31f8b05d1b4271744cb"
    assert execution.value == "execution-v1-48c87380caea8399f16276da91cab60c8bd0d26307004ec4e002e3683e475320"
    deriver = SeedDeriver(scientific, 91)
    for address, expected in (
        (SeedCoordinate("initial-design", (3,)), 6055010299371460347),
        (SeedCoordinate("simulator-observation-noise", ("search", 4, 0)), 8261592690764353720),
        (SeedCoordinate("simulator-observation-noise", ("search", 4, 1)), 1437934086348189473),
    ):
        assert deriver.seed_for(address) == expected
        assert deriver.generator(address, device=torch.device("cpu")).initial_seed() == expected
    observation = sequential_observation_identity(
        scientific_identity=scientific, global_logical_index=16, sequential_round_index=0,
        candidate_id="candidate", theta=torch.tensor([-0.0, 0.5, 1], dtype=torch.float64),
    )
    assert observation == "mutaco-sequential-observation-v1-b0fbab118723268eed7271ef724d697dabb2747383fc2f26df2a5684a07ce539"
    assert torch.equal(torch.random.get_rng_state(), before)
