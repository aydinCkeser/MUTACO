from __future__ import annotations

import pytest
import torch

from mutaco.identity import (
    ExecutionIdentity,
    ScientificIdentity,
    canonical_json_bytes,
    stable_hash,
)
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.simulator import SimulationContext


def test_canonical_hash_is_order_independent_but_value_sensitive() -> None:
    first = {"b": [2, 3], "a": {"x": 1}}
    second = {"a": {"x": 1}, "b": (2, 3)}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert stable_hash("config-v1", first) == stable_hash("config-v1", second)
    assert stable_hash("config-v1", first) != stable_hash("config-v1", {"a": 2})


def test_canonical_json_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises(TypeError, match="string keys"):
        canonical_json_bytes({1: "not allowed"})
    with pytest.raises(TypeError, match="unsupported"):
        canonical_json_bytes({"tensor": torch.tensor(1.0)})


def test_execution_identity_does_not_change_scientific_identity_or_seeds() -> None:
    scientific = ScientificIdentity.create(
        {"simulator": "analytic", "target_id": "target-7", "root_seed": 91}
    )
    cpu = ExecutionIdentity.create(scientific, {"device": "cpu", "dtype": "float64"})
    cuda = ExecutionIdentity.create(scientific, {"device": "cuda:0", "dtype": "float64"})
    assert cpu.value != cuda.value
    assert cpu.scientific_identity == cuda.scientific_identity == scientific.value
    coordinate = SeedCoordinate("simulator-observation-noise", ("search", 4, 0))
    assert SeedDeriver(scientific, 91).seed_for(coordinate) == SeedDeriver(
        ScientificIdentity(scientific.value), 91
    ).seed_for(coordinate)


def test_coordinate_seeds_are_order_independent_and_namespaced() -> None:
    scientific = ScientificIdentity.create({"study": "seed-test"})
    deriver = SeedDeriver(scientific, 12)
    first = SeedCoordinate("initial-design", (3,))
    second = SeedCoordinate("simulator-observation-noise", ("search", 3, 0))
    first_seed = deriver.seed_for(first)
    second_seed = deriver.seed_for(second)
    assert first_seed == deriver.seed_for(first)
    assert second_seed == deriver.seed_for(second)
    assert first_seed != second_seed


def test_dedicated_generator_does_not_consume_ambient_torch_rng() -> None:
    scientific = ScientificIdentity.create({"study": "generator-test"})
    deriver = SeedDeriver(scientific, 5)
    before = torch.random.get_rng_state().clone()
    generator = deriver.generator(SeedCoordinate("gp-landscape", ("state",)), device=torch.device("cpu"))
    _ = torch.randn(20, generator=generator)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_retry_attempt_reuses_randomness_address() -> None:
    first = SimulationContext("science-1", "search", 8, 2, 1234, attempt_index=0)
    retry = SimulationContext("science-1", "search", 8, 2, 1234, attempt_index=1)
    new_repeat = SimulationContext("science-1", "search", 8, 3, 5678, attempt_index=0)
    assert first.randomness_address() == retry.randomness_address()
    assert first.randomness_address() != new_repeat.randomness_address()
