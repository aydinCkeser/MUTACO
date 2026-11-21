"""Named scientific seeds independent of ambient RNG and traversal order."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TypeAlias

import torch

from mutaco.identity import ScientificIdentity, canonical_json_bytes


SEED_POLICY = "mutaco-coordinate-sha256-v1"
CoordinatePart: TypeAlias = str | int


@dataclass(frozen=True)
class SeedCoordinate:
    """Named scientific random-stream address, independent of traversal order.

    namespace separates design, fit, acquisition and simulator uses; coordinates
    identify their phase/round/repeat rather than a mutable draw counter. Address
    ordering matters, and booleans are rejected rather than aliasing integers.
    """
    namespace: str
    coordinates: tuple[CoordinatePart, ...] = ()

    def __post_init__(self) -> None:
        if not self.namespace.strip():
            raise ValueError("seed namespace must be nonempty")
        for item in self.coordinates:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise TypeError("seed coordinates must be strings or integers")


@dataclass(frozen=True)
class SeedDeriver:
    """Derive reproducible local streams from run identity, root seed and address.

    SHA-256 over the canonical payload gives a nonnegative 63-bit seed. Distinct
    named coordinates separate scientific uses without advancing ambient Torch
    RNG state; generator() creates a new locally seeded generator on the requested
    device. Matching addresses do not imply cross-platform bitwise equivalence.
    """
    scientific_identity: ScientificIdentity
    root_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("root_seed must be an integer")
        if self.root_seed < 0:
            raise ValueError("root_seed must be nonnegative")

    def seed_for(self, coordinate: SeedCoordinate) -> int:
        payload = {
            "policy": SEED_POLICY,
            "scientific_identity": self.scientific_identity.value,
            "root_seed": self.root_seed,
            "namespace": coordinate.namespace,
            "coordinates": coordinate.coordinates,
        }
        digest = sha256(canonical_json_bytes(payload)).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def generator(self, coordinate: SeedCoordinate, *, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed_for(coordinate))
        return generator
