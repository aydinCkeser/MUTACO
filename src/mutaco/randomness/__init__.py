"""Coordinate-addressed random seeds."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import deterministic_seed_derivation
from .deterministic_seed_derivation import SEED_POLICY, SeedCoordinate, SeedDeriver

seeds = _bind_legacy_module(deterministic_seed_derivation, __name__ + ".seeds")

__all__ = ["SEED_POLICY", "SeedCoordinate", "SeedDeriver"]
