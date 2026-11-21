"""Shared call-count values and validation exception; no experiment policy.

Legacy module identities deliberately remain stable for historical pickles.
Paper method registries, decision rules, budgets and assessment stay with their
owning programs. Importing these contracts needs only the standard library.
"""
from __future__ import annotations

from dataclasses import dataclass


class PaperContractError(ValueError):
    """A paper declaration or persisted record violates scientific identity."""


@dataclass(frozen=True)
class PhysicalCallCounts:
    """Physical attempt accounting for one ledger category."""

    attempted: int
    successful: int
    failed: int = 0
    retries: int = 0

    def __post_init__(self) -> None:
        values = (self.attempted, self.successful, self.failed, self.retries)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in values
        ):
            raise PaperContractError("physical call counts must be nonnegative integers")
        if self.attempted != self.successful + self.failed:
            raise PaperContractError("attempted calls must equal successful plus failed calls")
        if self.retries > self.attempted:
            raise PaperContractError("retry calls cannot exceed attempted calls")

    def document(self) -> dict[str, int]:
        return {
            "attempted_calls": self.attempted,
            "successful_calls": self.successful,
            "failed_calls": self.failed,
            "retry_calls": self.retries,
        }


# Keep persisted GLOBAL references, exception identities and representations.
PaperContractError.__module__ = "experiments.paper.common.validation"
PhysicalCallCounts.__module__ = "experiments.paper.common.protocol_v3"

__all__ = ("PaperContractError", "PhysicalCallCounts")
