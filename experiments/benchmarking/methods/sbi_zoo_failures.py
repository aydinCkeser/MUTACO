"""Typed, no-fallback failures for the frozen Paper-120 SBI-zoo extension."""

from __future__ import annotations

from experiments.contracts import PaperContractError


class SbiZooError(RuntimeError):
    """Base class for frozen SBI-zoo implementation failures."""


class DatasetIntegrityError(SbiZooError):
    """Canonical shared-dataset bytes or metadata failed validation."""


class DatasetSerializationError(SbiZooError):
    """A shared dataset could not be persisted canonically."""


class ScientificBudgetViolation(SbiZooError):
    """A write-ahead scientific-call ledger would exceed its frozen cap."""


class TrainingFailure(SbiZooError):
    """Pinned NLE/NRE construction or training failed."""


class PosteriorConstructionFailure(SbiZooError):
    """The configured NLE/NRE posterior could not be built."""


class MCMCInitializationFailure(SbiZooError):
    """The frozen proposal initialization failed or returned invalid state."""


class MCMCSamplingFailure(SbiZooError):
    """The frozen sampler failed after initialization."""


class InsufficientPosteriorSamples(SbiZooError):
    """Exactly 10,000 ordered valid posterior samples were not produced."""


class NonfinitePosteriorScore(SbiZooError):
    """No finite physical-coordinate posterior score was available."""


class PosteriorSupportViolation(SbiZooError):
    """A posterior candidate violated the frozen closed physical support."""


class SbiZooScoringContractError(SbiZooError, PaperContractError):
    """A method, score API, identity, or scoring path violated the frozen contract."""


class PhysicalSimulatorTripwireTriggered(SbiZooError):
    """Forbidden construction or invocation of the physical simulator occurred."""
