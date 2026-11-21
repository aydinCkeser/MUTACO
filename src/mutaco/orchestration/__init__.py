"""Compatibility imports for the sequential target-matching runner."""

from mutaco.sequential_search import sequential_runner as runner
from mutaco.sequential_search import (
    AttemptStatus,
    FitRecord,
    ObservationPhase,
    ObservationRecord,
    ProgressCallback,
    ProgressKind,
    RunnerProgressEvent,
    RunnerStatus,
    SequentialRunResult,
    SequentialRunnerConfig,
    SimulationAttemptRecord,
    TerminalRecommendationContext,
    run_sequential,
)

__all__ = [
    "AttemptStatus",
    "FitRecord",
    "ObservationPhase",
    "ObservationRecord",
    "ProgressCallback",
    "ProgressKind",
    "RunnerProgressEvent",
    "RunnerStatus",
    "SequentialRunResult",
    "SequentialRunnerConfig",
    "SimulationAttemptRecord",
    "TerminalRecommendationContext",
    "run_sequential",
]
