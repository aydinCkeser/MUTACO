"""Sequential target matching using the manuscript's four stages."""

from mutaco._legacy_module import bind_legacy_module as _bind_legacy_module

from . import sequential_runner
from .sequential_runner import (
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

_bind_legacy_module(sequential_runner, "mutaco.orchestration.runner")

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
