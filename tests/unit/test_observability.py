from __future__ import annotations

import torch

from mutaco.observability import (
    MonitoringConfig,
    MonitoringLevel,
    capture_resources,
)
from mutaco.runtime import RuntimeRequest, resolve_runtime


def test_lightweight_is_default_and_resource_capture_is_rng_neutral() -> None:
    config = MonitoringConfig()
    assert config.level is MonitoringLevel.LIGHTWEIGHT
    runtime = resolve_runtime(RuntimeRequest(device="cpu"))
    before = torch.random.get_rng_state().clone()
    snapshot = capture_resources(runtime)
    assert snapshot.monitoring_level == "lightweight"
    assert snapshot.device == "cpu"
    assert snapshot.cpu_memory_source == "not_collected_in_foundation"
    assert torch.equal(torch.random.get_rng_state(), before)


def test_diagnostic_profile_is_explicit() -> None:
    config = MonitoringConfig(level=MonitoringLevel.DIAGNOSTIC_PROFILE)
    runtime = resolve_runtime(RuntimeRequest(device="cpu"))
    snapshot = capture_resources(runtime, level=config.level)
    assert snapshot.monitoring_level == "diagnostic_profile"
