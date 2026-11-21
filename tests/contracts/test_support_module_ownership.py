"""Compatibility of scientific declarations, file writers and runtime resources."""

from __future__ import annotations

import hashlib
import importlib
import os
import pickle
import subprocess
import sys

import pytest


MODULE_PAIRS = (
    ("mutaco.design.initial", "mutaco.design.initial_design"),
    ("mutaco.policies.registry", "mutaco.policies.scientific_policy_registry"),
    ("mutaco.policies.surface", "mutaco.policies.policy_surface"),
    ("mutaco.runtime.parallelism", "mutaco.device_and_resources.cpu_parallelism_provenance"),
    ("mutaco.runtime.resolution", "mutaco.device_and_resources.device_and_precision_resolution"),
    ("mutaco.observability.monitoring", "mutaco.device_and_resources.resource_monitoring"),
    ("mutaco.artifacts.schemas", "mutaco.file_management.artifact_documents_and_manifests"),
    ("mutaco.atomic_io", "mutaco.file_management.atomic_file_writes"),
    ("mutaco.types.specifications", "mutaco.parameter_and_response.parameter_response_specifications"),
)


@pytest.mark.parametrize("legacy_first", [True, False])
def test_fresh_process_uses_one_module_per_owner(legacy_first: bool) -> None:
    pairs = MODULE_PAIRS if legacy_first else [(new, old) for old, new in MODULE_PAIRS]
    script = """
import importlib
import sys

for first, second in pairs:
    module = importlib.import_module(first)
    assert module is importlib.import_module(second)
    for path in (first, second):
        package, name = path.rsplit(".", 1)
        assert getattr(sys.modules[package], name) is module

from mutaco import ParameterSpec
from mutaco import api, parameter_and_response, types
assert ParameterSpec is api.ParameterSpec is types.ParameterSpec is parameter_and_response.ParameterSpec
"""
    subprocess.run(
        [sys.executable, "-c", f"pairs = {pairs!r}\n{script}"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=True,
    )


def test_patches_reach_the_writers_implementation(monkeypatch, tmp_path) -> None:
    from mutaco import atomic_io
    from mutaco.file_management import atomic_file_writes

    writes = []
    monkeypatch.setattr(atomic_io, "atomic_write_bytes", lambda path, data: writes.append((path, data)))
    atomic_file_writes.atomic_write_text(tmp_path / "unicode", "α")
    assert writes == [(tmp_path / "unicode", b"\xce\xb1")]


def test_historical_pickle_names_and_bytes() -> None:
    from mutaco import ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec
    from mutaco.file_management.artifact_documents_and_manifests import ArtifactValidationError

    # Digests of protocol-4 pickles captured on 9f3ce189, before relocation.
    params = ParameterSpec((
        ParameterAxis("coupling", 0.0, 1.0),
        ParameterAxis("noise", 0.125, 2.0, unit="s", transformation="log"),
    ))
    responses = ResponseSpec((ResponseAxis("omega", "Hz"), ResponseAxis("angle", "rad")))
    samples = (
        ((params, responses), "71ecfc61dbebd4b89cf28f53e9576ee628c77236a30be4abcf922c1271ae1902"),
        (ArtifactValidationError("mismatch"), "f16e24912f0fcee1af34b865329d12a35f561f94a0bf7efd8cf95aea7fa053c8"),
    )
    for value, original_digest in samples:
        payload = pickle.dumps(value, protocol=4)
        assert hashlib.sha256(payload).hexdigest() == original_digest
        restored = pickle.loads(payload)
        if isinstance(value, ArtifactValidationError):
            assert type(restored) is ArtifactValidationError
            assert restored.args == value.args
        else:
            assert restored == value
    for old, new in MODULE_PAIRS:
        legacy = importlib.import_module(old)
        for name, obj in vars(importlib.import_module(new)).items():
            if getattr(obj, "__module__", None) == old:
                assert pickle.loads(pickle.dumps(obj)) is getattr(legacy, name)


def test_monitoring_and_policy_patches_reach_original_call_sites(monkeypatch) -> None:
    from mutaco.device_and_resources import device_and_precision_resolution
    from mutaco.observability import monitoring
    from mutaco.policies import policy_surface, scientific_policy_registry, surface
    from mutaco.runtime import resolution

    assert monitoring.RuntimeDecision is device_and_precision_resolution.RuntimeDecision
    assert scientific_policy_registry.surface_record is policy_surface.surface_record

    monkeypatch.setattr(resolution.torch.cuda, "is_available", lambda: False)
    request = resolution.RuntimeRequest(device="cuda")
    with pytest.raises(resolution.DeviceUnavailableError, match="explicit CUDA"):
        device_and_precision_resolution.resolve_runtime(request)
    monkeypatch.setattr(surface, "POLICY_SURFACE", ())
    assert policy_surface.policy_surface_document()["records"] == []
