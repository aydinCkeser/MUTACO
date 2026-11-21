from __future__ import annotations

import json
from pathlib import Path

from mutaco.diagnostics import (
    FIXED_FULL_STREAM_CRN_DIAGNOSTIC,
    FULL_FLOAT64_DIAGNOSTIC,
    IID_UNIFORM_CANDIDATES,
    IID_UNIFORM_INITIAL_DESIGN,
    MINIMUM_NORMALIZED_TARGET_MISS_DIAGNOSTIC,
    NATIVE_FLOAT32_DIAGNOSTIC,
    PAIRED_COORDINATE_DIAGNOSTIC,
    TRACE_EFFECTIVE_DIMENSION_RADIUS_DIAGNOSTIC,
    TRACE_MARGINAL_VARIANCE_DIAGNOSTIC,
)
from mutaco.policies import (
    POLICY_SURFACE,
    PolicySurface,
    policy_surface_document,
    render_policy_surface_markdown,
    surface_record,
)


ROOT = Path(__file__).parents[2]


def test_policy_surface_is_complete_unique_and_machine_readable() -> None:
    identifiers = [item.identifier for item in POLICY_SURFACE]
    assert len(identifiers) == len(set(identifiers))
    assert {item.value for item in PolicySurface} == {
        "active_core",
        "active_specialized_core",
        "compatibility",
        "diagnostic",
        "internal",
    }
    assert surface_record("shared_isotropic_gaussian").surface is PolicySurface.COMPATIBILITY
    assert surface_record("high_compute").surface is PolicySurface.COMPATIBILITY
    assert surface_record("graph_m20_sequential_recovery_v1").surface is PolicySurface.ACTIVE_SPECIALIZED_CORE
    assert surface_record("continuous_terminal_optimizer").surface is PolicySurface.INTERNAL


def test_generated_policy_surface_files_match_registry() -> None:
    json_path = ROOT / "scientific_contracts/mutaco_d_contract/POLICY_SURFACE.json"
    markdown_path = ROOT / "scientific_contracts/mutaco_d_contract/POLICY_SURFACE.md"
    assert json.loads(json_path.read_text(encoding="utf-8")) == policy_surface_document()
    assert markdown_path.read_text(encoding="utf-8") == render_policy_surface_markdown()


def test_diagnostic_namespace_preserves_exact_identifiers() -> None:
    assert IID_UNIFORM_INITIAL_DESIGN.value == IID_UNIFORM_CANDIDATES.value == "iid_uniform"
    assert TRACE_MARGINAL_VARIANCE_DIAGNOSTIC.value == "trace_marginal_variance"
    assert NATIVE_FLOAT32_DIAGNOSTIC.value == "native_float32"
    assert FULL_FLOAT64_DIAGNOSTIC.value == "full_float64_diagnostic"
    assert TRACE_EFFECTIVE_DIMENSION_RADIUS_DIAGNOSTIC == "trace_effective_dimension_radius_diagnostic_v1"
    assert MINIMUM_NORMALIZED_TARGET_MISS_DIAGNOSTIC == "minimum_normalized_target_miss_diagnostic_v1"
    assert PAIRED_COORDINATE_DIAGNOSTIC == "paired_coordinate_diagnostic_v1"
    assert FIXED_FULL_STREAM_CRN_DIAGNOSTIC == "fixed_full_stream_crn_diagnostic_v1"
