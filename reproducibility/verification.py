"""Read-only API backed by the original verifier functions.

The archived checksums bind the original verifier sources. Their modern
bootstrap paths follow the repository layout; their comparisons stay the same.
These are function bindings, not copied algorithms or wrappers around numerical
execution. Private originals are read only when explicitly supplied to the
existing API; portable-package checks alone do not certify those originals.
"""
from infrastructure.truba.refactor_reference import (
    canonical, digest, numerical_payload, environment_signature, compare_files,
)
from infrastructure.truba.b100_regression import (
    first_difference, test_outcomes, compare_histories as compare_b100_histories,
)
from infrastructure.truba.readability_regression import (
    verify_inputs, history_gate, compare_history, tests_gate,
    HISTORICAL_TEST_EXPECTATION, CORRECTED_TEST_EXPECTATION,
)
from infrastructure.truba.fit_event_evidence import expected_fit_events, compare_fit_events
from infrastructure.truba.shareable_evidence import load, sha, verify as verify_b40
from infrastructure.truba.b100_evidence import verify as verify_b100

__all__ = [
    "canonical", "digest", "numerical_payload", "environment_signature", "compare_files",
    "first_difference", "test_outcomes", "compare_b100_histories", "verify_inputs",
    "history_gate", "compare_history", "tests_gate", "expected_fit_events", "compare_fit_events",
    "load", "sha", "verify_b40", "verify_b100", "HISTORICAL_TEST_EXPECTATION",
    "CORRECTED_TEST_EXPECTATION",
]
