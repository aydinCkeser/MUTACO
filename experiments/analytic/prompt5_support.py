"""Compatibility imports for the former Prompt 5 support module."""

from experiments.analytic_target_matching._support import (  # noqa: F401
    LEGACY_SCHEMA,
    AcceptanceProfile,
    build_manifest,
    load_config,
    prepare_profiles,
    run_and_assess_profile,
    write_json,
)

SCHEMA = LEGACY_SCHEMA
