from __future__ import annotations

from pathlib import Path

import pytest

from experiments.paper.figures.common.style_registry import (
    REQUIRED_ROLES,
    load_paper_style,
)


ROOT = Path(__file__).resolve().parents[2]
GRAMMAR = ROOT / "experiments/paper/figures/styles/visual_grammar.toml"


def test_authoritative_visual_grammar_resolves_complete_semantic_roles():
    style = load_paper_style(GRAMMAR)
    assert style.profile_id == "aps_full_width"
    assert (style.width_inches, style.height_inches) == (7.0, 2.9)
    assert set(style.roles) == REQUIRED_ROLES
    assert style.role("true")["markerfacecolor"] == "white"
    assert style.role("mutaco")["linestyle"] == ":"
    assert style.role("npe")["marker"] == "^"


def test_unknown_or_incomplete_roles_are_rejected(tmp_path):
    text = GRAMMAR.read_text(encoding="utf-8")
    rogue = tmp_path / "rogue.toml"
    rogue.write_text(text + '\n[roles.rogue]\ncolor = "black"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="visual roles mismatch"):
        load_paper_style(rogue)

    incomplete = tmp_path / "incomplete.toml"
    incomplete.write_text(
        text.replace("[roles.true]\ncolor = \"#111111\"", "[roles.true]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires a valid color"):
        load_paper_style(incomplete)


def test_malformed_values_and_unknown_overrides_are_rejected(tmp_path):
    text = GRAMMAR.read_text(encoding="utf-8")
    malformed = tmp_path / "malformed.toml"
    malformed.write_text(text.replace('alpha = 0.65', 'alpha = 1.5'), encoding="utf-8")
    with pytest.raises(ValueError, match="alpha must lie"):
        load_paper_style(malformed)
    with pytest.raises(ValueError, match="unknown overridden visual roles"):
        load_paper_style(GRAMMAR, role_overrides={"unknown": {"color": "black"}})


def test_explicit_override_is_resolved_and_preserved():
    style = load_paper_style(
        GRAMMAR, role_overrides={"mutaco": {"line_width": 2.0}}
    )
    assert style.role("mutaco")["linewidth"] == 2.0
    provenance = style.provenance(ROOT)
    assert provenance["role_overrides"] == {"mutaco": {"line_width": 2.0}}
    assert provenance["resolved_roles"]["mutaco"]["linewidth"] == 2.0
