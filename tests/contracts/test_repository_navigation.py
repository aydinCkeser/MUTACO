"""Offline checks for the supported navigation surface, not every old archive."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import shlex
from urllib.parse import unquote

import pytest


ROOT = Path(__file__).resolve().parents[2]
GUIDES = (
    "README.md", "experiments/README.md",
    "src/mutaco/README.md", "infrastructure/README.md",
    "docs/DOCUMENTATION_AUTHORITY.md", "scientific_archive/project_records/README.md", "docs/README.md",
    "experiments/paper/README.md", "experiments/contracts/README.md",
    "experiments/graph_inversion/README.md",
    "experiments/paper/figures/README.md",
    "results/README.md", "scientific_contracts/README.md", "scientific_archive/README.md",
    "reproducibility/README.md", "simulators/README.md",
    "src/mutaco/sequential_search/README.md", "src/mutaco/comparators/README.md",
    "infrastructure/truba/README.md",
    "experiments/paper/exploration/README.md",
    "infrastructure/local/README.md",
    "docs/decisions/legacy_checkout_paths.md",
)


def headings(text: str) -> set[str]:
    # The linked headings use ordinary words, punctuation and spaces only.
    return {re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
            for title in re.findall(r"^#+\s+(.+)$", text, re.M)}


def check_links(document: Path) -> None:
    for destination in re.findall(r"\]\(([^)]+)\)", document.read_text()):
        if "://" in destination or destination.startswith("mailto:"):
            continue  # This check is deliberately offline.
        relative, _, anchor = unquote(destination).partition("#")
        target = document.parent / relative if relative else document
        if not target.exists():
            raise ValueError(f"{document.name}: missing target {destination}")
        if anchor and (not target.is_file() or anchor not in headings(target.read_text())):
            raise ValueError(f"{document.name}: missing heading {destination}")


@pytest.mark.parametrize("relative", GUIDES)
def test_navigation_links_and_headings_exist(relative: str) -> None:
    check_links(ROOT / relative)


def test_archived_readiness_audit_references_resolve_by_source_identity() -> None:
    # A frozen report cites the paths at its publication date. Its current
    # reader can locate the same bytes using the versioned migration ledger.
    import hashlib
    import json

    ledger = json.loads((ROOT / "docs/decisions/repository_migration_manifest.json").read_text())
    by_path = {row["base_path"]: row for row in ledger["files"]}
    document = ROOT / "scientific_archive/validation_history/refactor_validation/submission_readiness_audit.md"
    row = by_path["scientific/refactor_validation/submission_readiness_audit.md"]
    assert hashlib.sha256(document.read_bytes()).hexdigest() == row["reviewed_sha256"]
    for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
        target = (document.parent / link).resolve()
        if not target.is_file():
            relative = target.relative_to(ROOT).as_posix()
            assert (ROOT / by_path[relative]["physical_path"]).is_file(), link


def test_link_check_rejects_missing_files_and_headings(tmp_path: Path) -> None:
    document = tmp_path / "README.md"
    document.write_text("[missing](absent.md)")
    with pytest.raises(ValueError, match="missing target"):
        check_links(document)
    document.write_text("# Present\n[missing](#absent)")
    with pytest.raises(ValueError, match="missing heading"):
        check_links(document)


def test_catalog_links_each_research_namespace() -> None:
    catalog = (ROOT / "experiments/README.md").read_text()
    for child in (ROOT / "experiments").iterdir():
        if child.is_dir() and (child / "__init__.py").is_file():
            assert re.search(r"\]\(" + re.escape(child.name) + r"/", catalog), child.name


def test_map_names_existing_scientific_symbols() -> None:
    expected = {
        "sequential_search/sequential_runner.py": (
            "run_sequential", "_validate_pre_run_options", "_initial_design",
            "_collect_initial_observations", "_evaluate", "_prepare_frozen_responses",
            "_prepare_gp_training_data", "_record_gp_fit_result", "fit_version",
            "_standardize_and_commit_sequential_observation",
        ),
        "design/initial_design.py": ("generate_initial_design",),
        "response/response_standardization.py": ("FrozenResponseTransform",),
        "surrogate/exact_multitask_gp.py": (
            "_build_model", "fit_exact_multitask_gp", "fit_exact_multitask_gp_with_recovery",
            "_summarize_fit_optimizer_result",
        ),
        "acquisition/candidate_proposal.py": (
            "propose_candidate", "_screen_and_rank_raw_candidates", "_barrier_restart",
            "_exploration_restart", "_select_nonworsening_refined_outcome",
            "recover_best_ranked_unevaluated_raw",
        ),
        "recommendation/physical_mae_terminal_recommendation.py": ("recommend_continuous_physical_mae",),
    }
    guide = (ROOT / "src/mutaco/README.md").read_text()
    for relative, symbols in expected.items():
        tree = ast.parse((ROOT / "src/mutaco" / relative).read_text())
        definitions = {node.name for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for symbol in symbols:
            assert symbol in definitions and symbol in guide, (relative, symbol)


def test_root_test_commands_select_the_same_existing_checks() -> None:
    text = (ROOT / "README.md").read_text()
    commands = [line for line in text.splitlines() if "run_pytest." in line]
    assert len(commands) == 2
    selections = []
    for command in commands:
        # Paths/arguments after the platform launcher have the same syntax.
        arguments = shlex.split(command.split(" -q ", 1)[1])
        assert all((ROOT / name.split("::", 1)[0]).is_file() for name in arguments)
        selections.append(arguments)
    assert selections[0] == selections[1]
    assert (
        "tests/experiments/test_exploration_entrypoint_safety.py::"
        "test_documented_cli_help_has_no_physical_calls"
    ) in selections[0]
