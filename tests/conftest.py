"""Detached legacy sources for tests of revision-specific readers and path contracts."""
from functools import lru_cache
from pathlib import Path
import re
import subprocess

import pytest


PRE_ELIMINATION = "16d5e5c4787c29c5daf176881f57ba8d39468530"


@pytest.fixture(scope="session")
def historical_checkout(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    if tmp_path_factory.getbasetemp().resolve().is_relative_to(root):
        raise ValueError("historical checkouts require a pytest temporary directory outside the repository")
    directory = tmp_path_factory.mktemp("historical-checkouts")

    @lru_cache(maxsize=None)
    def checkout(revision: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("historical checkout requires a complete commit SHA")
        source = directory / revision
        subprocess.run(
            ["git", "clone", "-q", "--local", "--no-checkout", str(root), str(source)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "-q", "--detach", revision], check=True,
        )
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
        )
        assert actual.strip() == revision
        return source

    return checkout


@pytest.fixture(scope="session")
def legacy_checkout(historical_checkout) -> Path:
    return historical_checkout(PRE_ELIMINATION)
