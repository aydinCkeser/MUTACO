"""Generate the policy-surface compatibility matrix without scientific calls."""

from __future__ import annotations

import json
from pathlib import Path

from mutaco.policies import policy_surface_document, render_policy_surface_markdown


CONTRACT_DIR = Path(__file__).absolute().parent
JSON_PATH = CONTRACT_DIR / "POLICY_SURFACE.json"
MARKDOWN_PATH = CONTRACT_DIR / "POLICY_SURFACE.md"


def main() -> None:
    JSON_PATH.write_text(
        json.dumps(policy_surface_document(), indent=2) + "\n", encoding="utf-8"
    )
    MARKDOWN_PATH.write_text(render_policy_surface_markdown(), encoding="utf-8")
    print(JSON_PATH)
    print(MARKDOWN_PATH)


if __name__ == "__main__":
    main()
