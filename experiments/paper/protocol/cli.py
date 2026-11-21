"""Small approval-aware command-line wrapper for paper entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from .campaign import execute_paper_campaign, load_execution_selection, load_paper_campaign


def run_family(default_declaration: Path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, default=default_declaration)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    campaign = load_paper_campaign(args.declaration)
    if args.selection is None or not args.confirm_approved:
        print(f"Prepared, unexecuted campaign: {campaign.document['campaign_id']}")
        print("Execution requires --selection PATH --confirm-approved after separate approval.")
        return
    selection = load_execution_selection(args.selection, campaign)
    print(execute_paper_campaign(campaign, selection))

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.cli")
