"""Paper-facing handwritten analytic campaign entry point."""

from pathlib import Path

from experiments.paper.protocol.cli import run_family


if __name__ == "__main__":
    run_family(Path(__file__).with_name("configs") / "campaign.json")
