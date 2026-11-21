"""Historical generic d=4 graph-inversion campaign interface (unsupported).

Import and --help are safe. The missing historical reporting modules prevent
execution before any campaign starts. Use the specific supported entry points
in experiments/graph_inversion/README.md; this is not an offline analysis CLI.

Its intended sequence was protocol loading, hidden-target construction/recovery,
MUTACO and cold-start NPE, then reporting from persisted results. Restoring that
execution interface requires a separate review; no missing analysis is invented
here and no scientific run is authorized by this module's presence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.benchmarking import load_campaign, run_campaign


ROOT = Path(__file__).resolve().parents[2]


def _validated_complete_manifest(output: Path, configuration: dict[str, Any]) -> dict[str, Any] | None:
    """Recover a finalized campaign without repeating scientific computation."""

    path = output / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("configuration") != configuration:
        raise ValueError("existing campaign manifest configuration mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("existing campaign manifest has no artifact inventory")
    for relative, expected in artifacts.items():
        artifact = output / relative
        if not artifact.is_file():
            raise ValueError(f"existing campaign artifact is missing: {relative}")
        content = artifact.read_bytes()
        if len(content) != int(expected["bytes"]):
            raise ValueError(f"existing campaign artifact size mismatch: {relative}")
        if hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"existing campaign artifact hash mismatch: {relative}")
    return manifest


def configured_output(config: Path) -> Path:
    """Resolve the declared output path without creating it."""

    document = json.loads(config.read_text(encoding="utf-8"))
    output = Path(str(document["outputs"]["directory"]))
    if output.is_absolute():
        raise ValueError("graph-inversion output directory must be repository-relative")
    return ROOT / output


def execute(
    *,
    config: Path,
    output: Path | None = None,
    device: str | None = None,
    restart: bool = True,
) -> dict[str, Any]:
    # Fail before configuration loading or science when legacy reporting is absent.
    from experiments.graph_inversion.plot_results import plot_campaign
    from experiments.graph_inversion.summarize_results import summarize_campaign

    # 1. Read and validate the immutable d=4 scientific declaration.
    campaign = load_campaign(config, device=device)
    destination = output or configured_output(config)

    # 2. Construct/recover one hidden-target ledger, then run MUTACO and NPE.
    recovered_manifest = (
        _validated_complete_manifest(destination, campaign.document) if restart else None
    )
    execution = None
    if recovered_manifest is None:
        execution = run_campaign(
            campaign,
            output=destination,
            restart=restart,
            verbose_events=True,
        )

    # 3. Recompute tables and plots strictly from the persisted result records.
    summary = summarize_campaign(destination)
    plots = plot_campaign(destination)
    return {
        "output": destination,
        "manifest": recovered_manifest or execution.manifest,
        "recovered_complete_campaign": recovered_manifest is not None,
        "summary_artifacts": summary["artifacts"],
        "plot_artifacts": plots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="explicitly approved graph-inversion campaign declaration",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="disable recovery of only complete, hash-valid target/method records",
    )
    arguments = parser.parse_args()
    result = execute(
        config=arguments.config,
        output=arguments.output,
        device=arguments.device,
        restart=not arguments.fresh,
    )
    print(f"[graph-inversion] complete output={result['output']}")


if __name__ == "__main__":
    main()
