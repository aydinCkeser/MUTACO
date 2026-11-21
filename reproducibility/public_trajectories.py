"""Verify the public trajectory archives without extraction or scientific execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "results/paper/scalar_primary_trajectory_archives_v3r2"
)
_MEMBERS = {
    "collection/collection.json",
    "collection/collection.sha256",
    "collection/compact_export.json",
    "collection/compact_export.sha256",
}


def _check_digest(path: Path, sidecar: Path) -> str:
    """Check a file against its SHA-256 sidecar."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != sidecar.read_text(encoding="ascii").strip():
        raise ValueError(f"{path.name}: SHA-256 mismatch")
    return digest


def verify(budget: int, directory: Path = _DIRECTORY) -> dict:
    """Validate a public archive, its manifest, and its member sidecars."""
    if budget not in (40, 100):
        raise ValueError("budget must be 40 or 100")
    location = directory / f"b{budget}-v3r2"
    prefix = f"scalar-primary-b{budget}-v3r2-public"
    archive_path = location / f"{prefix}.tar.gz"
    manifest_path = location / f"{prefix}.manifest.json"
    _check_digest(manifest_path, location / f"{prefix}.manifest.sha256")
    manifest = json.loads(manifest_path.read_bytes())
    archive_sha = _check_digest(archive_path, location / f"{prefix}.tar.sha256")
    if (
        manifest["schema_version"] != "mutaco-public-trajectory-export-v1"
        or manifest["budget"] != budget
        or manifest["archive_name"] != archive_path.name
        or manifest["archive_sha256"] != archive_sha
    ):
        raise ValueError("public trajectory manifest identity mismatch")

    expected = {entry["path"]: entry for entry in manifest["files"]}
    if len(manifest["files"]) != len(_MEMBERS) or set(expected) != _MEMBERS:
        raise ValueError("invalid manifest member inventory")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(_MEMBERS) or {m.name for m in members} != _MEMBERS:
            raise ValueError("invalid archive member inventory")
        for member in members:
            entry = expected[member.name]
            if not member.isfile() or member.size != entry["size_bytes"]:
                raise ValueError(f"{member.name}: member type or size mismatch")
            with archive.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    raise ValueError(f"{member.name}: SHA-256 mismatch")
        for stem in ("collection", "compact_export"):
            with archive.extractfile(f"collection/{stem}.sha256") as stream:
                sidecar = stream.read().decode("ascii").strip()
            if sidecar != expected[f"collection/{stem}.json"]["sha256"]:
                raise ValueError(f"{stem}: internal sidecar mismatch")

    return {"budget": budget, "archive_sha256": archive_sha, "members_verified": len(_MEMBERS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("budget", type=int, choices=(40, 100))
    parser.add_argument("--directory", type=Path, default=_DIRECTORY)
    args = parser.parse_args()
    try:
        result = verify(args.budget, args.directory)
    except (OSError, tarfile.TarError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Public trajectory verification failed: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
