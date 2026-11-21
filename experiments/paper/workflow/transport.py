"""Deterministic compact export and verified fetch for Paper-v3 populations."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from typing import Any

from experiments.benchmarking.artifacts import file_hash, write_pretty_json
from experiments.paper.common.validation import PaperContractError


_EXPORT_SCHEMA = "mutaco-paper-v3-compact-export-v1"
_INCLUDED_ROOTS = ("collection",)
_INCLUDED_WORKER_PARTS = ("records", "assessment_randomness")


def _sidecar(path: Path) -> Path:
    return Path(f"{path}.sha256") if path.suffix == ".jsonl" else path.with_suffix(".sha256")


def _write_sidecar(path: Path) -> None:
    _sidecar(path).write_text(f"{file_hash(path)}\n", encoding="ascii")


def _selected_files(root: Path) -> tuple[Path, ...]:
    selected: set[Path] = set()
    for name in _INCLUDED_ROOTS:
        selected.update(
            path for path in (root / name).rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl", ".sha256"}
        )
    for worker in root.glob("worker-*"):
        if not worker.is_dir():
            continue
        for name in _INCLUDED_WORKER_PARTS:
            selected.update(
                path for path in (worker / name).rglob("*")
                if path.is_file() and path.suffix in {".json", ".jsonl", ".sha256"}
            )
    files = tuple(sorted(selected, key=lambda path: path.relative_to(root).as_posix()))
    if not files:
        raise PaperContractError("compact export source contains no eligible artifacts")
    json_files = tuple(path for path in files if path.suffix in {".json", ".jsonl"})
    if not json_files:
        raise PaperContractError("compact export source contains no JSON evidence")
    for path in json_files:
        sidecar = _sidecar(path)
        if sidecar not in selected or sidecar.read_text(encoding="ascii").strip() != file_hash(path):
            raise PaperContractError(f"compact export hash pair is invalid: {path}")
    return files


def _deterministic_archive(root: Path, files: tuple[Path, ...], output: Path) -> None:
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in files:
                    data = path.read_bytes()
                    info = tarfile.TarInfo(path.relative_to(root).as_posix())
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(data))


def export_compact_population(
    source: Path, output_prefix: Path, *, confirm_authorized: bool
) -> dict[str, Any]:
    """Package only collection, records, and assessment coordinates."""

    if not confirm_authorized:
        raise PaperContractError("compact export requires explicit runtime authorization")
    root = source.resolve()
    if not root.is_dir():
        raise PaperContractError(f"compact export source is not a directory: {root}")
    archive_path = Path(f"{output_prefix}.tar.gz")
    manifest_path = Path(f"{output_prefix}.manifest.json")
    outputs = (archive_path, manifest_path, _sidecar(archive_path), _sidecar(manifest_path))
    if any(path.exists() for path in outputs):
        raise PaperContractError("compact export outputs are append-only and already exist")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    files = _selected_files(root)
    _deterministic_archive(root, files, archive_path)
    manifest = {
        "schema_version": _EXPORT_SCHEMA,
        "archive_name": archive_path.name,
        "archive_sha256": file_hash(archive_path),
        "scientific_calls": 0,
        "raw_posterior_artifacts_included": False,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in files
        ],
    }
    write_pretty_json(manifest_path, manifest)
    _write_sidecar(archive_path)
    _write_sidecar(manifest_path)
    return {
        "status": "exported",
        "archive": archive_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "file_count": len(files),
        "archive_sha256": manifest["archive_sha256"],
        "scientific_calls": 0,
    }


def _copy_or_scp(source: str, destination: Path) -> None:
    local = Path(source)
    if local.is_file():
        shutil.copyfile(local, destination)
        return
    completed = subprocess.run(
        ["scp", "--", source, str(destination)], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if completed.returncode:
        raise PaperContractError(f"compact fetch failed: {completed.stderr.strip()}")


def _safe_extract(archive_path: Path, destination: Path, manifest: dict[str, Any]) -> None:
    declared = {str(item["path"]): item for item in manifest["files"]}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        observed = {member.name for member in members}
        if observed != set(declared) or len(observed) != len(members):
            raise PaperContractError("compact archive inventory differs from manifest")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or any(":" in part for part in path.parts)
                or not member.isfile()
            ):
                raise PaperContractError("compact archive contains an unsafe member")
            target = destination.joinpath(*path.parts).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as exc:
                raise PaperContractError("compact archive member escapes destination") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise PaperContractError("compact archive member cannot be read")
            with target.open("xb") as handle:
                shutil.copyfileobj(stream, handle)
            expected = declared[member.name]
            if target.stat().st_size != expected["size_bytes"] or file_hash(target) != expected["sha256"]:
                raise PaperContractError(f"compact archive member hash mismatch: {member.name}")


def fetch_compact_export(
    source_prefix: str, destination: Path, *, confirm_authorized: bool
) -> dict[str, Any]:
    """Fetch an exact compact package, verify it, and safely extract it once."""

    if not confirm_authorized:
        raise PaperContractError("compact fetch requires explicit runtime authorization")
    target = destination.resolve()
    if target.exists():
        raise PaperContractError("compact fetch destination is append-only and already exists")
    target.mkdir(parents=True)
    archive = target / "population.tar.gz"
    manifest_path = target / "export_manifest.json"
    archive_hash = target / "population.tar.sha256"
    manifest_hash = target / "export_manifest.sha256"
    suffixes = (
        (".tar.gz", archive),
        (".manifest.json", manifest_path),
        (".tar.sha256", archive_hash),
        (".manifest.sha256", manifest_hash),
    )
    try:
        for suffix, output in suffixes:
            _copy_or_scp(f"{source_prefix}{suffix}", output)
        if manifest_hash.read_text(encoding="ascii").strip() != file_hash(manifest_path):
            raise PaperContractError("compact export manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != _EXPORT_SCHEMA:
            raise PaperContractError("unsupported compact export manifest")
        expected_archive = str(manifest.get("archive_sha256"))
        if archive_hash.read_text(encoding="ascii").strip() != expected_archive or file_hash(archive) != expected_archive:
            raise PaperContractError("compact export archive hash mismatch")
        extracted = target / "population"
        extracted.mkdir()
        _safe_extract(archive, extracted, manifest)
    except Exception:
        # Preserve the append-only failed staging directory for audit.
        raise
    return {
        "status": "fetched_verified",
        "destination": target.as_posix(),
        "population_root": extracted.as_posix(),
        "file_count": len(manifest["files"]),
        "archive_sha256": expected_archive,
        "scientific_calls": 0,
    }
