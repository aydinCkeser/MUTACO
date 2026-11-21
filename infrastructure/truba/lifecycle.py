"""Canonical exact-commit TRUBA preparation and zero-call validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

from mutaco import stable_hash
from mutaco.atomic_io import atomic_write_text


ROOT = Path(__file__).resolve().parents[2]
TRUBA = Path(__file__).resolve().parent
_SAFE_HOST = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
_SAFE_PROFILE = re.compile(r"^[A-Za-z0-9._/-]+$")
_SAFE_REMOTE_ARGUMENT = re.compile(r"^[A-Za-z0-9._/:-]+$")


class TrubaLifecycleError(RuntimeError):
    """Typed engineering failure before any scientific submission surface."""


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=input_bytes is None,
        check=False,
    )
    if completed.returncode:
        stderr = (
            completed.stderr.decode("utf-8", errors="replace")
            if isinstance(completed.stderr, bytes)
            else completed.stderr
        )
        raise TrubaLifecycleError(
            f"command failed ({completed.returncode}): {list(command)!r}: {stderr.strip()}"
        )
    if input_bytes is not None:
        return subprocess.CompletedProcess(
            completed.args,
            completed.returncode,
            completed.stdout.decode("utf-8"),
            completed.stderr.decode("utf-8"),
        )
    return completed


def _git(repository: Path, *arguments: str) -> str:
    return _run_checked(("git", "-C", str(repository), *arguments)).stdout.strip()


def _safe_remote_path(value: str, *, label: str) -> str:
    if not _SAFE_REMOTE_PATH.fullmatch(value) or "/../" in value or value.endswith("/.."):
        raise TrubaLifecycleError(f"unsafe {label}: {value!r}")
    return value.rstrip("/")


def _safe_profile(value: str) -> str:
    if (
        not _SAFE_PROFILE.fullmatch(value)
        or value.startswith("/")
        or value.startswith("../")
        or "/../" in value
        or value.endswith("/..")
    ):
        raise TrubaLifecycleError(f"unsafe repository-relative profile: {value!r}")
    return value


def _script_bytes(name: str) -> bytes:
    payload = (TRUBA / name).read_bytes()
    if b"\r" in payload:
        raise TrubaLifecycleError(f"remote script is not LF-normalized: {name}")
    return payload


def _transport_options(ssh_config: Path | None) -> tuple[str, ...]:
    if ssh_config is None:
        return ()
    config = ssh_config.resolve(strict=True)
    if not config.is_file():
        raise TrubaLifecycleError("SSH configuration must be a local file")
    return ("-F", str(config))


def _run_remote_script(
    host: str, script: str, arguments: Sequence[str], *, ssh_config: Path | None = None,
) -> str:
    if not _SAFE_HOST.fullmatch(host):
        raise TrubaLifecycleError(f"unsafe SSH host alias: {host!r}")
    for argument in arguments:
        if not _SAFE_REMOTE_ARGUMENT.fullmatch(argument):
            raise TrubaLifecycleError(f"unsafe remote argument: {argument!r}")
    completed = _run_checked(
        ("ssh", *_transport_options(ssh_config), host, "bash", "-s", "--", *arguments),
        input_bytes=_script_bytes(script),
    )
    return completed.stdout


def _parse_json_documents(payload: str) -> tuple[dict[str, Any], ...]:
    decoder = json.JSONDecoder()
    offset = 0
    documents = []
    while offset < len(payload):
        while offset < len(payload) and payload[offset].isspace():
            offset += 1
        if offset == len(payload):
            break
        value, offset = decoder.raw_decode(payload, offset)
        if not isinstance(value, Mapping):
            raise TrubaLifecycleError("TRUBA lifecycle output contains non-object JSON")
        documents.append(dict(value))
    return tuple(documents)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_zero_call_validation(
    layout: Mapping[str, Any], preview: Mapping[str, Any]
) -> None:
    """Accept supported previews while requiring every reported call count to be zero."""

    if layout.get("scientific_calls") != 0 or preview.get("scientific_calls_executed") != 0:
        raise TrubaLifecycleError("validate-only receipt reports scientific calls")
    call_preview = preview.get("call_preview")
    if call_preview is not None and (
        not isinstance(call_preview, Mapping)
        or call_preview.get("scientific_calls_executed") != 0
    ):
        raise TrubaLifecycleError("validate-only receipt reports scientific calls")


def _write_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    destination = path.resolve(strict=False)
    if destination.exists():
        raise TrubaLifecycleError("TRUBA validation receipt is append-only and already exists")
    document = {
        **dict(body),
        "receipt_identity": stable_hash("mutaco-truba-validation-receipt-v1", dict(body)),
    }
    atomic_write_text(
        destination,
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    return document


def prepare_validate(
    *,
    repository: Path,
    host: str,
    base_source: str,
    base_commit: str,
    remote_root: str,
    remote_python: str,
    profile: str,
    receipt: Path,
    confirm_transfer: bool,
    ssh_config: Path | None = None,
) -> dict[str, Any]:
    """Materialize one fresh exact source and run only zero-call validation."""

    if not confirm_transfer:
        raise TrubaLifecycleError("exact-source transfer requires --confirm-transfer")
    repo = repository.resolve()
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise TrubaLifecycleError("local source checkout must be clean")
    expected_commit = _git(repo, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise TrubaLifecycleError("base commit must be a full lowercase Git hash")
    _run_checked(("git", "-C", str(repo), "merge-base", "--is-ancestor", base_commit, expected_commit))
    branch = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    bundle_ref = f"refs/heads/{branch}"
    if _git(repo, "rev-parse", bundle_ref) != expected_commit:
        raise TrubaLifecycleError("checked-out branch does not resolve to HEAD")

    profile_name = _safe_profile(profile)
    profile_path = (repo / profile_name).resolve()
    if not profile_path.is_file() or not _inside(profile_path, repo):
        raise TrubaLifecycleError("TRUBA environment profile is missing or escapes the repository")
    remote_base = _safe_remote_path(base_source, label="base source")
    remote = _safe_remote_path(remote_root, label="remote root")
    runtime = _safe_remote_path(remote_python, label="remote Python")
    source_root = f"{remote}/source-exact"
    output_root = f"{remote}/workers"
    log_root = f"{remote}/logs"
    remote_bundle = f"/tmp/mutaco-source-{expected_commit[:12]}-{uuid4().hex}.bundle"

    system_temp = Path(tempfile.gettempdir()).resolve()
    if _inside(system_temp, repo):
        raise TrubaLifecycleError("system temporary directory resolves inside the repository")

    materialized: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="mutaco-truba-") as temporary:
        bundle = Path(temporary) / "source.bundle"
        _run_checked(
            ("git", "-C", str(repo), "bundle", "create", str(bundle), f"{base_commit}..{bundle_ref}")
        )
        _run_checked(("git", "bundle", "verify", str(bundle)))
        heads = _run_checked(("git", "bundle", "list-heads", str(bundle))).stdout
        if f"{expected_commit} {bundle_ref}" not in heads.splitlines():
            raise TrubaLifecycleError("incremental bundle does not expose the exact expected ref")
        try:
            _run_checked(("scp", *_transport_options(ssh_config), "--", str(bundle), f"{host}:{remote_bundle}"))
            materialization_output = _run_remote_script(
                host,
                "materialize_exact_source.sh",
                (
                    remote_base,
                    remote,
                    remote_bundle,
                    base_commit,
                    expected_commit,
                    bundle_ref,
                ),
                ssh_config=ssh_config,
            )
            documents = _parse_json_documents(materialization_output)
            if len(documents) != 1 or documents[0].get("status") != "materialized_clean":
                raise TrubaLifecycleError("TRUBA source materialization receipt is invalid")
            materialized = documents[0]
        finally:
            try:
                _run_remote_script(host, "remove_transfer_bundle.sh", (remote_bundle,), ssh_config=ssh_config)
            except TrubaLifecycleError:
                if materialized is not None:
                    raise

    validation_output = _run_remote_script(
        host,
        "validate_exact_source.sh",
        (source_root, output_root, log_root, runtime, expected_commit, profile_name),
        ssh_config=ssh_config,
    )
    validation_documents = _parse_json_documents(validation_output)
    if len(validation_documents) not in {2, 3}:
        raise TrubaLifecycleError(
            "validate-only must emit layout, preview, and optional execution preflight"
        )
    layout, preview, *additional = validation_documents
    _require_zero_call_validation(layout, preview)
    execution_preflight = additional[0] if additional else None
    if execution_preflight is not None and (
        execution_preflight.get("scientific_calls_executed") != 0
        or execution_preflight.get("job_submitted") is not False
    ):
        raise TrubaLifecycleError("execution preflight is not zero-call and non-submitting")
    body = {
        "schema_version": "mutaco-truba-validation-receipt-v1",
        "status": "validated_zero_call",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "source_commit": expected_commit,
        "base_commit": base_commit,
        "base_source_root": remote_base,
        "source_root": source_root,
        "output_root": output_root,
        "log_root": log_root,
        "runtime_python": runtime,
        "profile": profile_name,
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "transfer": "incremental_git_bundle_v1",
        "source_materialization": materialized,
        "layout": layout,
        "worker_preview": preview,
        "execution_preflight": execution_preflight,
        "scientific_calls": 0,
        "job_submitted": False,
        "submission_authorized": False,
    }
    return _write_receipt(receipt, body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare-validate")
    prepare.add_argument("--repository", type=Path, default=ROOT)
    prepare.add_argument("--host", required=True)
    prepare.add_argument("--base-source", required=True)
    prepare.add_argument("--base-commit", required=True)
    prepare.add_argument("--remote-root", required=True)
    prepare.add_argument("--remote-python", required=True)
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--receipt", type=Path, required=True)
    prepare.add_argument("--confirm-transfer", action="store_true")
    prepare.add_argument("--ssh-config", type=Path, help="local transport-only OpenSSH config; never copied into source or receipts")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    document = prepare_validate(
        repository=args.repository,
        host=args.host,
        base_source=args.base_source,
        base_commit=args.base_commit,
        remote_root=args.remote_root,
        remote_python=args.remote_python,
        profile=args.profile,
        receipt=args.receipt,
        confirm_transfer=args.confirm_transfer,
        ssh_config=args.ssh_config,
    )
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
