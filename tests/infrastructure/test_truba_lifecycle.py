from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from infrastructure.truba import lifecycle


ROOT = Path(__file__).resolve().parents[2]


def test_remote_scripts_are_lf_only_and_validation_has_no_submit_surface() -> None:
    materialize = (ROOT / "infrastructure/truba/materialize_exact_source.sh").read_bytes()
    validate = (ROOT / "infrastructure/truba/validate_exact_source.sh").read_bytes()
    cleanup = (ROOT / "infrastructure/truba/remove_transfer_bundle.sh").read_bytes()
    assert b"\r" not in materialize + validate + cleanup
    assert b"[[ ! -e \"$remote_root\" ]]" in materialize
    assert b"--validate-only" in validate
    assert b"--submit" not in validate
    assert b"mutaco-source-" in cleanup


def test_remote_dispatch_uses_argument_list_and_exact_script_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, b"{}\n", b"")

    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)
    output = lifecycle._run_remote_script(
        "truba", "remove_transfer_bundle.sh", ("/tmp/mutaco-source-safe.bundle",)
    )
    assert output == "{}\n"
    assert observed["command"] == [
        "ssh", "truba", "bash", "-s", "--", "/tmp/mutaco-source-safe.bundle"
    ]
    assert observed["input"] == (
        ROOT / "infrastructure/truba/remove_transfer_bundle.sh"
    ).read_bytes()
    assert "shell" not in observed


def test_remote_dispatch_rejects_shell_metacharacters() -> None:
    with pytest.raises(lifecycle.TrubaLifecycleError, match="unsafe remote argument"):
        lifecycle._run_remote_script(
            "truba", "remove_transfer_bundle.sh", ("/tmp/safe;touch-unsafe",)
        )


def test_transport_configuration_is_an_argument_not_shell_text(tmp_path, monkeypatch):
    config=tmp_path/'transport config'
    config.write_text('Host unit-test\n  HostName example.invalid\n')
    observed=[]
    def run(command,**kwargs):
        observed.extend(command)
        return subprocess.CompletedProcess(command,0,b'{}\n',b'')
    monkeypatch.setattr(lifecycle.subprocess,'run',run)
    lifecycle._run_remote_script('unit-test','remove_transfer_bundle.sh',('/tmp/mutaco-source-safe.bundle',),ssh_config=config)
    assert observed[:4]==['ssh','-F',str(config),'unit-test']


@pytest.mark.parametrize(
    "value",
    ("relative/path", "/path with spaces", "/safe/../escape", "/safe/.."),
)
def test_remote_path_validation_rejects_shell_sensitive_or_escaping_values(value: str) -> None:
    with pytest.raises(lifecycle.TrubaLifecycleError):
        lifecycle._safe_remote_path(value, label="test path")


def test_validation_json_stream_and_receipt_are_machine_readable(tmp_path: Path) -> None:
    documents = lifecycle._parse_json_documents('{"scientific_calls":0}\n{"scientific_calls_executed":0}\n')
    assert documents == ({"scientific_calls": 0}, {"scientific_calls_executed": 0})
    body = {
        "schema_version": "mutaco-truba-validation-receipt-v1",
        "status": "validated_zero_call",
        "scientific_calls": 0,
        "job_submitted": False,
        "submission_authorized": False,
    }
    path = tmp_path / "receipt.json"
    receipt = lifecycle._write_receipt(path, body)
    assert json.loads(path.read_text(encoding="utf-8")) == receipt
    assert receipt["receipt_identity"].startswith("mutaco-truba-validation-receipt-v1-")
    with pytest.raises(lifecycle.TrubaLifecycleError, match="append-only"):
        lifecycle._write_receipt(path, body)


def test_validation_json_stream_accepts_zero_call_execution_preflight() -> None:
    documents = lifecycle._parse_json_documents(
        '{"scientific_calls":0}\n'
        '{"scientific_calls_executed":0}\n'
        '{"scientific_calls_executed":0,"job_submitted":false}\n'
    )
    assert len(documents) == 3
    preflight = documents[2]
    assert preflight["scientific_calls_executed"] == 0
    assert preflight["job_submitted"] is False


def test_zero_call_validation_accepts_preview_without_optional_nested_call_preview() -> None:
    lifecycle._require_zero_call_validation(
        {"scientific_calls": 0}, {"scientific_calls_executed": 0}
    )


@pytest.mark.parametrize(
    ("layout", "preview"),
    (
        ({}, {"scientific_calls_executed": 0}),
        ({"scientific_calls": 0}, {}),
        ({"scientific_calls": 1}, {"scientific_calls_executed": 0}),
        ({"scientific_calls": 0}, {"scientific_calls_executed": 1}),
        (
            {"scientific_calls": 0},
            {
                "scientific_calls_executed": 0,
                "call_preview": {"scientific_calls_executed": 1},
            },
        ),
        (
            {"scientific_calls": 0},
            {"scientific_calls_executed": 0, "call_preview": {}},
        ),
    ),
)
def test_zero_call_validation_rejects_missing_or_nonzero_reported_counts(
    layout: dict[str, object], preview: dict[str, object]
) -> None:
    with pytest.raises(lifecycle.TrubaLifecycleError, match="reports scientific calls"):
        lifecycle._require_zero_call_validation(layout, preview)


def test_lifecycle_requires_explicit_transfer_and_exposes_no_submit_command(tmp_path: Path) -> None:
    parser = lifecycle.build_parser()
    args = parser.parse_args([
        "prepare-validate",
        "--host", "truba",
        "--base-source", "/arf/scratch/base/source-exact",
        "--base-commit", "0" * 40,
        "--remote-root", "/arf/scratch/new",
        "--remote-python", "/arf/env/bin/python",
        "--profile", "infrastructure/truba/profile.env",
        "--receipt", str(tmp_path / "receipt.json"),
    ])
    assert args.confirm_transfer is False
    assert parser._subparsers._group_actions[0].choices.keys() == {"prepare-validate"}


def test_multitarget_oscillator_validate_dispatch_has_no_submission_surface() -> None:
    validate = (ROOT / "infrastructure/truba/validate_exact_source.sh").read_text()
    assert "oscillator_physics_multitarget)" in validate
    dispatch = validate.split("oscillator_physics_multitarget)", 1)[1].split(";;", 1)[0]
    assert "oscillator_physics_multi_target_sensitivity/truba/submit.sh --validate-only" in dispatch
    assert "--submit" not in dispatch
