"""Offline evidence I/O only: no fitting, simulator execution or cluster access."""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import subprocess
import sys
from types import MappingProxyType

import pytest
import torch

from reproducibility.tensor_export import (
    TensorEvidenceExportError,
    save_tensor_evidence,
)


def test_original_native_export_failure_is_reproduced(tmp_path: Path) -> None:
    metadata = MappingProxyType({"seed_manifest": MappingProxyType({"thermal/0": 73007})})
    # This is exactly the insufficient shallow conversion in the failed worker.
    with pytest.raises(TypeError, match="mappingproxy"):
        torch.save({"response": torch.tensor([1.25]), "metadata": dict(metadata)}, tmp_path / "failed.pt")


def test_nested_metadata_and_tensor_bits_ownership_views_and_order(tmp_path: Path) -> None:
    bits = torch.tensor([0, -2147483648, 0x7FC12345, 1067450368], dtype=torch.int32)
    response = bits.view(torch.float32)
    base = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    view = base[:, 1::2]
    scalar = torch.tensor(-0.0, dtype=torch.float64)
    nested = MappingProxyType({"z_seed": 9223372036854775807, "a_values": (1, [MappingProxyType({"flag": True})])})
    payload = {"response": response, "metadata": MappingProxyType({"seed_manifest": nested}),
               "shared": response, "base": base, "views": (view, scalar), "minus_zero": -0.0}
    target = tmp_path / "evidence.pt"
    result = save_tensor_evidence(payload, target, required_tensors=[("response",), ("views", 0)])
    restored = torch.load(target, weights_only=True)
    assert torch.equal(restored["response"].view(torch.int32), bits)
    assert restored["response"] is restored["shared"]
    assert restored["views"][0].untyped_storage().data_ptr() == restored["base"].untyped_storage().data_ptr()
    assert restored["views"][0].stride() == view.stride()
    assert restored["views"][0].storage_offset() == view.storage_offset()
    assert restored["base"].requires_grad and restored["views"][0].requires_grad
    assert restored["views"][0].dtype == torch.float64 and restored["views"][0].device == view.device
    assert restored["views"][1].shape == torch.Size([])
    assert isinstance(restored["views"], tuple)
    assert isinstance(restored["metadata"]["seed_manifest"]["a_values"], tuple)
    assert isinstance(restored["metadata"]["seed_manifest"]["a_values"][1], list)
    assert list(restored["metadata"]["seed_manifest"]) == ["z_seed", "a_values"]
    assert restored["metadata"]["seed_manifest"]["z_seed"] == 9223372036854775807
    assert restored["metadata"]["seed_manifest"]["a_values"][1][0] == {"flag": True}
    assert struct.pack("<d", restored["minus_zero"]) == struct.pack("<d", -0.0)
    assert torch.equal(response.view(torch.int32), bits)
    assert payload["metadata"]["seed_manifest"] is nested  # original remains immutable
    assert result["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert result["size_bytes"] == target.stat().st_size


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.bool, torch.complex64])
def test_supported_tensor_dtypes_and_empty_shapes(tmp_path: Path, dtype: torch.dtype) -> None:
    values = torch.tensor([0, 1], dtype=dtype)
    empty = torch.empty((0, 3), dtype=dtype)
    target = tmp_path / "evidence.pt"
    save_tensor_evidence({"values": values, "empty": empty}, target,
                         required_tensors=[("values",), ("empty",)])
    restored = torch.load(target, weights_only=True)
    assert restored["values"].dtype == dtype and restored["empty"].shape == (0, 3)
    assert torch.equal(restored["values"], values)


@pytest.mark.parametrize("payload,path,message", [
    ({"metadata": {}}, ("response",), "missing"),
    ({"response": None}, ("response",), "not a tensor"),
    ({"cases": []}, ("cases", 0, "response"), "missing"),
])
def test_missing_scientific_evidence_is_an_explicit_failure(tmp_path: Path, payload, path, message) -> None:
    with pytest.raises(TensorEvidenceExportError, match=message):
        save_tensor_evidence(payload, tmp_path / "capture.pt", required_tensors=[path])
    assert list(tmp_path.iterdir()) == []


def test_required_contract_cannot_be_empty(tmp_path: Path) -> None:
    with pytest.raises(TensorEvidenceExportError, match="required tensor"):
        save_tensor_evidence({"response": torch.ones(1)}, tmp_path / "capture.pt", required_tensors=[])


def test_existing_evidence_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "original.pt"
    target.write_bytes(b"original evidence bytes")
    with pytest.raises(FileExistsError):
        save_tensor_evidence({"response": torch.ones(1)}, target, required_tensors=[("response",)])
    assert target.read_bytes() == b"original evidence bytes"


def test_dangling_symlink_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "capture.pt"
    target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        save_tensor_evidence({"response": torch.ones(1)}, target, required_tensors=[("response",)])
    assert target.is_symlink() and not (tmp_path / "missing").exists()


@pytest.mark.parametrize("failure", ["serialization", "readback", "publication_race"])
def test_failed_exports_leave_no_partial_success(tmp_path: Path, monkeypatch, failure: str) -> None:
    target = tmp_path / "capture.pt"
    if failure == "serialization":
        def fail(*args, **kwargs):
            raise OSError("synthetic write failure")
        monkeypatch.setattr(torch, "save", fail)
        expected, message = OSError, "synthetic write"
    elif failure == "readback":
        monkeypatch.setattr(torch, "load", lambda *a, **k: {"response": torch.zeros(1)})
        expected, message = TensorEvidenceExportError, "bits changed"
    else:
        import reproducibility.tensor_export as exporter
        original_link = exporter.os.link
        def race(source, destination):
            Path(destination).write_bytes(b"other original capture")
            return original_link(source, destination)
        monkeypatch.setattr(exporter.os, "link", race)
        expected, message = FileExistsError, None
    with pytest.raises(expected, match=message):
        save_tensor_evidence({"response": torch.ones(1)}, target, required_tensors=[("response",)])
    assert not list(tmp_path.glob("tensor-export-*"))
    if failure == "publication_race":
        assert target.read_bytes() == b"other original capture"
    else:
        assert not target.exists()


@pytest.mark.parametrize("unsupported", ["cycle", "object", "key", "meta_tensor"])
def test_unsupported_payload_is_rejected_without_running_pickle_hooks(tmp_path: Path, unsupported: str) -> None:
    payload = {"response": torch.ones(1)}
    if unsupported == "cycle":
        payload["cycle"] = payload
    elif unsupported == "object":
        class Unsafe:
            def __reduce__(self):
                pytest.fail("arbitrary pickle hook executed")
        payload["object"] = Unsafe()
    elif unsupported == "key":
        payload[1.25] = "invalid key"
    else:
        payload["meta"] = torch.empty(1, device="meta")
    with pytest.raises(TensorEvidenceExportError):
        save_tensor_evidence(payload, tmp_path / "capture.pt", required_tensors=[("response",)])
    assert list(tmp_path.iterdir()) == []


def test_import_needs_no_torch_and_starts_no_scientific_work() -> None:
    root = Path(__file__).resolve().parents[2]
    program = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','mutaco','mutaco_graph','mutaco_analytic'}:
   raise AssertionError('scientific dependency imported: '+fullname)
sys.meta_path.insert(0,Block())
sys.path.insert(0,sys.argv[1])
import reproducibility.tensor_export
"""
    result = subprocess.run([sys.executable, "-I", "-c", program, str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
