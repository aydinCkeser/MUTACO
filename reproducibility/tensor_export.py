"""Explicit offline export of dense tensor evidence and immutable metadata.

Mapping containers become insertion-ordered dictionaries; lists and tuples keep
their structure. Tensor objects, values, storage/views, dtype and device are not
converted. Torch is imported only when this API is called, so importing the
read-only verification surface still needs no scientific dependencies.

The caller must declare required tensor paths from its capture contract. A
missing tensor is an export error, never a successful partial capture. This API
cannot recover values that were not recorded, and makes no scientific verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any


TensorPath = tuple[str | int, ...]


class TensorEvidenceExportError(ValueError):
    """The supplied evidence cannot satisfy the declared export contract."""


def save_tensor_evidence(
    payload: Any,
    destination: str | Path,
    *,
    required_tensors: Sequence[TensorPath],
) -> dict[str, str | int]:
    """Write a new Torch archive after bitwise readback verification.

    Accept dense, non-quantized tensors, ordinary scalars, mappings with str/int
    keys, lists and tuples. Preserve shared tensor storage through native Torch
    serialization. Reject cycles and other objects rather than invoking arbitrary
    pickle hooks. The parent directory must already exist. Publish only a fully
    verified file; an existing destination (including a symlink) is never replaced.

    Required paths address mapping keys and list/tuple indices, e.g.
    ``("graph_fixed", 0, "reference_response")``. All supplied tensors, not just
    these required leaves, are checked after serialization. No simulator or model
    is constructed or evaluated.
    """
    import torch

    target = Path(destination)
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not required_tensors:
        raise TensorEvidenceExportError("at least one required tensor path is needed")
    tensors: list[tuple[TensorPath, torch.Tensor]] = []
    active: set[int] = set()
    memo: dict[int, Any] = {}

    def plain(value: Any, path: TensorPath) -> Any:
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
                raise TensorEvidenceExportError(f"unsupported tensor storage at {path!r}")
            tensors.append((path, value))
            return value
        if value is None or type(value) in (bool, int, float, complex, str, bytes):
            return value
        if not isinstance(value, (Mapping, list, tuple)):
            raise TensorEvidenceExportError(f"unsupported {type(value).__name__} at {path!r}")
        identity = id(value)
        if identity in active:
            raise TensorEvidenceExportError(f"cyclic evidence at {path!r}")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                if any(type(key) not in (str, int) for key in value):
                    raise TensorEvidenceExportError(f"unsupported mapping key at {path!r}")
                result = {key: plain(item, path + (key,)) for key, item in value.items()}
            else:
                values = [plain(item, path + (index,)) for index, item in enumerate(value)]
                result = tuple(values) if isinstance(value, tuple) else values
            memo[identity] = result
            return result
        finally:
            active.remove(identity)

    def at(document: Any, path: TensorPath) -> Any:
        value = document
        try:
            for key in path:
                value = value[key]
        except (KeyError, IndexError, TypeError) as error:
            raise TensorEvidenceExportError(f"required tensor missing at {path!r}") from error
        return value

    document = plain(payload, ())
    for path in required_tensors:
        if not isinstance(at(document, tuple(path)), torch.Tensor):
            raise TensorEvidenceExportError(f"required value is not a tensor at {path!r}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=target.parent, prefix="tensor-export-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(document, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            recovered = torch.load(stream, weights_only=True)
            for path, original in tensors:
                restored = at(recovered, path)
                if not isinstance(restored, torch.Tensor) or (
                    restored.shape, restored.dtype, restored.device,
                    restored.stride(), restored.storage_offset(), restored.requires_grad,
                ) != (
                    original.shape, original.dtype, original.device,
                    original.stride(), original.storage_offset(), original.requires_grad,
                ):
                    raise TensorEvidenceExportError(f"tensor contract changed at {path!r}")
                # Compare bytes, including NaN payloads and signed zero. Copies
                # used for verification never replace or mutate source tensors.
                left = original.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
                right = restored.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
                if not torch.equal(left, right):
                    raise TensorEvidenceExportError(f"tensor bits changed at {path!r}")
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = stream.seek(0, os.SEEK_END)
        # Hard-link publication is atomic and fails if another file appeared.
        # Unlike replace(), it cannot overwrite original evidence in a race.
        os.link(temporary, target)
        return {"sha256": digest, "size_bytes": size, "tensor_leaves_verified": len(tensors)}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
