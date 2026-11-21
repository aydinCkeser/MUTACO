"""Strict loader for immutable retained CPU numerical-oracle fixtures."""

from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import torch


_DTYPES = {
    "bool": torch.bool,
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class RetainedTensorRecord:
    name: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    raw: bytes

    def tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        dtype = _DTYPES[self.dtype]
        value = torch.frombuffer(bytearray(self.raw), dtype=dtype)
        return value.reshape(self.shape).clone().to(device=device)


@dataclass(frozen=True)
class RetainedGraphFixture:
    schema_version: str
    fixture_identity: str
    provenance: Mapping[str, Any]
    configuration: Mapping[str, Any]
    tensors: Mapping[str, RetainedTensorRecord]

    @classmethod
    def load(cls, path: str | Path) -> "RetainedGraphFixture":
        source = Path(path)
        document = json.loads(source.read_text(encoding="utf-8"))
        if document.get("schema_version") != "retained-graph-fixture-v1":
            raise ValueError("unsupported retained graph fixture schema")
        if not isinstance(document.get("fixture_identity"), str):
            raise ValueError("retained graph fixture identity is missing")
        records: dict[str, RetainedTensorRecord] = {}
        for name, item in document.get("tensors", {}).items():
            if item.get("dtype") not in _DTYPES:
                raise ValueError(f"unsupported fixture dtype for {name}")
            raw = b64decode(item["data_base64"], validate=True)
            if sha256(raw).hexdigest() != item.get("sha256"):
                raise ValueError(f"fixture tensor hash mismatch for {name}")
            shape = tuple(item["shape"])
            element_size = torch.empty((), dtype=_DTYPES[item["dtype"]]).element_size()
            expected_bytes = element_size
            for size in shape:
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValueError(f"invalid fixture shape for {name}")
                expected_bytes *= size
            if len(raw) != expected_bytes:
                raise ValueError(f"fixture byte length mismatch for {name}")
            records[name] = RetainedTensorRecord(
                name=name,
                axes=tuple(item["axes"]),
                shape=shape,
                dtype=item["dtype"],
                sha256=item["sha256"],
                raw=raw,
            )
        if not records:
            raise ValueError("retained graph fixture contains no tensors")
        return cls(
            schema_version=document["schema_version"],
            fixture_identity=document["fixture_identity"],
            provenance=document["provenance"],
            configuration=document["configuration"],
            tensors=records,
        )

    def tensor(self, name: str, *, device: torch.device | str = "cpu") -> torch.Tensor:
        try:
            record = self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"unknown retained fixture tensor: {name}") from exc
        return record.tensor(device=device)
