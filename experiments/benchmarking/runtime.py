"""Experiment runtime setup that does not alter scientific configuration."""

from __future__ import annotations

import os
from typing import Any, Mapping

import torch

from mutaco import RuntimeRequest


def runtime_request_from_resolved_device(
    device: str, *, dtype: str
) -> RuntimeRequest:
    """Reconstruct a strict runtime request from a parent's resolved device."""

    if device == "cpu":
        return RuntimeRequest(device="cpu", dtype=dtype)
    if device == "cuda":
        return RuntimeRequest(device="cuda", dtype=dtype)
    prefix = "cuda:"
    if device.startswith(prefix):
        index_text = device[len(prefix) :]
        if index_text.isascii() and index_text.isdecimal():
            return RuntimeRequest(
                device="cuda", cuda_index=int(index_text), dtype=dtype
            )
    raise ValueError(f"unsupported resolved runtime device: {device!r}")


def configure_graph_determinism(
    config: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    """Enable the accepted exact-replay mode for scientific graph CUDA work."""

    applies = config.get("simulator", {}).get("adapter") == "graph_scientific_d4"
    if applies and device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    return {
        "applies": applies,
        "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "policy": "graph_exact_replay_required_on_cuda_v1",
    }
