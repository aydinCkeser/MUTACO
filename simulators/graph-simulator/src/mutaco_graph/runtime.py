"""Device selection that is independent of graph-oscillator physics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GraphDeviceResolution:
    """Requested and effective Torch device with explicit fallback metadata."""

    requested: str
    effective: str
    used_fallback: bool
    fallback_reason: str | None

    @property
    def device(self) -> torch.device:
        return torch.device(self.effective)


def resolve_graph_device(requested: str = "auto") -> GraphDeviceResolution:
    """Resolve CPU/CUDA without silently weakening an explicit CUDA request."""

    if not isinstance(requested, str):
        raise TypeError("graph device request must be a string")
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return GraphDeviceResolution(
                requested="auto",
                effective="cuda:0",
                used_fallback=False,
                fallback_reason=None,
            )
        return GraphDeviceResolution(
            requested="auto",
            effective="cpu",
            used_fallback=True,
            fallback_reason="torch CUDA is unavailable",
        )
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid graph device request: {requested!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("graph device must be 'auto', 'cpu', or a CUDA device")
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("indexed CPU devices are not supported")
        return GraphDeviceResolution(
            requested=normalized,
            effective="cpu",
            used_fallback=False,
            fallback_reason=None,
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"graph device {normalized!r} explicitly requires CUDA, but Torch CUDA is unavailable"
        )
    index = 0 if device.index is None else device.index
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise RuntimeError(
            f"graph CUDA device index {index} is unavailable; detected {device_count} device(s)"
        )
    return GraphDeviceResolution(
        requested=normalized,
        effective=f"cuda:{index}",
        used_fallback=False,
        fallback_reason=None,
    )
