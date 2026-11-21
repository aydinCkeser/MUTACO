"""Format-level validation for generated figure artifacts."""

from __future__ import annotations

from pathlib import Path
import struct
import xml.etree.ElementTree as ET


def validate_svg(path: Path) -> None:
    root = ET.parse(path).getroot()
    if not root.tag.endswith("svg"):
        raise ValueError("generated SVG lacks an svg root")


def validate_png(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("generated PNG signature is invalid")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1000 or height < 400:
        raise ValueError("generated PNG is below preview resolution")
    return width, height


def validate_pdf(path: Path) -> None:
    data = path.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise ValueError("generated PDF framing is invalid")

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.figures.common.plot_validation")
