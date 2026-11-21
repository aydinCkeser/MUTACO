"""Characterize persisted-data figures using an explicit test-only font profile."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import weakref
from unittest.mock import patch

import matplotlib as mpl
from matplotlib import font_manager, ft2font
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


RENDERERS = (
    "plot_paper120_primary_boxplot_v1", "plot_paper120_primary_parameter_boxplot_v1",
    "plot_paper120_sbi_response_boxplot_v1", "plot_paper120_sbi_response_parameter_scatter_v1",
    "plot_paper120_terminal_domain_ablation_v1", "plot_paper120_local_sensitivity_v6",
)
SENSITIVITY_DATA = "experiments/paper/oscillator_physics_multi_target_sensitivity/outputs/paper120_maximin12_job6353452/plotting_data.json"


def digest(value) -> str:
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False, default=convert).encode()).hexdigest()


def render_profile() -> dict:
    font = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif")))
    return {"matplotlib": mpl.__version__, "numpy": np.__version__,
            "freetype": ft2font.__freetype_version__, "backend": mpl.get_backend().lower(),
            "system": platform.system(), "architecture": platform.machine(),
            "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
            "test_font": "DejaVu Serif", "usetex": False, "source_date_epoch": "0",
            "base_style": "matplotlib-defaults"}


def load_plot_data(module, root: Path):
    name = module.__name__.rsplit(".", 1)[-1]
    if name == RENDERERS[0]:
        return module.load_frozen_values(root / "results/paper/scalar_primary_b40_b100_synthesis_v1")
    if name == RENDERERS[1]:
        return module.load_parameter_errors(root)
    if name == RENDERERS[2]:
        return module.load_frozen_values(root)
    if name == RENDERERS[3]:
        return module.load_frozen_points(root)
    if name == RENDERERS[4]:
        return module.load_frozen_evidence(root)
    if name == RENDERERS[5]:
        path = root / SENSITIVITY_DATA
        assert hashlib.sha256(path.read_bytes()).hexdigest() == module.DATA_SHA256
        return json.loads(path.read_text())
    raise AssertionError(name)


def array_state(value) -> dict:
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            # Object arrays store process addresses; compare their values.
            "sha256": digest(array.tolist()) if array.dtype.hasobject
            else hashlib.sha256(array.tobytes()).hexdigest()}


def figure_state(figure: Figure) -> dict:
    """Compare labels, dimensions, order, geometry, colors and all plotted arrays."""
    axes = []
    for axis in figure.axes:
        legend = axis.get_legend()
        axes.append({
            "title": axis.get_title(), "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel(),
            "position": list(axis.get_position().bounds), "xscale": axis.get_xscale(),
            "yscale": axis.get_yscale(), "xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim()),
            "xticks": array_state(axis.get_xticks()), "yticks": array_state(axis.get_yticks()),
            "xticklabels": [text.get_text() for text in axis.get_xticklabels()],
            "yticklabels": [text.get_text() for text in axis.get_yticklabels()],
            "legend": [text.get_text() for text in legend.get_texts()] if legend else [],
            "texts": [{"text": text.get_text(), "position": list(text.get_position()),
                       "size": text.get_fontsize(), "color": text.get_color(),
                       "horizontal": text.get_ha(), "vertical": text.get_va()} for text in axis.texts],
            "lines": [{"x": array_state(line.get_xdata()), "y": array_state(line.get_ydata()),
                       "label": line.get_label(), "color": line.get_color(),
                       "width": line.get_linewidth(), "style": line.get_linestyle(),
                       "marker": line.get_marker(), "markersize": line.get_markersize()}
                      for line in axis.lines],
            "collections": [{"offsets": array_state(item.get_offsets()),
                             "paths": [array_state(path.vertices) for path in item.get_paths()],
                             "faces": array_state(item.get_facecolors()),
                             "edges": array_state(item.get_edgecolors()),
                             "widths": array_state(item.get_linewidths()), "alpha": item.get_alpha()}
                            for item in axis.collections],
            "patches": [{"path": array_state(item.get_path().vertices),
                         "face": item.get_facecolor(), "edge": item.get_edgecolor(),
                         "width": item.get_linewidth()} for item in axis.patches],
        })
    return {"inches": figure.get_size_inches().tolist(), "dpi": figure.dpi, "axes": axes,
            "texts": [text.get_text() for text in figure.texts],
            "legends": [[text.get_text() for text in legend.get_texts()] for legend in figure.legends]}


def capture_rendering(module, root: Path, output: Path) -> dict:
    """Render original expressions; only TeX/font settings and output root differ.

    This is a before/after engineering capture, never a replacement paper figure.
    Production defaults remain untouched outside the scoped test patches.
    """
    output.mkdir(parents=True, exist_ok=True)
    name = module.__name__.rsplit(".", 1)[-1]
    data = load_plot_data(module, root)
    data_before = digest(data)
    states = []
    seen = weakref.WeakSet()
    update = mpl.RcParams.update
    save = Figure.savefig

    def fallback(parameters, *args, **kwargs):
        update(parameters, *args, **kwargs)
        parameters["text.usetex"] = False
        parameters["font.serif"] = ["DejaVu Serif"]

    def observe(figure, *args, **kwargs):
        if figure not in seen:
            states.append(figure_state(figure))
            seen.add(figure)
        return save(figure, *args, **kwargs)

    with mpl.rc_context(), patch.object(mpl.RcParams, "update", fallback), \
            patch.object(Figure, "savefig", observe), patch.dict(os.environ, SOURCE_DATE_EPOCH="0"):
        mpl.rcdefaults()
        if name == RENDERERS[5]:
            path = output / "fig07_local_sensitivity.pdf"
            with patch.object(module, "DATA", root / SENSITIVITY_DATA), \
                    patch.object(module, "PDF", path), redirect_stdout(io.StringIO()):
                module.main()
            files = (path,)
        else:
            if name == RENDERERS[0]:
                arguments = (data[0],)
            elif name == RENDERERS[4]:
                arguments = data
            else:
                arguments = (data,)
            files = module.render_deterministically(*arguments, output)
    assert digest(data) == data_before
    if name != RENDERERS[5]:
        assert len(states) == 2 and states[0] == states[1]
    plt.close("all")
    return {"plot_data_sha256": data_before, "artists_sha256": digest(states[0]),
            "fallback_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
