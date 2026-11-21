"""Typed, human-editable visual grammar for paper figures."""

from __future__ import annotations

from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping

from matplotlib.colors import is_color_like
from matplotlib.markers import MarkerStyle
import matplotlib as mpl
import matplotlib.pyplot as plt

from experiments.benchmarking.artifacts import file_hash


SCHEMA_VERSION = "mutaco-paper-visual-grammar-v1"
REQUIRED_ROLES = frozenset(
    {
        "true",
        "mutaco",
        "npe",
        "true_uncertainty",
        "ensemble_mean",
        "ensemble_uncertainty",
        "individual_realization",
    }
)
ROLE_KEYS = frozenset(
    {
        "color",
        "linestyle",
        "marker",
        "marker_face",
        "marker_edge",
        "marker_size",
        "marker_edge_width",
        "line_width",
        "alpha",
        "zorder",
    }
)
MATPLOTLIB_KEYS = {
    "marker_face": "markerfacecolor",
    "marker_edge": "markeredgecolor",
    "marker_size": "markersize",
    "marker_edge_width": "markeredgewidth",
    "line_width": "linewidth",
}
LINESTYLES = frozenset({"-", "--", "-.", ":", "None", "none", ""})


def _positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    result = float(value)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _validate_role(role_id: str, values: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(values) - ROLE_KEYS
    if unknown:
        raise ValueError(f"unknown keys for visual role {role_id}: {sorted(unknown)}")
    if "color" not in values or not is_color_like(values["color"]):
        raise ValueError(f"visual role {role_id} requires a valid color")
    if "linestyle" in values and str(values["linestyle"]) not in LINESTYLES:
        raise ValueError(f"invalid linestyle for visual role {role_id}")
    for key in ("marker_face", "marker_edge"):
        if key in values and not is_color_like(values[key]):
            raise ValueError(f"invalid {key} for visual role {role_id}")
    if "marker" in values:
        try:
            MarkerStyle(values["marker"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid marker for visual role {role_id}") from error
    for key in ("marker_size", "marker_edge_width", "line_width"):
        if key in values:
            _positive(f"{role_id}.{key}", values[key], allow_zero=key != "marker_size")
    if "alpha" in values and not 0.0 <= float(values["alpha"]) <= 1.0:
        raise ValueError(f"{role_id}.alpha must lie in [0,1]")
    if "zorder" in values:
        float(values["zorder"])
    return dict(values)


@dataclass(frozen=True)
class PaperStyle:
    profile_id: str
    grammar_path: Path
    mplstyle_path: Path
    width_inches: float
    height_inches: float
    roles: Mapping[str, Mapping[str, Any]]
    role_overrides: Mapping[str, Mapping[str, Any]]

    def context(self) -> AbstractContextManager[Any]:
        return plt.style.context(self.mplstyle_path)

    def role(self, role_id: str) -> dict[str, Any]:
        if role_id not in self.roles:
            raise ValueError(f"unknown visual role: {role_id}")
        values = dict(self.roles[role_id])
        values.update(self.role_overrides.get(role_id, {}))
        validated = _validate_role(role_id, values)
        return {MATPLOTLIB_KEYS.get(key, key): value for key, value in validated.items()}

    def provenance(self, repository_root: Path) -> dict[str, Any]:
        resolved = {role_id: self.role(role_id) for role_id in sorted(self.roles)}
        with self.context():
            effective_rcparams = {
                key: list(mpl.rcParams[key])
                if isinstance(mpl.rcParams[key], (list, tuple))
                else mpl.rcParams[key]
                for key in (
                    "figure.figsize",
                    "font.size",
                    "font.family",
                    "font.serif",
                    "mathtext.fontset",
                    "axes.labelsize",
                    "legend.fontsize",
                    "xtick.labelsize",
                    "ytick.labelsize",
                    "pdf.fonttype",
                    "svg.fonttype",
                    "savefig.dpi",
                )
            }
        return {
            "schema_version": "mutaco-paper-resolved-style-v1",
            "profile_id": self.profile_id,
            "canvas_inches": [self.width_inches, self.height_inches],
            "grammar_path": self.grammar_path.relative_to(repository_root).as_posix(),
            "grammar_sha256": file_hash(self.grammar_path),
            "mplstyle_path": self.mplstyle_path.relative_to(repository_root).as_posix(),
            "mplstyle_sha256": file_hash(self.mplstyle_path),
            "role_overrides": deepcopy(dict(self.role_overrides)),
            "resolved_roles": resolved,
            "effective_rcparams": effective_rcparams,
        }


def load_paper_style(
    grammar_path: Path,
    *,
    profile_id: str | None = None,
    role_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> PaperStyle:
    grammar_path = grammar_path.resolve()
    document = tomllib.loads(grammar_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown paper visual-grammar schema")
    profiles = document.get("profiles")
    roles = document.get("roles")
    if not isinstance(profiles, dict) or not isinstance(roles, dict):
        raise ValueError("visual grammar requires profiles and roles")
    selected = profile_id or document.get("default_profile")
    if selected not in profiles:
        raise ValueError(f"unknown paper style profile: {selected}")
    if set(roles) != REQUIRED_ROLES:
        missing = sorted(REQUIRED_ROLES - set(roles))
        unknown = sorted(set(roles) - REQUIRED_ROLES)
        raise ValueError(f"visual roles mismatch; missing={missing}, unknown={unknown}")
    validated_roles = {
        role_id: _validate_role(role_id, values) for role_id, values in roles.items()
    }
    overrides = deepcopy(dict(role_overrides or {}))
    unknown_overrides = set(overrides) - REQUIRED_ROLES
    if unknown_overrides:
        raise ValueError(f"unknown overridden visual roles: {sorted(unknown_overrides)}")
    for role_id, values in overrides.items():
        if not isinstance(values, dict):
            raise ValueError(f"override for {role_id} must be a table")
        _validate_role(role_id, {**validated_roles[role_id], **values})
    profile = profiles[selected]
    mplstyle_path = (grammar_path.parent / profile["mplstyle"]).resolve()
    if not mplstyle_path.is_file():
        raise ValueError(f"missing mplstyle for profile {selected}")
    width = _positive("width_inches", profile["width_inches"])
    height = _positive("height_inches", profile["height_inches"])
    return PaperStyle(
        profile_id=str(selected),
        grammar_path=grammar_path,
        mplstyle_path=mplstyle_path,
        width_inches=width,
        height_inches=height,
        roles=validated_roles,
        role_overrides=overrides,
    )

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.figures.common.style_registry")
