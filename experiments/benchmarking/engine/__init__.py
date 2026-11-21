"""Generic benchmark execution engines, loaded lazily for method isolation."""

from typing import Any

__all__ = ["replay_indices", "run_reference_cohort"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import reference

    return getattr(reference, name)
