"""Preserve importable Python GLOBAL addresses after moving an experiment module."""

from types import FunctionType
from typing import Any


def preserve_pickle_addresses(namespace: dict[str, Any], previous_module: str) -> None:
    """Keep historical pickle names on definitions owned by this module."""
    current = namespace["__name__"]
    if current == "__main__":
        return
    for value in namespace.values():
        if isinstance(value, (type, FunctionType)) and value.__module__ == current:
            value.__module__ = previous_module
