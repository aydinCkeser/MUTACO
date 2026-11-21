"""Preserve Python globals recorded before a scientific module was relocated."""

import sys
from types import FunctionType, ModuleType


def preserve_legacy_definition(value: FunctionType | type, old_name: str) -> None:
    """Keep the original pickle globals of one relocated definition and its methods."""
    defined_in = value.__module__
    if isinstance(value, type):
        for member in vars(value).values():
            if isinstance(member, property):
                functions = (member.fget, member.fset, member.fdel)
            elif isinstance(member, (staticmethod, classmethod)):
                functions = (member.__func__,)
            else:
                functions = (member,)
            for function in functions:
                if isinstance(function, FunctionType) and function.__module__ == defined_in:
                    function.__module__ = old_name
    value.__module__ = old_name


def bind_legacy_module(module: ModuleType, old_name: str) -> ModuleType:
    """Expose the same loaded module and keep legacy pickle names on local definitions."""
    sys.modules[old_name] = module
    for value in vars(module).values():
        if isinstance(value, (FunctionType, type)) and value.__module__ == module.__name__:
            preserve_legacy_definition(value, old_name)
    return module
