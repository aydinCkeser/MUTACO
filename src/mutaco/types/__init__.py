"""Dimension-generic physical parameter and response declarations."""

import sys as _sys

from mutaco.parameter_and_response import parameter_response_specifications as specifications
from mutaco.parameter_and_response.parameter_response_specifications import (
    ParameterAxis,
    ParameterSpec,
    ResponseAxis,
    ResponseSpec,
)

_sys.modules[__name__ + ".specifications"] = specifications

__all__ = ["ParameterAxis", "ParameterSpec", "ResponseAxis", "ResponseSpec"]
