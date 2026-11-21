"""Compatibility address for experiments.paper.figures.style_registry."""

import sys
from experiments.paper.figures import style_registry as _implementation

sys.modules[__name__] = _implementation
