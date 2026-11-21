"""Compatibility address for experiments.paper.figures.figure_artifacts."""

import sys
from experiments.paper.figures import figure_artifacts as _implementation

sys.modules[__name__] = _implementation
