"""Compatibility address for experiments.paper.common.population_v3."""

import sys
from experiments.paper.common import population_v3 as _implementation

sys.modules[__name__] = _implementation
