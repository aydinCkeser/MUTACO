"""Compatibility address for experiments.paper.figures.validated_inputs."""

import sys
from experiments.paper.figures import validated_inputs as _implementation

sys.modules[__name__] = _implementation
