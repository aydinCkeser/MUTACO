"""Compatibility address for experiments.paper.common.protocol_v3."""

import sys
from experiments.paper.common import protocol_v3 as _implementation

sys.modules[__name__] = _implementation
