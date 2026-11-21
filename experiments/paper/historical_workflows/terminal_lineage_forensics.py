"""Compatibility address for experiments.paper.common.terminal_lineage_forensics."""

import sys
from experiments.paper.common import terminal_lineage_forensics as _implementation

sys.modules[__name__] = _implementation
