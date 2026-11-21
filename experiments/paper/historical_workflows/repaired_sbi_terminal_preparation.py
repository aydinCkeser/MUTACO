"""Compatibility address for experiments.paper.common.repaired_sbi_terminal_preparation."""

import sys
from experiments.paper.common import repaired_sbi_terminal_preparation as _implementation

sys.modules[__name__] = _implementation
