"""Compatibility address for experiments.paper.common.repaired_sbi_terminal_archive."""

import sys
from experiments.paper.common import repaired_sbi_terminal_archive as _implementation

sys.modules[__name__] = _implementation
