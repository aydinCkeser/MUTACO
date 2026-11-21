"""Compatibility address for experiments.paper.common.repaired_sbi_recovery."""

import sys
from experiments.paper.common import repaired_sbi_recovery as _implementation

sys.modules[__name__] = _implementation
