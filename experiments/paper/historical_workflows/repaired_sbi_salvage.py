"""Compatibility address for experiments.paper.common.repaired_sbi_salvage."""

import sys
from experiments.paper.common import repaired_sbi_salvage as _implementation

sys.modules[__name__] = _implementation
