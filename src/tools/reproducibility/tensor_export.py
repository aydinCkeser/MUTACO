"""Legacy checkout import of the original tensor evidence exporter."""
import sys
from reproducibility import tensor_export as _implementation

sys.modules[__name__] = _implementation
