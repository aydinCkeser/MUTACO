"""Legacy checkout import of the saved-evidence verification API."""
import sys
from reproducibility import verification as _implementation

sys.modules[__name__] = _implementation
