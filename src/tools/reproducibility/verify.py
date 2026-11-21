"""Legacy checkout command and trusted pickle address for the verifier."""
import sys
from reproducibility import verify as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
