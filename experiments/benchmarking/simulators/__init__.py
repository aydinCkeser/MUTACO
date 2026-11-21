"""Legacy registry address and native file for the source-pinned graph adapter."""

import sys

# Load this pinned child while the legacy package still owns its search path.
from . import graph_scientific_d4
from .. import simulator_adapters as _registry

sys.modules[__name__] = _registry
