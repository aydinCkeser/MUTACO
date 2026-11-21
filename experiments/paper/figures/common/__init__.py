"""Historical imports for shared figure helpers."""

import sys as _sys

from .. import plot_validation, target_selection

# These two helpers need only the standard library. Plot style and evidence
# readers keep their on-demand imports so they retain optional dependency isolation.
_sys.modules[__name__ + ".plot_validation"] = plot_validation
_sys.modules[__name__ + ".target_selection"] = target_selection
