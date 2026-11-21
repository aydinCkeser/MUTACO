"""Paper campaign declarations, freeze and assessment protocols."""

from importlib import import_module
import sys

# Preserve the legacy module-name object used by the shared exception and by
# pre-migration pickles. Only this standard-library validator loads eagerly.
validation = import_module("experiments.paper.common.validation")
sys.modules[__name__ + ".validation"] = validation
