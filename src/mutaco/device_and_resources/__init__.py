"""Torch placement, CPU parallelism and run-time resource measurements."""

import sys as _sys

from mutaco.observability import monitoring as resource_monitoring
from mutaco.runtime import parallelism as cpu_parallelism_provenance
from mutaco.runtime import resolution as device_and_precision_resolution

# The implementations live here on disk. Loading them through their historical
# names keeps pickle globals intact, including when monitoring imports runtime.
_sys.modules[__name__ + ".cpu_parallelism_provenance"] = cpu_parallelism_provenance
_sys.modules[__name__ + ".device_and_precision_resolution"] = device_and_precision_resolution
_sys.modules[__name__ + ".resource_monitoring"] = resource_monitoring
