"""Compatibility import for mutaco.file_management.atomic_file_writes."""

import sys

from mutaco.file_management import atomic_file_writes as _implementation

sys.modules[__name__] = _implementation
