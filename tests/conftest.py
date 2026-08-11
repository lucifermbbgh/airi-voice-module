"""Pytest conftest — fixes Hermes PYTHONPATH contamination.

The Hermes runtime injects its own Python 3.11 site-packages into
PYTHONPATH, which causes numpy C-extension mismatches when running
under this project's Python 3.14 venv.

This conftest strips those entries from sys.path before test collection.
"""

import os
import sys

# Remove Hermes-injected paths that break the project venv
HERMES_HOME = os.path.expanduser("~/.hermes")
_removed = []
_keep = []

for p in sys.path:
    if p and HERMES_HOME in p:
        _removed.append(p)
    else:
        _keep.append(p)

if _removed:
    sys.path[:] = _keep
