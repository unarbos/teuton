"""V3-owned wrapper for the legacy v2 task implementation.

The implementation lives under `teuton_legacy_v2` so the legacy task surface
remains available from the v3 checkout.
"""
from __future__ import annotations

from teuton_legacy_v2.tasks.pluralis_lossless_wire import *  # noqa: F401,F403
