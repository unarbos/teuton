"""Legacy compatibility hook.

Teuton v3 is self-contained; this module remains only for older callers that
imported the hook while the v3 package was being split out.
"""
from __future__ import annotations


def ensure_v2_path() -> None:
    return None
