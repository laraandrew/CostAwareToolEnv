"""Compatibility shim for the legacy top-level import path.

The real environment implementation lives in :mod:`env.environment`.
This module stays intentionally thin so the two orchestrator entrypoints
cannot drift apart again.
"""
from __future__ import annotations

from env.environment import *  # noqa: F401,F403
