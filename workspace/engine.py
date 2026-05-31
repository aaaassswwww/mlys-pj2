"""Compatibility wrapper for the evaluator entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import Engine, create_engine

__all__ = ["Engine", "create_engine"]
