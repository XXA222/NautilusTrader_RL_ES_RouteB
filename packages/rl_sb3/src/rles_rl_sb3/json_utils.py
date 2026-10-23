"""Strict JSON parsing for RouteB RL checkpoints and frozen actors."""
from __future__ import annotations

from typing import Any

from rles_contracts.json_utils import loads_strict_json


def loads_unique_json(value: str | bytes | bytearray) -> Any:
    """Parse RL state with the shared contracts boundary."""
    return loads_strict_json(value)
