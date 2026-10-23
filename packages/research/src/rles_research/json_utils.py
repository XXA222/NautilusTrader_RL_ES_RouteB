"""Compatibility entry point for strict JSON parsing in research workflows."""
from __future__ import annotations

from typing import Any

from rles_contracts.serialization import strict_json_loads


def loads_unique_json(value: str | bytes | bytearray) -> Any:
    """Parse research state with the shared contracts JSON boundary."""
    if isinstance(value, bytearray):
        value = bytes(value)
    return strict_json_loads(value, context="research JSON data")
