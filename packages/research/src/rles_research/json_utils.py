"""Strict JSON parsing for persisted research registry identities."""
from __future__ import annotations

from typing import Any

from rles_contracts.json_utils import loads_strict_json


def loads_unique_json(value: str | bytes | bytearray) -> Any:
    """Parse research state with the shared contracts JSON boundary."""
    return loads_strict_json(value)
