"""Strict JSON parsing helpers for persisted RouteB orchestration inputs."""
from __future__ import annotations

from typing import Any

from rles_contracts.json_utils import loads_strict_json


def loads_unique_json(value: str | bytes | bytearray) -> Any:
    """Parse orchestration JSON with the shared contracts boundary."""
    return loads_strict_json(value)
