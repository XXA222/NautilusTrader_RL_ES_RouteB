"""Strict JSON parsing for RouteB data markers and external archive rows."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rles_contracts.json_utils import loads_strict_json


def loads_unique_json(
    value: str | bytes | bytearray,
    *,
    parse_constant: Callable[[str], Any] | None = None,
) -> Any:
    """Compatibility wrapper retaining caller-specific constant diagnostics."""
    return loads_strict_json(value, parse_constant=parse_constant)
