"""Strict JSON helpers for world-model artifacts and request files."""
from __future__ import annotations

from typing import Any

from rles_contracts.json_utils import loads_strict_json


def loads_unique_json(
    value: str | bytes | bytearray,
    *,
    allow_nonstandard_constants: bool = False,
) -> Any:
    """Use shared parsing while preserving schema-specific NaN diagnostics."""
    return loads_strict_json(
        value,
        parse_constant=float if allow_nonstandard_constants else None,
    )
