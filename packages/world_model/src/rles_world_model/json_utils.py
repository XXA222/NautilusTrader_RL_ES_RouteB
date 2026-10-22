"""Strict JSON helpers for world-model artifacts and request files."""
from __future__ import annotations

import json
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def loads_unique_json(value: str | bytes | bytearray, *,
                      allow_nonstandard_constants: bool = False) -> Any:
    """Parse JSON while rejecting ambiguous duplicate object keys recursively.

    Non-standard numeric constants remain rejected unless a caller explicitly
    opts in so it can produce a more specific field-validation error.
    """
    parse_constant = float if allow_nonstandard_constants else _reject_json_constant
    return json.loads(value, object_pairs_hook=_unique_object,
                      parse_constant=parse_constant)
