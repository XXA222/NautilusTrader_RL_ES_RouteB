"""Strict JSON parsing for persisted research registry identities."""
from __future__ import annotations

import json
from typing import Any
from rles_contracts.json_utils import ensure_json_nesting_depth


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def loads_unique_json(value: str | bytes | bytearray) -> Any:
    """Parse JSON recursively and reject ambiguous repeated fields."""
    ensure_json_nesting_depth(value)
    try:
        return json.loads(value, object_pairs_hook=_unique_object,
                          parse_constant=_reject_json_constant)
    except RecursionError as exc:
        raise ValueError("JSON data exceeds the supported nesting depth") from exc
