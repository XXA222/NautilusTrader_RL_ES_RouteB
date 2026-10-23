"""Strict JSON parsing for RouteB data markers and external archive rows."""
from __future__ import annotations

import json
from typing import Any, Callable
from rles_contracts.json_utils import ensure_json_nesting_depth


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise json.JSONDecodeError(f"duplicate JSON object key: {key!r}", "", 0)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def loads_unique_json(value: str | bytes | bytearray, *,
                      parse_constant: Callable[[str], Any] | None = None) -> Any:
    """Parse JSON recursively, optionally overriding non-finite constant handling."""
    ensure_json_nesting_depth(value)
    options: dict[str, Any] = {"object_pairs_hook": _unique_object}
    options["parse_constant"] = _reject_json_constant if parse_constant is None else parse_constant
    try:
        return json.loads(value, **options)
    except RecursionError as exc:
        raise ValueError("JSON data exceeds the supported nesting depth") from exc

