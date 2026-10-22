"""Strict JSON parsing for RouteB data markers and external archive rows."""
from __future__ import annotations

import json
from typing import Any, Callable


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
    options: dict[str, Any] = {"object_pairs_hook": _unique_object}
    options["parse_constant"] = _reject_json_constant if parse_constant is None else parse_constant
    return json.loads(value, **options)

