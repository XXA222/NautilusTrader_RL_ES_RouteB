"""Cross-version JSON input guards shared by RLES package readers."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

MAX_JSON_NESTING_DEPTH = 512


def ensure_json_nesting_depth(value: str | bytes | bytearray, *, max_depth: int = MAX_JSON_NESTING_DEPTH) -> None:
    """Reject excessive structural nesting consistently across Python versions.

    This lightweight scan ignores delimiters inside quoted strings. It bounds
    parser work but deliberately leaves JSON syntax validation to ``json.loads``.
    """
    if type(max_depth) is not int or max_depth < 1:
        raise ValueError("maximum JSON nesting depth must be a positive integer")
    if type(value) not in (str, bytes, bytearray):
        raise TypeError("JSON input must be text or bytes")

    binary = not isinstance(value, str)
    quote = 34 if binary else '"'
    escape = 92 if binary else "\\"
    openers = (91, 123) if binary else ("[", "{")
    closers = (93, 125) if binary else ("]", "}")
    depth = 0
    in_string = False
    escaped = False

    for token in value:
        if in_string:
            if escaped:
                escaped = False
            elif token == escape:
                escaped = True
            elif token == quote:
                in_string = False
            continue
        if token == quote:
            in_string = True
        elif token in openers:
            depth += 1
            if depth > max_depth:
                raise ValueError("JSON data exceeds the supported nesting depth")
        elif token in closers:
            depth -= 1


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise json.JSONDecodeError(f"duplicate JSON object key: {key}", "", 0)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def loads_strict_json(
    value: str | bytes | bytearray,
    *,
    parse_constant: Callable[[str], Any] | None = None,
    max_depth: int = MAX_JSON_NESTING_DEPTH,
) -> Any:
    """Decode JSON with shared depth, duplicate-key, and constant handling.

    Callers may override constant handling only when their schema needs a more
    specific diagnostic for NaN/Infinity. Duplicate object keys are always
    rejected by this shared boundary.
    """
    ensure_json_nesting_depth(value, max_depth=max_depth)
    try:
        return json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant if parse_constant is None else parse_constant,
        )
    except RecursionError as exc:
        raise ValueError("JSON data exceeds the supported nesting depth") from exc
