"""Cross-version JSON input guards shared by RLES package readers."""
from __future__ import annotations

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
