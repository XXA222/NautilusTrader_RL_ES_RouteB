from pathlib import Path

path = Path(r"D:\Program Files\NautilusTrader_RL_ES_RouteB\packages\data\src\rles_data\binance_archive.py")
raw = path.read_bytes()
newline = b"\r\n" if b"\r\n" in raw else b"\n"
needle = b"\ndef _archive_int(value: str, *, field: str) -> int:" + newline
helper = (
    b"\ndef _archive_day(value: object, *, field: str) -> date:\n"
    b"    \"\"\"Parse one canonical UTC archive date at the public API boundary.\"\"\"\n"
    b"    if type(value) is not str:\n"
    b"        raise DataError(f\"{field} must be YYYY-MM-DD\")\n"
    b"    try:\n"
    b"        parsed = date.fromisoformat(value)\n"
    b"    except ValueError as exc:\n"
    b"        raise DataError(f\"{field} must be YYYY-MM-DD\") from exc\n"
    b"    if parsed.isoformat() != value:\n"
    b"        raise DataError(f\"{field} must be YYYY-MM-DD\")\n"
    b"    return parsed\n\n"
    b"\ndef _archive_int(value: str, *, field: str) -> int:"
    + newline
)
if raw.count(needle) != 1:
    raise SystemExit("archive integer parser insertion point is ambiguous")
raw = raw.replace(needle, helper, 1)
old_day = (
    b"    parsed_day = date.fromisoformat(day)\n"
    b"    if parsed_day.isoformat() != day:\n"
    b"        raise DataError(\"day must be YYYY-MM-DD\")\n"
)
new_day = b"    parsed_day = _archive_day(day, field=\"day\")\n"
if raw.count(old_day) != 1:
    raise SystemExit("single-day parser block is ambiguous")
raw = raw.replace(old_day, new_day, 1)
old_range = (
    b"    first, last = date.fromisoformat(start_day), date.fromisoformat(end_day)\n"
    b"    if first.isoformat() != start_day or last.isoformat() != end_day or last < first:\n"
    b"        raise DataError(\"range must use ordered YYYY-MM-DD start/end dates\")\n"
)
new_range = (
    b"    first = _archive_day(start_day, field=\"start_day\")\n"
    b"    last = _archive_day(end_day, field=\"end_day\")\n"
    b"    if last < first:\n"
    b"        raise DataError(\"range must use ordered YYYY-MM-DD start/end dates\")\n"
)
if raw.count(old_range) != 1:
    raise SystemExit("date-range parser block is ambiguous")
path.write_bytes(raw.replace(old_range, new_range, 1))
