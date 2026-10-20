from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import re
from pathlib import Path

from rles_data.binance_archive import BASE, _read_remote, fetch_archive


def _day(source: date, symbol: str, raw: Path) -> dict[str, str | int]:
    name = f"{symbol}-1m-{source:%Y-%m-%d}.zip"
    relative = f"daily/klines/{symbol}/1m/{name}"
    destination = raw / name
    receipt = fetch_archive(relative, destination)
    sidecar_url = BASE + relative + ".CHECKSUM"
    content = _read_remote(sidecar_url, limit=1024, temp_parent=raw)
    fields = content.decode("ascii").split()
    if (len(fields) != 2 or fields[1].lstrip("*") != name
            or fields[0] != receipt["sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None):
        raise RuntimeError(f"exchange checksum sidecar mismatch for {name}")
    sidecar = destination.with_suffix(destination.suffix + ".CHECKSUM")
    if sidecar.exists() and sidecar.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite different checksum sidecar: {sidecar}")
    if not sidecar.exists():
        sidecar.write_bytes(content)
    return {"file": name, "sha256": receipt["sha256"], "bytes": destination.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download checksum-verified Binance USD-M daily klines.")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day-exclusive", required=True, type=date.fromisoformat)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if (not re.fullmatch(r"[A-Z0-9]{3,24}", args.symbol)
            or args.end_day_exclusive <= args.start_day
            or not 1 <= args.workers <= 12):
        raise SystemExit("invalid symbol, date interval, or worker count")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    days = []
    day = args.start_day
    while day < args.end_day_exclusive:
        days.append(day)
        day += timedelta(days=1)
    receipts = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_day, day, args.symbol, args.raw_dir) for day in days]
        for index, future in enumerate(as_completed(futures), start=1):
            receipts.append(future.result())
            if index % 10 == 0 or index == len(days):
                print(f"verified {index}/{len(days)} daily archives", flush=True)
    receipts.sort(key=lambda row: row["file"])
    print(f"complete: {len(receipts)} archives, "
          f"{sum(int(row['bytes']) for row in receipts)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
