"""Download checksum-verified official DOGEUSDT archives for the frozen R4 Fold."""
from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from rles_data.json_utils import loads_unique_json
import re
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


BASE = "https://data.binance.vision/data/futures/um/monthly/klines"


def _months(start: date, end_exclusive: date) -> tuple[str, ...]:
    current = date(start.year, start.month, 1)
    final = date(end_exclusive.year, end_exclusive.month, 1)
    values = []
    while current <= final:
        values.append(f"{current.year}-{current.month:02d}")
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return tuple(values)


def _download(url: str, target: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "data.binance.vision":
        raise ValueError("archive downloader accepts only official Binance HTTPS URLs")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen archive: {target}")
    temporary = None
    request = Request(url, headers={"User-Agent": "RouteB-R4-DOGE-archive/1.0"})
    try:
        with urlopen(request, timeout=90) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {url}")
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".partial-",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                handle.flush()
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"empty Binance response: {url}")
        temporary.replace(target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path,
                        default=Path("configs/routeb/r4_dogeusdt_e2_e3_baseline_protocol.json"))
    parser.add_argument("--run-id", default="fresh-dogeusdt-r4-e2-e3-20261022")
    args = parser.parse_args()
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", args.run_id) is None:
        raise ValueError("run id must be a lowercase alphanumeric hyphenated slug")
    protocol_path = args.protocol_json if args.protocol_json.is_absolute() else root / args.protocol_json
    protocol = loads_unique_json(protocol_path.read_bytes())
    if (protocol.get("schema") != "routeb.r4-e2-e3-baseline-comparison.v1"
            or protocol.get("source_symbol") != "DOGEUSDT"
            or protocol.get("archive_receipt_schema")
            != "routeb.r4-doge-e2-e3.official-archive-receipt.v1"):
        raise ValueError("downloader requires the frozen DOGEUSDT R4 protocol")
    start = date.fromisoformat(protocol["source_open_interval_utc"][0][:10])
    end = date.fromisoformat(protocol["source_open_interval_utc"][1][:10])
    months = _months(start, end)
    if len(months) != 14:
        raise ValueError(f"frozen source should require 14 monthly archives, got {len(months)}")
    raw = root / ".artifacts/world_model" / args.run_id / "source/raw"
    raw.mkdir(parents=True, exist_ok=True)
    receipt = raw.parent / "official_archive_receipt.json"
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite frozen archive receipt: {receipt}")
    records = []
    for month in months:
        name = f"DOGEUSDT-1m-{month}.zip"
        url = f"{BASE}/DOGEUSDT/1m/{name}"
        archive, checksum = raw / name, raw / f"{name}.CHECKSUM"
        if not checksum.exists():
            _download(url + ".CHECKSUM", checksum)
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\s]+)",
                             checksum.read_text(encoding="ascii").strip())
        if match is None or Path(match.group(2)).name != name:
            raise ValueError(f"invalid official Binance CHECKSUM for {name}")
        if not archive.exists():
            _download(url, archive)
        digest = sha256(archive.read_bytes()).hexdigest()
        if digest != match.group(1).lower():
            raise ValueError(f"Binance SHA-256 mismatch for {name}")
        records.append({"file": name, "url": url, "checksum_url": url + ".CHECKSUM",
                        "sha256": digest, "bytes": archive.stat().st_size,
                        "market_rows_decoded": False})
    payload = {"schema": protocol["archive_receipt_schema"], "symbol": "DOGEUSDT",
               "interval": "1m", "source": "Binance USD-M Futures monthly klines",
               "open_time_interval_utc": protocol["source_open_interval_utc"],
               "archives": records, "market_rows_decoded": False}
    receipt.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CHECKSUM_VERIFIED_MARKET_ROWS_UNREAD",
                      "archive_count": len(records), "receipt": str(receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
