"""Download checksum-verified official SOLUSDT archives for the frozen R4 Fold."""
from __future__ import annotations

from datetime import date
from hashlib import sha256
import argparse
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
    request = Request(url, headers={"User-Agent": "RouteB-R4-SOL-archive/1.0"})
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


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path,
                        default=Path("configs/routeb/r4_solusdt_e2_e3_baseline_protocol.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol_json if args.protocol_json.is_absolute() else root / args.protocol_json
    protocol = loads_unique_json(protocol_path.read_bytes())
    if (protocol.get("schema") != "routeb.r4-e2-e3-baseline-comparison.v1"
            or protocol.get("source_symbol") != "SOLUSDT"
            or protocol.get("archive_receipt_schema")
            != "routeb.r4-sol-e2-e3.official-archive-receipt.v1"):
        raise ValueError("downloader requires the frozen SOL R4 protocol")
    start = date.fromisoformat(protocol["source_open_interval_utc"][0][:10])
    end = date.fromisoformat(protocol["source_open_interval_utc"][1][:10])
    months = _months(start, end)
    run_id = "fresh-solusdt-r4-e2-e3-20261021"
    raw = root / ".artifacts/world_model" / run_id / "source/raw"
    raw.mkdir(parents=True, exist_ok=True)
    receipt = raw.parent / "official_archive_receipt.json"
    records = []
    for month in months:
        name = f"SOLUSDT-1m-{month}.zip"
        url = f"{BASE}/SOLUSDT/1m/{name}"
        archive, checksum = raw / name, raw / f"{name}.CHECKSUM"
        if not checksum.exists():
            _download(url + ".CHECKSUM", checksum)
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\s]+)",
                             checksum.read_text(encoding="ascii").strip())
        if match is None or Path(match.group(2)).name != name:
            raise ValueError(f"invalid official Binance CHECKSUM for {name}")
        if not archive.exists():
            _download(url, archive)
        actual = _digest(archive)
        if actual != match.group(1).lower():
            raise ValueError(f"Binance SHA-256 mismatch for {name}")
        records.append({"file": name, "url": url, "checksum_url": url + ".CHECKSUM",
                        "sha256": actual, "bytes": archive.stat().st_size,
                        "market_rows_decoded": False})
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite archive receipt: {receipt}")
    payload = {"schema": protocol["archive_receipt_schema"], "symbol": "SOLUSDT",
               "interval": "1m", "source": "Binance USD-M Futures monthly klines",
               "open_time_interval_utc": protocol["source_open_interval_utc"],
               "archives": records, "market_rows_decoded": False}
    receipt.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CHECKSUM_VERIFIED_MARKET_ROWS_UNREAD",
                      "archive_count": len(records), "receipt": str(receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
