"""Download and checksum official ADAUSDT archives for the frozen R4 cross-market Fold."""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
from pathlib import Path
from rles_data.json_utils import loads_unique_json
import subprocess
import re
import tempfile
from urllib.parse import urlparse


MONTHS = tuple(
    f"{year}-{month:02d}"
    for year, first, last in ((2023, 7, 12), (2024, 1, 8))
    for month in range(first, last + 1)
)
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"


def _download(url: str, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen archive: {target}")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "data.binance.vision":
        raise ValueError("archive downloader accepts only official Binance HTTPS URLs")
    windows_path = target.resolve()
    drive = windows_path.drive.rstrip(":").lower()
    if len(drive) != 1 or not windows_path.is_absolute():
        raise ValueError("archive target must be an absolute Windows drive path")
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".partial-", delete=False) as handle:
        temporary = Path(handle.name)
    temporary.unlink()
    temp_wsl_path = "/mnt/" + drive + "/" + temporary.resolve().as_posix().split(":/", 1)[1]
    try:
        subprocess.run(["wsl.exe", "curl", "--fail", "--silent", "--show-error",
                        "--location", "--retry", "3", "--max-time", "180",
                        "--output", temp_wsl_path, url], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=240)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"empty Binance response: {url}")
        temporary.replace(target)
    except Exception:
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
    parser.add_argument("--symbol", default="ADAUSDT",
                        choices=("ADAUSDT", "BNBUSDT", "XRPUSDT"))
    parser.add_argument("--run-id", default="fresh-adausdt-r4-e2-e3-20261021")
    parser.add_argument("--protocol-json", type=Path,
                        default=Path("configs/routeb/r4_adausdt_e2_e3_baseline_protocol.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol_json if args.protocol_json.is_absolute() else root / args.protocol_json
    protocol = loads_unique_json(protocol_path.read_bytes())
    symbol = args.symbol
    if symbol != protocol.get("source_symbol") or not args.run_id.startswith("fresh-"):
        raise ValueError("archive symbol/run id differs from the frozen protocol")
    raw = root / ".artifacts/world_model" / args.run_id / "source/raw"
    raw.mkdir(parents=True, exist_ok=True)
    receipt = raw.parent / "official_archive_receipt.json"
    records = []
    for month in MONTHS:
        name = f"{symbol}-1m-{month}.zip"
        url = f"{BASE}/{symbol}/1m/{name}"
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
    receipt_schema = protocol.get("archive_receipt_schema")
    if not isinstance(receipt_schema, str) or not receipt_schema.startswith("routeb.r4-"):
        raise ValueError("frozen protocol must declare an archive receipt schema")
    payload = {"schema": receipt_schema,
               "symbol": symbol, "interval": "1m",
               "source": "Binance USD-M Futures monthly klines",
               "open_time_interval_utc": protocol["source_open_interval_utc"],
               "archives": records, "market_rows_decoded": False}
    receipt.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CHECKSUM_VERIFIED_MARKET_ROWS_UNREAD",
                      "archive_count": len(records), "receipt": str(receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
