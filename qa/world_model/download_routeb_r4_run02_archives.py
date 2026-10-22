"""Download and checksum a disjoint early AVAX source for R4 run 02."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen


SYMBOL = "AVAXUSDT"
MONTHS = tuple(
    f"{year}-{month:02d}"
    for year, first, last in ((2020, 11, 12), (2021, 1, 12))
    for month in range(first, last + 1)
)
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"


def _download(url: str, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite R4 run-02 archive: {target}")
    request = Request(url, headers={"User-Agent": "RouteB-R4-run02-archive/1.0"})
    with urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".partial-", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                handle.flush()
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"empty Binance response: {url}")
    temporary.replace(target)


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    raw = root / ".artifacts/world_model/fresh-avaxusdt-20201126-r4-run02-20261021/source/raw"
    raw.mkdir(parents=True, exist_ok=True)
    receipt = raw.parent / "official_archive_receipt.json"
    records = []
    for month in MONTHS:
        name = f"{SYMBOL}-1m-{month}.zip"
        url = f"{BASE}/{SYMBOL}/1m/{name}"
        archive, checksum = raw / name, raw / f"{name}.CHECKSUM"
        if not checksum.exists():
            _download(url + ".CHECKSUM", checksum)
        text = checksum.read_text(encoding="ascii").strip()
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\s]+)", text)
        if match is None or Path(match.group(2)).name != name:
            raise ValueError(f"invalid official Binance CHECKSUM for {name}")
        if not archive.exists():
            _download(url, archive)
        actual = _digest(archive)
        if actual != match.group(1).lower():
            raise ValueError(f"Binance SHA-256 mismatch for {name}")
        records.append({"file": name, "url": url, "checksum_url": url + ".CHECKSUM",
                        "sha256": actual, "bytes": archive.stat().st_size,
                        "payload_decoded": False})
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite archive receipt: {receipt}")
    payload = {"schema": "routeb.r4-run02.official-archive-receipt.v1", "symbol": SYMBOL,
               "interval": "1m", "source": "Binance USD-M futures monthly klines",
               "open_time_interval_utc": ["2020-11-20T00:00:00Z", "2021-12-23T00:01:00Z"],
               "archives": records, "market_rows_decoded": False}
    receipt.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CHECKSUM_VERIFIED_MARKET_ROWS_UNREAD",
                      "archive_count": len(records), "receipt": str(receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
