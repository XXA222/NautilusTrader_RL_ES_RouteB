from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen

SYMBOL = "AVAXUSDT"
MONTHS = ("2024-01", "2024-02", "2024-03", "2024-04")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch(url: str, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite downloaded source: {target}")
    request = Request(url, headers={"User-Agent": "RouteB-research-archive-fetch/1.0"})
    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"archive server returned HTTP {response.status} for {url}")
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".partial-", delete=False) as temp:
            temporary = Path(temp.name)
            try:
                while block := response.read(1024 * 1024):
                    temp.write(block)
                temp.flush()
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"empty archive response: {url}")
    temporary.replace(target)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    raw = root / ".artifacts" / "world_model" / "fresh-avaxusdt-20240101-r3-20261021" / "source" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for month in MONTHS:
        name = f"{SYMBOL}-1m-{month}.zip"
        archive_url = f"{BASE_URL}/{SYMBOL}/1m/{name}"
        checksum_url = archive_url + ".CHECKSUM"
        archive = raw / name
        sidecar = raw / (name + ".CHECKSUM")
        if not sidecar.exists():
            fetch(checksum_url, sidecar)
        sidecar_text = sidecar.read_text(encoding="ascii").strip()
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\s]+)", sidecar_text)
        if match is None or Path(match.group(2)).name != name:
            raise ValueError(f"unexpected official checksum format for {name}")
        expected = match.group(1).lower()
        if not archive.exists():
            fetch(archive_url, archive)
        actual = digest(archive)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
        records.append({"file": name, "url": archive_url, "checksum_url": checksum_url,
                        "sha256": actual, "bytes": archive.stat().st_size,
                        "payload_decoded": False})
    receipt = {"schema": "routeb.r3.official-archive-receipt.v1", "symbol": SYMBOL,
              "interval": "1m", "source": "Binance USD-M futures monthly klines",
              "archives": records, "market_rows_decoded": False}
    target = raw.parent / "official_archive_receipt.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite archive receipt: {target}")
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CHECKSUM_VERIFIED_MARKET_ROWS_UNREAD",
                      "archive_count": len(records), "archives": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
