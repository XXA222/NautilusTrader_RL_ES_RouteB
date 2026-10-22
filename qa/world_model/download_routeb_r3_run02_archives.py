from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import Request, urlopen

SYMBOL = "AVAXUSDT"
MONTHS = ("2024-04", "2024-05", "2024-06", "2024-07")
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"


def checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    request = Request(url, headers={"User-Agent": "RouteB-R3-run02-archive/1.0"})
    with urlopen(request, timeout=60) as response:
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
        raise RuntimeError(f"empty archive response: {url}")
    temporary.replace(target)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    artifact = root / ".artifacts/world_model/fresh-avaxusdt-20240415-r3-20261021"
    raw = artifact / "source/raw"
    raw.mkdir(parents=True, exist_ok=True)
    prior_raw = root / ".artifacts/world_model/fresh-avaxusdt-20240101-r3-20261021/source/raw"
    records = []
    for month in MONTHS:
        name = f"{SYMBOL}-1m-{month}.zip"
        url = f"{BASE}/{SYMBOL}/1m/{name}"
        sidecar_url = url + ".CHECKSUM"
        archive, sidecar = raw / name, raw / (name + ".CHECKSUM")
        prior_archive = prior_raw / name
        prior_sidecar = prior_raw / (name + ".CHECKSUM")
        if month == "2024-04" and prior_archive.is_file() and prior_sidecar.is_file():
            shutil.copyfile(prior_archive, archive)
            shutil.copyfile(prior_sidecar, sidecar)
        else:
            download(sidecar_url, sidecar)
            download(url, archive)
        expected_text = sidecar.read_text(encoding="ascii").strip()
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\s]+)", expected_text)
        if match is None or Path(match.group(2)).name != name:
            raise ValueError(f"invalid Binance sidecar for {name}")
        actual = checksum(archive)
        if actual != match.group(1).lower():
            raise ValueError(f"SHA-256 mismatch for {name}")
        records.append({"file": name, "url": url, "checksum_url": sidecar_url,
                        "sha256": actual, "bytes": archive.stat().st_size,
                        "payload_decoded": False})
    receipt = {"schema": "routeb.r3.run02.official-archive-receipt.v1",
               "symbol": SYMBOL, "interval": "1m",
               "source": "Binance USD-M futures monthly klines",
               "archives": records, "market_rows_decoded": False}
    target = artifact / "source/official_archive_receipt.json"
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CHECKSUM_VERIFIED_MARKET_ROWS_UNREAD",
                      "archive_count": len(records), "archives": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
