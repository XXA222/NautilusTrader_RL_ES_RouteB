from __future__ import annotations

import hashlib
from datetime import date

import pytest

from rles_data import binance_archive


@pytest.mark.parametrize(
    ("mode", "field", "value", "message"),
    [
        ("day", "symbol", None, "invalid Binance symbol"),
        ("day", "symbol", 123, "invalid Binance symbol"),
        ("day", "instrument", 123, "instrument must be a nonempty trimmed string"),
        ("day", "instrument", [], "instrument must be a nonempty trimmed string"),
        ("day", "instrument", "", "instrument must be a nonempty trimmed string"),
        ("day", "instrument", "  ", "instrument must be a nonempty trimmed string"),
        ("day", "instrument", " ETHUSDT-PERP.SIM", "instrument must be a nonempty trimmed string"),
        ("day", "instrument", "ETHUSDT-PERP.SIM ", "instrument must be a nonempty trimmed string"),
        ("range", "symbol", None, "invalid Binance symbol"),
        ("range", "symbol", 123, "invalid Binance symbol"),
        ("range", "instrument", 123, "instrument must be a nonempty trimmed string"),
        ("range", "instrument", [], "instrument must be a nonempty trimmed string"),
        ("range", "instrument", "", "instrument must be a nonempty trimmed string"),
        ("range", "instrument", "  ", "instrument must be a nonempty trimmed string"),
        ("range", "instrument", " ETHUSDT-PERP.SIM", "instrument must be a nonempty trimmed string"),
        ("range", "instrument", "ETHUSDT-PERP.SIM ", "instrument must be a nonempty trimmed string"),
    ],
)
def test_invalid_symbol_and_instrument_fail_before_fetch_or_output(
    tmp_path, monkeypatch, mode, field, value, message,
):
    monkeypatch.setattr(
        binance_archive,
        "fetch_archive",
        lambda *_: pytest.fail("invalid archive arguments must fail before fetch"),
    )
    output = tmp_path / "bundle"
    kwargs = {"symbol": "ETHUSDT", "output": output, "delay_ns": 1, field: value}
    if mode == "day":
        kwargs["day"] = "2024-03-01"
        importer = binance_archive.import_day
    else:
        kwargs.update(start_day="2024-03-01", end_day="2024-03-02")
        importer = binance_archive.import_range

    with pytest.raises(binance_archive.DataError, match=message):
        importer(**kwargs)
    assert not output.exists()


@pytest.mark.parametrize(
    "relative_url",
    [None, "", "/daily/klines/ETHUSDT/file.zip", "../daily/file.zip",
     "daily/../private/file.zip", "daily//klines/file.zip", "daily/klines/file.zip?x=1",
     "daily/klines/file%2ezip", "daily\\klines\\file.zip", "other/klines/file.zip"],
)
def test_fetch_archive_rejects_unsafe_relative_url_before_io(
    tmp_path, monkeypatch, relative_url,
):
    monkeypatch.setattr(
        binance_archive,
        "_read_remote",
        lambda *_args, **_kwargs: pytest.fail("unsafe archive path must not access network"),
    )
    parent = tmp_path / "not-created"

    with pytest.raises(binance_archive.DataError, match="safe relative URL"):
        binance_archive.fetch_archive(relative_url, parent / "archive.zip")
    assert not parent.exists()



def test_cached_archive_receipt_normalizes_nonstandard_json_constant(tmp_path):
    relative = "daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-03-01.zip"
    destination = tmp_path / "archive.zip"
    destination.write_bytes(b"synthetic archive")
    receipt = destination.with_suffix(destination.suffix + ".source.json")
    receipt.write_text(
        '{"schema":"rles.binance-archive-source.v1",'
        f'"url":"{binance_archive.BASE + relative}","sha256":NaN}}',
        encoding="utf-8",
    )

    with pytest.raises(binance_archive.DataError, match="source receipt is unreadable"):
        binance_archive.fetch_archive(relative, destination)


def test_funding_api_response_normalizes_nonstandard_json_constant():
    payload = b"NaN"
    with pytest.raises(binance_archive.DataError, match="response is not valid JSON"):
        binance_archive.funding_api_events(
            payload, symbol="ETHUSDT", instrument="ETHUSDT-PERP.SIM",
            source_url="synthetic://funding", expected_sha256=hashlib.sha256(payload).hexdigest(),
            delay_ns=1,
        )


def test_cached_funding_receipt_normalizes_nonstandard_json_constant(tmp_path):
    month = date(2024, 3, 1)
    end_day = date(2024, 3, 2)
    destination = tmp_path / "fundingRate-2024-03-api.json"
    destination.write_bytes(b"[]")
    receipt_path = destination.with_suffix(destination.suffix + ".source.json")
    receipt_path.write_text(
        '{"schema":"rles.binance-funding-api-source.v1","url":NaN,"sha256":"' +
        hashlib.sha256(b"[]").hexdigest() + '"}',
        encoding="utf-8",
    )

    with pytest.raises(binance_archive.DataError, match="source receipt is unreadable"):
        binance_archive.fetch_funding_api("ETHUSDT", month, end_day, tmp_path)

def test_fetch_archive_converts_non_ascii_checksum_to_data_error(tmp_path, monkeypatch):
    relative = "daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-03-01.zip"
    monkeypatch.setattr(
        binance_archive, "_read_remote",
        lambda *_args, **_kwargs: b"\xff",
    )

    with pytest.raises(binance_archive.DataError, match="checksum.*ASCII"):
        binance_archive.fetch_archive(relative, tmp_path / "archive.zip")

def test_archive_events_converts_invalid_zip_to_data_error(tmp_path):
    content = b"synthetic bytes that are not a ZIP archive"
    path = tmp_path / "broken.zip"
    path.write_bytes(content)

    with pytest.raises(binance_archive.DataError, match="invalid or unreadable.*ZIP"):
        binance_archive.archive_events(
            path, stream="klines", instrument="ETHUSDT-PERP.SIM",
            source_url="synthetic://broken.zip",
            expected_sha256=hashlib.sha256(content).hexdigest(), delay_ns=1,
        )

def test_fetch_archive_bounds_preexisting_cache_before_reading(tmp_path, monkeypatch):
    relative = "daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-03-01.zip"
    destination = tmp_path / "archive.zip"
    content = b"oversized"
    destination.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        binance_archive, "_read_remote",
        lambda *_args, **_kwargs: f"{digest} {relative.rsplit('/', 1)[-1]}\n".encode("ascii"),
    )
    monkeypatch.setattr(binance_archive, "_archive_size_limit", lambda _path: 4, raising=False)
    path_type = type(destination)
    read_bytes = path_type.read_bytes

    def guarded_read(path):
        if path == destination:
            pytest.fail("oversized cache must be rejected before reading its bytes")
        return read_bytes(path)

    monkeypatch.setattr(path_type, "read_bytes", guarded_read)

    with pytest.raises(binance_archive.DataError, match="cached archive exceeds bounded"):
        binance_archive.fetch_archive(relative, destination)
    assert not destination.with_suffix(".zip.source.json").exists()
