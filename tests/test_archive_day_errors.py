import pytest

from rles_data import binance_archive


@pytest.mark.parametrize("day", ["2024-02-30", "2024-1-01", None, 20240101])
def test_import_day_normalizes_invalid_date_inputs_before_fetch(tmp_path, monkeypatch, day):
    monkeypatch.setattr(
        binance_archive,
        "fetch_archive",
        lambda *_: pytest.fail("invalid date must be rejected before any fetch"),
    )

    with pytest.raises(binance_archive.DataError, match="day must be YYYY-MM-DD"):
        binance_archive.import_day(
            symbol="ETHUSDT", day=day, output=tmp_path / "bundle", delay_ns=1
        )


@pytest.mark.parametrize(
    ("start_day", "end_day", "message"),
    [
        ("2024-02-30", "2024-03-01", "start_day must be YYYY-MM-DD"),
        ("2024-03-01", "2024-13-01", "end_day must be YYYY-MM-DD"),
        (None, "2024-03-01", "start_day must be YYYY-MM-DD"),
        ("2024-03-01", 20240302, "end_day must be YYYY-MM-DD"),
    ],
)
def test_import_range_normalizes_invalid_date_inputs_before_fetch(
    tmp_path, monkeypatch, start_day, end_day, message,
):
    monkeypatch.setattr(
        binance_archive,
        "fetch_archive",
        lambda *_: pytest.fail("invalid date must be rejected before any fetch"),
    )

    with pytest.raises(binance_archive.DataError, match=message):
        binance_archive.import_range(
            symbol="ETHUSDT", start_day=start_day, end_day=end_day,
            output=tmp_path / "bundle", delay_ns=1,
        )


@pytest.mark.parametrize("delay_ns", [0, -1, None, True])
def test_import_day_rejects_invalid_delay_before_fetch(tmp_path, monkeypatch, delay_ns):
    monkeypatch.setattr(
        binance_archive,
        "fetch_archive",
        lambda *_: pytest.fail("invalid delay must be rejected before any fetch"),
    )

    with pytest.raises(binance_archive.DataError, match="explicit positive availability delay"):
        binance_archive.import_day(
            symbol="ETHUSDT", day="2024-03-01", output=tmp_path / "bundle",
            delay_ns=delay_ns,
        )


@pytest.mark.parametrize("delay_ns", [0, -1, None, True])
def test_import_range_rejects_invalid_delay_before_fetch(tmp_path, monkeypatch, delay_ns):
    monkeypatch.setattr(
        binance_archive,
        "fetch_archive",
        lambda *_: pytest.fail("invalid delay must be rejected before any fetch"),
    )

    with pytest.raises(binance_archive.DataError, match="explicit positive availability delay"):
        binance_archive.import_range(
            symbol="ETHUSDT", start_day="2024-03-01", end_day="2024-03-02",
            output=tmp_path / "bundle", delay_ns=delay_ns,
        )
