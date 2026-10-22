from datetime import date

import pytest

from qa.world_model.download_routeb_r4_sol_archives import _download, _months


def test_sol_archive_month_range_covers_frozen_source_interval():
    months = _months(date(2023, 7, 25), date(2024, 8, 27))
    assert len(months) == 14
    assert months[0] == "2023-07"
    assert months[-1] == "2024-08"


def test_sol_archive_downloader_rejects_non_official_host_before_network(tmp_path):
    target = tmp_path / "archive.zip"
    with pytest.raises(ValueError, match="official Binance HTTPS"):
        _download("https://example.com/archive.zip", target)
    assert not target.exists()


def test_sol_archive_downloader_refuses_overwrite_before_network(tmp_path):
    target = tmp_path / "archive.zip"
    target.write_bytes(b"frozen")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _download("https://data.binance.vision/archive.zip", target)
    assert target.read_bytes() == b"frozen"
