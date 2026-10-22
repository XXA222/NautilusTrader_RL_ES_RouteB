from __future__ import annotations

import pytest

from rles_data import DataError
from rles_data.io import _event_from_row, curated_source_identity, read_curated_window


@pytest.mark.parametrize(
    "marker",
    [
        b'{"schema_version":NaN}',
        b'{"schema_version":"0.4.1","schema_version":"0.4.1"}',
        b"\xff",
    ],
)
def test_curated_source_identity_normalizes_strict_json_failures(tmp_path, marker):
    target = tmp_path / "market.parquet"
    target.write_bytes(b"placeholder")
    target.with_name(target.name + ".COMPLETED.json").write_bytes(marker)

    with pytest.raises(DataError, match="invalid curated commit metadata"):
        curated_source_identity(target)


def test_curated_event_row_normalizes_nonstandard_json_constant():
    with pytest.raises(DataError, match="invalid curated event payload JSON"):
        _event_from_row({"payload_json": '{"nested":{"value":Infinity}}'})


def test_curated_window_reader_normalizes_nonstandard_marker_constant(tmp_path):
    target = tmp_path / "market.parquet"
    target.write_bytes(b"placeholder")
    target.with_name(target.name + ".COMPLETED.json").write_bytes(b'{"schema_version":NaN}')

    with pytest.raises(DataError, match="invalid curated commit metadata"):
        read_curated_window(target, 0, 10, expected_marker_sha256="0" * 64)


def test_curated_source_identity_normalizes_invalid_ascii_marker_receipt(tmp_path):
    target = tmp_path / "market.parquet"
    target.write_bytes(b"placeholder")
    marker = target.with_name(target.name + ".COMPLETED.json")
    marker.write_text(
        '{"schema_version":"0.4.1","snapshot_hash":"' + "0" * 64 +
        '","file_sha256":"' + "0" * 64 + '","events":1,"row_groups":[]}',
        encoding="utf-8",
    )
    marker.with_suffix(marker.suffix + ".sha256").write_bytes(b"\xff")

    with pytest.raises(DataError, match="curated scoped manifest receipt is unreadable"):
        curated_source_identity(target)
