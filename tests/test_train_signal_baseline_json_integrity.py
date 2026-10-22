from pathlib import Path

import pytest

from qa import prepare_train_signal_baselines_20261016, train_signal_baselines
from rles_data.core import DataError


@pytest.mark.parametrize(("reader", "error_type"), [
    (prepare_train_signal_baselines_20261016._read_json_object, ValueError),
    (train_signal_baselines._read_json_object, DataError),
])
@pytest.mark.parametrize("payload", [
    b'{"bundle":"old","bundle":"new"}',
    b'{"source":\xff}',
    b'["not", "an", "object"]',
])
def test_train_baseline_source_reader_rejects_ambiguous_json(
        tmp_path: Path, reader, error_type: type[Exception], payload: bytes) -> None:
    source = tmp_path / "baseline.json"
    source.write_bytes(payload)
    with pytest.raises(error_type, match="JSON"):
        reader(source)
