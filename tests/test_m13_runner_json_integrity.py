from pathlib import Path

import pytest

from qa.m13_stage_contribution_20261017 import _load_json_object


@pytest.mark.parametrize("payload", [
    b'{"bundle_sha256":"a","bundle_sha256":"b"}',
    b'{"bundle":\xff}',
    b'["not", "an", "object"]',
])
def test_m13_frozen_runner_json_reader_fails_closed(tmp_path: Path, payload: bytes) -> None:
    source = tmp_path / "frozen.json"
    source.write_bytes(payload)

    with pytest.raises(ValueError, match="M1.3 JSON"):
        _load_json_object(source)
