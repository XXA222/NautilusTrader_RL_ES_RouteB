from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

from rles_data.core import DataError

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "qa"
sys.path.insert(0, str(QA))


def _module(filename: str):
    path = QA / filename
    spec = spec_from_file_location(f"routeb_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


READERS = (
    _module("low_capacity_predictor_20261016.py")._load_json_object,
    _module("zero_target_exit_probe_20261016.py")._load_json_object,
    _module("evaluate_discount_half_life_probe_20261017.py")._load_json_object,
    _module("run_discount_half_life_probe_20261016.py")._load_json_object,
    _module("prepare_low_capacity_predictor_20261016.py")._read_json_object,
    _module("prepare_zero_target_exit_probe_20261016.py")._read_json_object,
)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("payload", [
    b'{"split":"train_inner","split":"test"}',
    b'{"bundle":\xff}',
    b'["not", "an", "object"]',
])
def test_legacy_train_diagnostic_readers_reject_ambiguous_json(
        tmp_path: Path, reader, payload: bytes) -> None:
    source = tmp_path / "plan.json"
    source.write_bytes(payload)

    with pytest.raises(DataError, match="JSON"):
        reader(source)
