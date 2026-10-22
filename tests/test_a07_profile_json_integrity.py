from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

from rles_data.core import DataError

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "qa"
sys.path.insert(0, str(QA))


def _load(name: str):
    path = QA / name
    spec = spec_from_file_location(f"routeb_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


READERS = (
    _load("profile_native_history_growth_20261016.py")._load_json_object,
    _load("profile_episode_reset_20261017.py")._load_json_object,
)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("payload", [
    b'{"split":"train_inner","split":"holdout"}',
    b'{"market_path":\xff}',
    b'["not", "an", "object"]',
])
def test_a07_profile_readers_reject_ambiguous_json(
        tmp_path: Path, reader, payload: bytes) -> None:
    source = tmp_path / "plan.json"
    source.write_bytes(payload)

    with pytest.raises(DataError, match="JSON"):
        reader(source)
