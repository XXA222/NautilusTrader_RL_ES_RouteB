from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

from rles_data.core import DataError

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "qa"
sys.path.insert(0, str(QA))
PATH = QA / "reconcile_native_equity_long_20261017.py"
SPEC = spec_from_file_location("routeb_a08_equity_reconciliation_test", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("payload", [
    b'{"split":"train_inner","split":"holdout"}',
    b'{"bundle":\xff}',
    b'["not", "an", "object"]',
])
def test_a08_equity_reconciliation_rejects_ambiguous_json(
        tmp_path: Path, payload: bytes) -> None:
    source = tmp_path / "plan.json"
    source.write_bytes(payload)

    with pytest.raises(DataError, match="JSON"):
        MODULE._load_json_object(source)
