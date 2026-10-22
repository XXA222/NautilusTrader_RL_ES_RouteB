from pathlib import Path
import sys

import pytest

QA_ROOT = Path(__file__).resolve().parents[1] / "qa"
sys.path.insert(0, str(QA_ROOT))

from qa import compare_bar_cost_modes_20261017, compare_episode_horizon_20261017  # noqa: E402
from rles_data.core import DataError  # noqa: E402


@pytest.mark.parametrize("reader", [
    compare_bar_cost_modes_20261017._load_json_object,
    compare_episode_horizon_20261017._load_json_object,
])
@pytest.mark.parametrize("payload", [
    b'{"split":"train_inner","split":"holdout"}',
    b'{"source":\xff}',
    b'["not", "an", "object"]',
])
def test_frozen_cost_and_episode_plans_reject_ambiguous_json(
        tmp_path: Path, reader, payload: bytes) -> None:
    source = tmp_path / "plan.json"
    source.write_bytes(payload)
    with pytest.raises(DataError, match="JSON"):
        reader(source)
