from pathlib import Path
import sys

import pytest

QA_ROOT = Path(__file__).resolve().parents[1] / "qa"
sys.path.insert(0, str(QA_ROOT))

from qa import preregister_m33_action_value, preregister_stage59_train_diagnostics  # noqa: E402
from qa import preregister_stage60_fixed_regimes, profile_stage69_episode_reset_20261017  # noqa: E402
from rles_data.core import DataError  # noqa: E402


LOADERS = (
    preregister_m33_action_value.load,
    preregister_stage59_train_diagnostics.load,
    preregister_stage60_fixed_regimes.load,
)
BAD_JSON = (
    b'{"window":{"start_ns":1,"start_ns":2}}',
    b'{"scope":\xff}',
    b'["not", "an", "object"]',
)


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("payload", BAD_JSON)
def test_stage_preregistration_readers_reject_ambiguous_json(
        tmp_path: Path, loader, payload: bytes) -> None:
    path = tmp_path / "prereg.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="JSON object"):
        loader(path)


@pytest.mark.parametrize("payload", BAD_JSON)
def test_stage69_train_runner_rejects_ambiguous_scope_json(payload: bytes) -> None:
    with pytest.raises(DataError, match="JSON input root|unambiguous JSON"):
        profile_stage69_episode_reset_20261017._parse_json_object(
            payload, "synthetic test input")
