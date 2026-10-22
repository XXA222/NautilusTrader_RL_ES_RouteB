from pathlib import Path

import pytest

from qa.world_model import run_registered_fold_ensemble as ensemble
from qa.world_model import run_registered_r3_ablation as r3
from rles_world_model.contracts import DataError


@pytest.mark.parametrize("reader", [r3._read_e2, lambda path: ensemble._read_forecasts(path, "E2")])
@pytest.mark.parametrize(
    "filename,payload",
    [
        ("validation_forecast_request.json", b'{"anchor_timestamps_ns":[],"anchor_timestamps_ns":[]}'),
        ("fold_experiment_report.json", b'{"identity":{"fold":"a","fold":"b"}}'),
    ],
)
def test_registered_report_readers_reject_duplicate_json_before_npz(
        tmp_path: Path, reader, filename: str, payload: bytes) -> None:
    (tmp_path / filename).write_bytes(payload)

    with pytest.raises(DataError, match="valid JSON input"):
        reader(tmp_path)
