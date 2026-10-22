from types import SimpleNamespace
from hashlib import sha256

import numpy as np
import pytest

from qa.world_model import run_registered_r4_hierarchy as runner


def test_r4_preregistered_digest_rejects_mismatch_and_duplicate_controls():
    digest = "a" * 64
    preregistration = f"- Protocol JSON SHA-256: `{digest}`\n"
    runner._require_preregistered_digest(preregistration, "Protocol JSON", digest)

    with pytest.raises(ValueError, match="differs from the frozen preregistration"):
        runner._require_preregistered_digest(preregistration, "Protocol JSON", "b" * 64)
    with pytest.raises(ValueError, match="differs from the frozen preregistration"):
        runner._require_preregistered_digest(
            preregistration + preregistration, "Protocol JSON", digest,
        )


def test_r4_preflight_binds_protocol_fold_config_and_runner_bytes(tmp_path):
    protocol = tmp_path / "protocol.json"
    fold_config = tmp_path / "fold.json"
    runner_path = tmp_path / "runner.py"
    protocol.write_bytes(b'{"protocol":1}')
    fold_config.write_bytes(b'{"fold":1}')
    runner_path.write_bytes(b"print('frozen')\n")
    controls = (("Protocol JSON", protocol), ("Fold config", fold_config),
                ("Runner", runner_path))
    preregistration = "\n".join(
        f"- {label} SHA-256: `{sha256(path.read_bytes()).hexdigest()}`"
        for label, path in controls
    )

    runner._verify_preregistered_controls(
        preregistration, protocol_path=protocol,
        fold_config_path=fold_config, runner_path=runner_path,
    )
    protocol.write_bytes(b'{"protocol":2}')
    try:
        runner._verify_preregistered_controls(
            preregistration, protocol_path=protocol,
            fold_config_path=fold_config, runner_path=runner_path,
        )
    except ValueError as exc:
        assert "Protocol JSON digest differs" in str(exc)
    else:
        raise AssertionError("R4 must reject a protocol changed after preregistration")


def test_r4_protocol_rejects_invalid_integer_and_sampling_controls():
    protocol = {
        "training_seeds": [31, 102, 20261018],
        "forecast_seeds": list(range(32)),
        "paths_per_seed": 32,
        "bootstrap_block_length": 12,
        "bootstrap_replicates": 2000,
        "anchor_interval_minutes": 360,
        "horizon": 15,
        "warmup_days": 2,
        "training": {},
        "minute_model": {},
        "hour_model": {},
        "day_model": {},
    }
    runner._validate_protocol_controls(protocol)

    for field, invalid in (("paths_per_seed", True), ("bootstrap_replicates", 0),
                           ("bootstrap_block_length", 61),
                           ("training_seeds", [31, True, 20261018])):
        with pytest.raises(ValueError, match="R4"):
            runner._validate_protocol_controls({**protocol, field: invalid})


def test_registered_r4_control_reader_rejects_duplicate_nested_keys_and_invalid_utf8(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"training":{"seed":1,"seed":2}}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key.*seed"):
        runner._load_json_file(protocol)

    protocol.write_bytes(b'{"training":\xff}')
    with pytest.raises(ValueError, match="invalid R4 JSON input"):
        runner._load_json_file(protocol)


def test_r4_uses_first_required_safe_paths_and_reports_rejections(monkeypatch):
    anchor = 1_700_000_040_000_000_000
    replay = SimpleNamespace(frames=(SimpleNamespace(
        ts_close_ns=anchor, close=100.0, instrument="AVAXUSDT-PERP.BINANCE"),))
    monkeypatch.setattr(
        runner, "_validation_labels",
        lambda *_args: (np.full((1, 16), 100.0), np.zeros((1, 15, 5)), np.zeros(1)),
    )
    candidates = np.zeros((1, 4, 15, 5), dtype=np.float32)
    candidates[0, 0, :, 2] = np.log(4.0)  # Decoder rejects a >2 log-point wick.
    scalers = SimpleNamespace(minute=SimpleNamespace(inverse=lambda values: values))

    result = runner._evaluate_paths(
        candidates, np.asarray([anchor]), replay, object(), scalers, required_paths=2)

    assert result["path_count"].tolist() == [2]
    assert result["candidate_count"].tolist() == [4]
    assert result["rejected_count"].tolist() == [1]
    assert np.isfinite(result["return_crps"]).all()
    assert np.isfinite(result["path_log_rv_crps"]).all()


def test_r4_fails_if_candidate_pool_has_too_few_safe_paths(monkeypatch):
    anchor = 1_700_000_040_000_000_000
    replay = SimpleNamespace(frames=(SimpleNamespace(
        ts_close_ns=anchor, close=100.0, instrument="AVAXUSDT-PERP.BINANCE"),))
    monkeypatch.setattr(
        runner, "_validation_labels",
        lambda *_args: (np.full((1, 16), 100.0), np.zeros((1, 15, 5)), np.zeros(1)),
    )
    candidates = np.zeros((1, 3, 15, 5), dtype=np.float32)
    candidates[0, :, :, 2] = np.log(4.0)
    scalers = SimpleNamespace(minute=SimpleNamespace(inverse=lambda values: values))

    try:
        runner._evaluate_paths(
            candidates, np.asarray([anchor]), replay, object(), scalers, required_paths=2)
    except ValueError as exc:
        assert "insufficient safe forecast paths" in str(exc)
    else:
        raise AssertionError("R4 must fail when fewer than the required safe paths remain")
