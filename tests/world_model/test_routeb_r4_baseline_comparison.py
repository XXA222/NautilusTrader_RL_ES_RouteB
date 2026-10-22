from __future__ import annotations

import runpy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rles_data.core import MINUTE_NS
from rles_orchestrator import research_access
from rles_research.access import ProjectResearchRegistry
from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.market import Candle
from rles_world_model.routeb.jax_model import RSSMConfig


RUNNER = Path(__file__).resolve().parents[2] / "qa/world_model/run_registered_r4_e2_e3_comparison.py"


def test_r4_e2_e3_preregistration_binds_protocol_fold_and_runner_bytes(tmp_path):
    runner = runpy.run_path(str(RUNNER))
    protocol = tmp_path / "protocol.json"
    fold_config = tmp_path / "fold.json"
    source_runner = tmp_path / "runner.py"
    protocol.write_bytes(b'{"protocol":1}')
    fold_config.write_bytes(b'{"fold":1}')
    source_runner.write_bytes(b"print('frozen')\n")
    preregistration = "\n".join((
        f"- Protocol SHA-256: `{sha256(protocol.read_bytes()).hexdigest()}`",
        f"- Registered Fold config SHA-256: `{sha256(fold_config.read_bytes()).hexdigest()}`",
        f"- Registered runner SHA-256: `{sha256(source_runner.read_bytes()).hexdigest()}`",
    ))
    runner["_verify_preregistered_controls"](
        preregistration, protocol_path=protocol,
        fold_config_path=fold_config, runner_path=source_runner)

    source_runner.write_bytes(b"print('changed')\n")
    with pytest.raises(ValueError, match="Runner digest differs"):
        runner["_verify_preregistered_controls"](
            preregistration, protocol_path=protocol,
            fold_config_path=fold_config, runner_path=source_runner)


def test_r4_e2_e3_preregistration_requires_one_runner_digest():
    runner = runpy.run_path(str(RUNNER))
    digest = "a" * 64
    preregistration = (f"- Protocol JSON SHA-256: `{digest}`\n"
                       f"- Fold config SHA-256: `{digest}`\n")
    with pytest.raises(ValueError, match="digest differs from or is missing"):
        runner["_require_preregistered_digest"](
            preregistration, "Runner", ("Registered runner", "Runner"), digest)


def test_partial_r4_attempt_registration_is_terminalized(tmp_path, monkeypatch):
    runner = runpy.run_path(str(RUNNER))
    registrations = []
    registry = ProjectResearchRegistry(tmp_path / "r4.sqlite")
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    real_register = research_access.register_routeb_r4_baseline_comparison_attempt

    def register(*, candidate_id, **kwargs):
        if candidate_id == "R4_E3_HIERARCHY":
            raise RuntimeError("ledger write failed")
        registrations.append(candidate_id)
        return real_register(candidate_id=candidate_id, **kwargs)

    namespace = runner["_register_comparison_attempts"].__globals__
    namespace["register_routeb_r4_baseline_comparison_attempt"] = register
    namespace["record_routeb_candidate_event"] = research_access.record_routeb_candidate_event
    namespace["ATTEMPTS"].clear()
    namespace["ATTEMPT_STATES"].clear()
    configurations = {
        (candidate, 31): (None, {"candidate": candidate})
        for candidate in runner["CANDIDATES"]
    }
    fold = SimpleNamespace(hash="b" * 64, feature_hash="c" * 64)
    with pytest.raises(RuntimeError, match="ledger write failed"):
        runner["_register_comparison_attempts"](
            configurations, (31,), "routeb-fold-test", fold)
    assert registrations == ["R4_E2_SINGLE_SCALE"]
    summary = registry.trial_attempt_summary("routeb-r4-baseline:routeb-fold-test")
    assert summary["protocol_registered_attempt_count"] == 1
    assert [event["status"] for event in summary["attempts"][0]["events"]] == [
        "registered", "failed",
    ]


def test_r4_e3_risk_objective_is_frozen_to_e2_model_configuration():
    runner = runpy.run_path(str(RUNNER))
    e2_model = RSSMConfig(
        deter=32, hidden=16, stoch=2, classes=4, blocks=2,
        min_scale=0.007, risk_prediction_scale=0.4,
        risk_horizons=(1, 7, 21), risk_horizon_weights=(1.0, 2.0, 3.0),
        path_log_rv_crps_horizon=7)
    e3_protocol = {"variant": "e3b_fusion", "hour_context_steps": 8,
                   "day_context_steps": 4}
    protocol = {"e3_training_steps": 20, "e3_window_length": 256,
                "burn_in": 16, "learning_rate": 1e-3, "grad_clip": 10.0,
                "report_interval": 5}
    config = runner["_matched_e3_training_config"](
        e2_model=e2_model, e3_protocol=e3_protocol, protocol=protocol, seed=17)
    assert config.risk_auxiliary_scale == e2_model.risk_prediction_scale
    assert config.risk_min_scale == e2_model.min_scale
    assert config.risk_horizons == e2_model.risk_horizons
    assert config.risk_horizon_weights == e2_model.risk_horizon_weights
    assert config.risk_barrier_quantile == runner["RISK_BARRIER_VOLATILITY_QUANTILE"]


def test_e2_and_e3_path_metrics_share_anchor_label_conventions():
    runner = runpy.run_path(str(RUNNER))
    factory = runpy.run_path(str(Path(__file__).with_name("test_routeb_fold_evaluation.py")))
    fold, replay = factory["make_fold_replay"]()
    anchor = 150 * MINUTE_NS
    frames = {frame.ts_close_ns: frame for frame in replay.frames}
    anchor_frame = frames[anchor]
    future = tuple(frames[anchor + minute * MINUTE_NS] for minute in range(1, 16))
    bars = np.asarray([[(frame.open, frame.high, frame.low, frame.close, frame.volume)
                        for frame in future]] * 2, dtype=np.float64)[None, ...]
    e2 = runner["_metrics_from_real_bars"](
        bars, np.asarray([anchor], dtype=np.int64), replay, fold, required=2)
    train_frames = tuple(frame for frame in replay.frames
                         if frame.ts_close_ns < fold.train.score_end_ns)
    train_candles = tuple(Candle(
        frame.ts_close_ns, frame.available_at_ns, frame.instrument, frame.open,
        frame.high, frame.low, frame.close, frame.volume) for frame in train_frames)
    scaler = TrainOnlyStandardizer.fit(
        OHLCVCodec.encode(train_candles[1:], previous_close=train_candles[0].close),
        dataset_id=fold.dataset_hash, scope="train")
    encoded = OHLCVCodec.encode(
        tuple(Candle(frame.ts_close_ns, frame.available_at_ns, frame.instrument,
                     frame.open, frame.high, frame.low, frame.close, frame.volume)
              for frame in future), previous_close=anchor_frame.close)
    standardized = scaler.transform(encoded)
    e3 = runner["_metrics_from_path_samples"](
        np.stack((standardized, standardized))[None, ...],
        np.asarray([anchor], dtype=np.int64), replay, fold,
        type("Scalers", (), {"minute": scaler})(), required=2)
    for result in (e2, e3):
        assert result["return_crps"].shape == (1,)
        assert result["path_log_rv_crps"].shape == (1,)
        assert np.isfinite(result["return_crps"]).all()
        assert np.isfinite(result["path_log_rv_crps"]).all()
