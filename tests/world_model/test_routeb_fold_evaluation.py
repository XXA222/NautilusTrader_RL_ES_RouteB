from __future__ import annotations

import json
import numpy as np
import pytest

from rles_data.core import FoldManifest, MINUTE_NS, Window
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.routeb.evaluation import verify_predictive_report_digest
from rles_world_model.routeb.fold_evaluation import build_fold_predictive_report
from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.integration import prepare_fold_forecast_batch
from rles_world_model.routeb.market import Candle
from rles_world_model.routeb.fold_experiment import (
    run_fold_experiment, verify_fold_experiment_report_digest,
)
from rles_world_model.routeb.fold_protocol import validation_anchor_timestamps


def make_fold_replay():
    dataset_hash = "a" * 64
    fold = FoldManifest(
        "0.4.0", dataset_hash, "b" * 64,
        Window(MINUTE_NS, 20 * MINUTE_NS, 140 * MINUTE_NS),
        Window(140 * MINUTE_NS, 150 * MINUTE_NS, 300 * MINUTE_NS),
        Window(300 * MINUTE_NS, 310 * MINUTE_NS, 400 * MINUTE_NS),
        0, 0, None,
    )
    frames = []
    close = 100.0
    for index in range(1, 300):
        opening = close * np.exp(0.0003 * np.sin(index / 7))
        close = opening * np.exp(0.001 * np.sin(index / 11))
        high = max(opening, close) * 1.001
        low = min(opening, close) * 0.999
        frames.append(MarketFrame(
            index * MINUTE_NS, index * MINUTE_NS, "ETH-PERP",
            float(opening), float(high), float(low), float(close),
            float(10 + index % 13),
        ))
    return fold, MarketReplay(tuple(frames), dataset_hash)


def candidate_paths(replay, anchors, paths=8, horizon=15, bias=0.0):
    by_time = {frame.ts_close_ns: frame for frame in replay.frames}
    result = np.empty((len(anchors), paths, horizon, 5), dtype=np.float64)
    for row, anchor in enumerate(anchors):
        previous = by_time[int(anchor)].close
        for path in range(paths):
            last = previous
            for step in range(horizon):
                gap = (path - (paths - 1) / 2) * 0.00001
                opening = last * np.exp(gap)
                close = opening * np.exp(bias + 0.00001 * np.sin(step + path))
                result[row, path, step] = (
                    opening, max(opening, close) * 1.001,
                    min(opening, close) * 0.999, close, 10.0 + path,
                )
                last = close
    return result


def training_runs(fold, scope="routeb-fold-fresh", snapshot="9" * 40):
    return {
        name: {
            "schema": "routeb.train-run.v1",
            "status": "COMPLETED",
            "candidate": name.lower(),
            "dataset_hash": fold.dataset_hash,
            "fold_hash": fold.hash,
            "train_scope_id": scope,
            "train_npz_sha256": "1" * 64,
            "training_config": {"seed": 31, "steps": 100, "batch_size": 4,
                                "lr": 0.0003, "grad_clip": 100.0,
                                "report_interval": 10},
            "optimizer_updates": 100,
            "train_examples": 24,
            "training_seconds": 4.5,
            "parameter_count": count,
            "device": "cpu:0",
            "configuration_sha256": sha,
            "checkpoint_sha256": artifact,
            "snapshot_sha256": snapshot,
            "upstream_sha256": "e01491fad6434b2245a3b8ca201dd7faedcc458c",
            "resume": False,
        }
        for name, sha, artifact, count in (
            ("E1", "c" * 64, "e" * 64, 1_200),
            ("E2", "d" * 64, "f" * 64, 9_400),
        )
    }


def test_fold_report_generates_e0_and_scores_e1_e2_on_same_future_labels():
    fold, replay = make_fold_replay()
    first_anchor = fold.validation.score_start_ns - MINUTE_NS
    last_anchor = fold.validation.score_end_ns - 15 * MINUTE_NS
    anchors = np.arange(first_anchor, last_anchor + 1, 15 * MINUTE_NS,
                        dtype=np.int64)
    forecasts = {
        "E1": candidate_paths(replay, anchors, bias=0.0),
        "E2": candidate_paths(replay, anchors, bias=0.00002),
    }
    report = build_fold_predictive_report(
        replay, fold, forecasts,
        candidate_training_runs=training_runs(fold),
        candidate_anchor_timestamps_ns={"E1": anchors, "E2": anchors},
        train_scope_id="routeb-fold-fresh",
        snapshot_sha256="9" * 40, forecast_seed=31,
        horizons=(1, 5, 15), anchor_interval_minutes=15,
        context_lookback=32, volatility_window=5, block_length=2,
        bootstrap_replicates=100, seed=7,
    )
    assert set(report["overall"]) == {
        "E0_S0_ZERO_DRIFT_FIXED_VOL", "E0_S1_EWMA_VOL_TRAIN_INNOVATIONS",
        "E0_S2_STATIONARY_BLOCK_BOOTSTRAP", "E1", "E2",
    }
    assert report["authorization_verified"] is False
    assert report["fold_protocol"]["test_payload_decoded"] is False
    assert report["metadata"]["fold_identity"] == fold.hash
    assert report["overall"]["E2"][-1]["horizon"] == 15
    assert report["overall"]["E2"][-1]["samples"] == len(anchors)
    assert verify_predictive_report_digest(report)


def test_prepare_fold_forecast_batch_exports_only_causal_common_anchor_contexts(tmp_path):
    fold, replay = make_fold_replay()
    train_frames = tuple(frame for frame in replay.frames
                         if frame.ts_close_ns < fold.train.score_end_ns)
    train_candles = tuple(Candle(
        frame.ts_close_ns, frame.available_at_ns, frame.instrument, frame.open,
        frame.high, frame.low, frame.close, frame.volume) for frame in train_frames)
    scaler = TrainOnlyStandardizer.fit(
        OHLCVCodec.encode(train_candles[1:], previous_close=train_candles[0].close),
        dataset_id=fold.dataset_hash, scope="train")
    report = prepare_fold_forecast_batch(
        fold_replay=replay, fold=fold, frozen_scaler=scaler,
        train_scope_id="routeb-fold-fresh", train_npz_sha256="1" * 64,
        output_context_npz=tmp_path / "contexts.npz",
        output_request_json=tmp_path / "request.json", horizon=15,
        n_paths=8, seed=31, anchor_interval_minutes=15, context_lookback=32)
    with np.load(tmp_path / "contexts.npz", allow_pickle=False) as contexts:
        assert contexts["histories"].shape == (10, 31, 5)
        assert contexts["anchor_timestamps_ns"][0] == 149 * MINUTE_NS
        assert contexts["anchor_timestamps_ns"][-1] == 284 * MINUTE_NS
        assert contexts["close_timestamps"][:, -1].tolist() == contexts["anchor_timestamps_ns"].tolist()
        assert np.all(contexts["available_timestamps"] <= contexts["anchor_timestamps_ns"][:, None])
    request = json.loads((tmp_path / "request.json").read_text())
    assert request["schema"] == "routeb.forecast-batch-request.v1"
    assert report["anchors"] == 10
    assert report["test_payload_decoded"] is False


@pytest.mark.parametrize("horizon", (1, 5, 15, 16, 60))
def test_common_validation_anchors_keep_final_target_before_exclusive_end(horizon):
    fold, _ = make_fold_replay()
    anchors = validation_anchor_timestamps(
        fold=fold, max_horizon=horizon, anchor_interval_minutes=1)
    assert anchors[0] == fold.validation.score_start_ns - MINUTE_NS
    assert anchors[-1] + horizon * MINUTE_NS < fold.validation.score_end_ns
    assert anchors[-1] + (horizon + 1) * MINUTE_NS == fold.validation.score_end_ns


def test_one_minute_forecast_batch_excludes_anchor_at_last_validation_minute(tmp_path):
    fold, replay = make_fold_replay()
    train_frames = tuple(frame for frame in replay.frames
                         if frame.ts_close_ns < fold.train.score_end_ns)
    train_candles = tuple(Candle(
        frame.ts_close_ns, frame.available_at_ns, frame.instrument, frame.open,
        frame.high, frame.low, frame.close, frame.volume) for frame in train_frames)
    scaler = TrainOnlyStandardizer.fit(
        OHLCVCodec.encode(train_candles[1:], previous_close=train_candles[0].close),
        dataset_id=fold.dataset_hash, scope="train")
    prepare_fold_forecast_batch(
        fold_replay=replay, fold=fold, frozen_scaler=scaler,
        train_scope_id="routeb-fold-fresh", train_npz_sha256="1" * 64,
        output_context_npz=tmp_path / "contexts.npz",
        output_request_json=tmp_path / "request.json", horizon=1,
        n_paths=8, seed=31, anchor_interval_minutes=1, context_lookback=32)
    with np.load(tmp_path / "contexts.npz", allow_pickle=False) as contexts:
        anchors = contexts["anchor_timestamps_ns"]
        assert anchors[-1] == fold.validation.score_end_ns - 2 * MINUTE_NS
        assert anchors[-1] + MINUTE_NS < fold.validation.score_end_ns


def test_run_fold_experiment_uses_one_replay_for_train_and_validation(tmp_path):
    fold, replay = make_fold_replay()

    class FakeWorker:
        def _train(self, sequences, scope, checkpoint, configuration, resume=False):
            checkpoint.write_bytes(b"synthetic checkpoint")
            checkpoint.with_suffix(".json").write_text("{}")
            metadata = json.loads(sequences.with_suffix(".json").read_text())
            config = json.loads(configuration.read_text())
            return {"status": "COMPLETED", "kind": "SYNTHETIC_TRAIN",
                    "reports": [{"parameter_count": 12,
                                 "train_examples": metadata["features_shape"][0],
                                 "optimizer_updates": config["training"]["steps"],
                                 "training_seconds": 1.0, "device": "cpu:0"}]}

        def train_e1(self, **kwargs):
            return self._train(**kwargs)

        def train(self, **kwargs):
            return self._train(**kwargs)

        def forecast_batch(self, *, request, output, candidate, **kwargs):
            payload = json.loads(request.read_text())
            anchors = np.asarray(payload["anchor_timestamps_ns"], dtype=np.int64)
            paths = candidate_paths(replay, anchors, paths=payload["n_paths"],
                                    horizon=payload["horizon"])
            np.savez_compressed(output, bars=paths)
            import hashlib
            return {"status": "COMPLETED",
                    "kind": "E1_BATCH_PRIOR_PATHS" if candidate == "E1" else "F1_BATCH_PRIOR_PATHS",
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "anchors": len(anchors), "paths": payload["n_paths"],
                    "horizon": payload["horizon"]}

        def next_distribution_batch(self, *, request, output, candidate, **kwargs):
            payload = json.loads(request.read_text())
            assert payload["horizon"] == 1
            anchors = np.asarray(payload["anchor_timestamps_ns"], dtype=np.int64)
            np.savez_compressed(
                output,
                standardized_mean=np.zeros((len(anchors), 5), dtype=np.float32),
                standardized_scale=np.ones((len(anchors), 5), dtype=np.float32),
            )
            import hashlib
            return {"status": "COMPLETED",
                    "kind": f"{candidate}_BATCH_NEXT_DISTRIBUTION",
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "anchors": len(anchors), "paths": 1, "horizon": 1}

    training = {"seed": 31, "steps": 2, "batch_size": 1, "lr": 0.001,
                "grad_clip": 10.0, "report_interval": 1}
    report = run_fold_experiment(
        replay=replay, fold=fold, train_scope_id="routeb-fold-single-read",
        candidate_configurations={
            "E1": {"model": {"hidden": 8}, "training": training},
            "E2": {"model": {"deter": 16}, "training": training},
        }, worker=FakeWorker(), output_dir=tmp_path,
        repository_commit="9" * 40, source_code_sha256="8" * 64,
        marker_sha256="7" * 64, reservation_label="routeb-fold-single-read",
        forecast_seed=31, forecast_paths=8, horizons=(1, 5, 15),
        anchor_interval_minutes=15, context_lookback=32, volatility_window=5,
        block_length=2, bootstrap_replicates=100, seed=7)
    assert report["status"] == "COMPLETED"
    assert report["test_payload_decoded"] is False
    assert report["predictive_comparison"]["authorization_verified"] is False
    assert report["predictive_comparison"]["metadata"]["fold_identity"] == fold.hash
    assert set(report["forecast_artifacts"]) == {"E1", "E2"}
    assert len(report["forecast_artifacts"]["E1"]["artifact_sha256"]) == 64
    assert report["forecast_context"]["request_sha256"]
    assert np.isfinite(report["predictive_comparison"]["overall"]["E1"][0]["nll"])
    assert np.isfinite(report["predictive_comparison"]["overall"]["E2"][0]["nll"])
    assert verify_fold_experiment_report_digest(report)
    tampered = {**report, "marker_sha256": "0" * 64}
    assert not verify_fold_experiment_report_digest(tampered)
    assert (tmp_path / "fold_experiment_report.json").is_file()
    assert (tmp_path / "e1_checkpoint.npz").is_file()
    assert (tmp_path / "e2_validation_forecasts.npz").is_file()


def test_fold_report_rejects_test_rows_and_unmatched_candidate_anchor_count():
    fold, replay = make_fold_replay()
    anchors = np.arange(fold.validation.score_start_ns - MINUTE_NS,
                        fold.validation.score_end_ns - 15 * MINUTE_NS + 1,
                        15 * MINUTE_NS, dtype=np.int64)
    paths = candidate_paths(replay, anchors)
    candidate = {"E1": paths, "E2": paths[:-1]}
    with pytest.raises(ValueError, match="common anchor"):
        build_fold_predictive_report(
            replay, fold, candidate,
            candidate_training_runs=training_runs(fold),
            candidate_anchor_timestamps_ns={"E1": anchors, "E2": anchors},
            train_scope_id="routeb-fold-fresh",
            snapshot_sha256="9" * 40, forecast_seed=31,
            horizons=(1, 5, 15), anchor_interval_minutes=15,
            context_lookback=32, volatility_window=5,
            bootstrap_replicates=50,
        )

    test_replay = MarketReplay(replay.frames + (
        MarketFrame(300 * MINUTE_NS, 300 * MINUTE_NS, "ETH-PERP",
                    100, 101, 99, 100, 1),
    ), replay.dataset_hash)
    with pytest.raises(ValueError, match="exclude Test"):
        build_fold_predictive_report(
            test_replay, fold, {"E1": paths, "E2": paths},
            candidate_training_runs=training_runs(fold),
            candidate_anchor_timestamps_ns={"E1": anchors, "E2": anchors},
            train_scope_id="routeb-fold-fresh", snapshot_sha256="9" * 40,
            forecast_seed=31, horizons=(1, 5, 15), anchor_interval_minutes=15,
            context_lookback=32, volatility_window=5, block_length=2,
            bootstrap_replicates=50,
        )


def test_fold_report_requires_matched_e1_e2_training_examples():
    fold, replay = make_fold_replay()
    anchors = np.arange(fold.validation.score_start_ns - MINUTE_NS,
                        fold.validation.score_end_ns - 15 * MINUTE_NS + 1,
                        15 * MINUTE_NS, dtype=np.int64)
    paths = candidate_paths(replay, anchors)
    mismatched = training_runs(fold)
    mismatched["E2"]["train_examples"] += 1
    with pytest.raises(ValueError, match="share one Train artifact"):
        build_fold_predictive_report(
            replay, fold, {"E1": paths, "E2": paths},
            candidate_training_runs=mismatched,
            candidate_anchor_timestamps_ns={"E1": anchors, "E2": anchors},
            train_scope_id="routeb-fold-fresh", snapshot_sha256="9" * 40,
            forecast_seed=31, horizons=(1, 5, 15), anchor_interval_minutes=15,
            context_lookback=32, volatility_window=5, block_length=2,
            bootstrap_replicates=50,
        )

    mismatched = training_runs(fold)
    mismatched["E2"]["train_npz_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="share one Train artifact"):
        build_fold_predictive_report(
            replay, fold, {"E1": paths, "E2": paths},
            candidate_training_runs=mismatched,
            candidate_anchor_timestamps_ns={"E1": anchors, "E2": anchors},
            train_scope_id="routeb-fold-fresh", snapshot_sha256="9" * 40,
            forecast_seed=31, horizons=(1, 5, 15), anchor_interval_minutes=15,
            context_lookback=32, volatility_window=5, block_length=2,
            bootstrap_replicates=50,
        )

    shifted = anchors.copy()
    shifted[0] += MINUTE_NS
    with pytest.raises(ValueError, match="common ordered Validation anchors"):
        build_fold_predictive_report(
            replay, fold, {"E1": paths, "E2": paths},
            candidate_training_runs=training_runs(fold),
            candidate_anchor_timestamps_ns={"E1": anchors, "E2": shifted},
            train_scope_id="routeb-fold-fresh", snapshot_sha256="9" * 40,
            forecast_seed=31, horizons=(1, 5, 15), anchor_interval_minutes=15,
            context_lookback=32, volatility_window=5, block_length=2,
            bootstrap_replicates=50,
        )
