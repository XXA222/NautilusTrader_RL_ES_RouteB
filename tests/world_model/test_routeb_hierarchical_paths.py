from __future__ import annotations

import numpy as np
import pytest

from rles_world_model.routeb.codec import TrainOnlyStandardizer
from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
from rles_world_model.routeb.hierarchical_paths import (
    PathSamplingError,
    align_bucket_feedback,
    generate_recursive_hierarchical_ohlcv_paths,
    sample_hierarchical_ohlcv_paths,
)
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchicalModelConfig,
    HierarchicalTrainingConfig,
    generate_recursive_hierarchical_f2_paths,
)
from rles_world_model.routeb.jax_model import RSSMConfig
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS


def _bars(count: int) -> tuple[Candle, ...]:
    return tuple(Candle(
        ts_close_ns=index * MINUTE_NS,
        available_at_ns=index * MINUTE_NS,
        instrument="SYNTHETIC.PATHS",
        open=100.0, high=101.0, low=99.0, close=100.0, volume=2.0,
    ) for index in range(1, count + 1))


def _scaler() -> TrainOnlyStandardizer:
    return TrainOnlyStandardizer(np.zeros(5), np.ones(5), "synthetic-paths", "train")


def _scalers() -> tuple[TrainOnlyStandardizer, TrainOnlyStandardizer, TrainOnlyStandardizer]:
    return _scaler(), _scaler(), _scaler()


def test_sampled_paths_are_legal_and_future_slow_bars_conserve_each_path():
    observed = _bars(30)
    minute_scaler, hour_scaler, day_scaler = _scalers()
    result = sample_hierarchical_ohlcv_paths(
        minute_mean=np.zeros((90, 5)), minute_scale=np.full((90, 5), 0.01),
        minute_scaler=minute_scaler, hour_scaler=hour_scaler,
        day_scaler=day_scaler, observed_prefix=observed, paths=3, seed=77,
    )

    assert result.minute.shape == (3, 90, 5)
    assert result.minute_features.shape == (3, 90, 5)
    assert result.hour.shape == (3, 2, 5)
    assert result.day.shape == (3, 0, 5)
    assert result.hour_feedback.shape == (3, 2, 5)
    assert result.day_feedback.shape == (3, 0, 5)
    assert result.hour_close_ns.tolist() == [60 * MINUTE_NS, 120 * MINUTE_NS]
    assert result.rejected_paths == 0
    assert result.rejection_rate == 0.0
    assert not result.minute.flags.writeable
    for path_index, path in enumerate(result.minute):
        first_bucket = np.concatenate((
            np.asarray([[bar.open, bar.high, bar.low, bar.close, bar.volume]
                        for bar in observed], dtype=np.float64),
            path[:30],
        ))
        expected = np.asarray((
            first_bucket[0, 0], first_bucket[:, 1].max(), first_bucket[:, 2].min(),
            first_bucket[-1, 3], first_bucket[:, 4].sum(),
        ))
        np.testing.assert_array_equal(result.hour[path_index, 0], expected)


def test_path_sampler_rejects_unsafe_draws_without_clipping():
    with pytest.raises(PathSamplingError) as caught:
        sample_hierarchical_ohlcv_paths(
            minute_mean=np.tile([30.0, 0.0, 0.0, 0.0, 0.0], (2, 1)),
            minute_scale=np.full((2, 5), 1e-6), minute_scaler=_scaler(),
            hour_scaler=_scaler(), day_scaler=_scaler(),
            observed_prefix=_bars(60), paths=2, seed=4,
            max_attempts_per_path=2,
        )

    assert caught.value.requested == 2
    assert caught.value.attempted == 4
    assert caught.value.rejected == 4


def test_day_bucket_is_derived_from_observed_prefix_and_synthetic_minutes():
    minute_scaler, hour_scaler, day_scaler = _scalers()
    result = sample_hierarchical_ohlcv_paths(
        minute_mean=np.zeros((20, 5)), minute_scale=np.full((20, 5), 0.005),
        minute_scaler=minute_scaler, hour_scaler=hour_scaler,
        day_scaler=day_scaler, observed_prefix=_bars(1_430), paths=2, seed=11,
    )

    assert result.hour.shape == (2, 1, 5)
    assert result.day.shape == (2, 1, 5)
    assert result.hour_feedback.shape == (2, 1, 5)
    assert result.day_feedback.shape == (2, 1, 5)
    assert result.day_close_ns.tolist() == [1_440 * MINUTE_NS]
    for path_index, path in enumerate(result.minute):
        source = np.concatenate((
            np.asarray([[bar.open, bar.high, bar.low, bar.close, bar.volume]
                        for bar in _bars(1_430)], dtype=np.float64),
            path,
        ))
        bucket_rows = source[:1_440]
        expected = np.asarray((
            bucket_rows[0, 0], bucket_rows[:, 1].max(), bucket_rows[:, 2].min(),
            bucket_rows[-1, 3], bucket_rows[:, 4].sum(),
        ))
        np.testing.assert_array_equal(result.day[path_index, 0], expected)


def test_path_feedback_aligns_to_first_decision_after_availability_delay():
    feedback = np.arange(2 * 2 * 5, dtype=np.float32).reshape((2, 2, 5))
    decisions = np.asarray([60, 61, 120, 121], dtype=np.int64) * MINUTE_NS
    aligned = align_bucket_feedback(
        close_ns=np.asarray([60, 120], dtype=np.int64) * MINUTE_NS,
        feedback=feedback, minute_decisions_ns=decisions,
        availability_delay_ns=MINUTE_NS,
        expected_events=(False, True, False, True),
    )

    assert aligned.shape == (2, 4, 5)
    np.testing.assert_array_equal(aligned[:, 1], feedback[:, 0])
    np.testing.assert_array_equal(aligned[:, 3], feedback[:, 1])
    np.testing.assert_array_equal(aligned[:, (0, 2)], 0)


def test_path_feedback_rejects_event_plan_without_matching_generated_bucket():
    with pytest.raises(ValueError, match="does not match"):
        align_bucket_feedback(
            close_ns=np.asarray([60], dtype=np.int64) * MINUTE_NS,
            feedback=np.ones((1, 1, 5), dtype=np.float32),
            minute_decisions_ns=np.asarray([60, 61], dtype=np.int64) * MINUTE_NS,
            availability_delay_ns=MINUTE_NS,
            expected_events=(True, False),
        )


def test_recursive_path_generation_conditions_each_step_on_its_sampled_past():
    observed = _bars(30)
    decisions = np.arange(31, 35, dtype=np.int64) * MINUTE_NS
    seen_feedback = []
    predicted_means = []

    def predictor(prefix, minute_feedback, hour_feedback, day_feedback):
        seen_feedback.append((minute_feedback.copy(), hour_feedback.copy(),
                              day_feedback.copy()))
        mean = np.zeros((minute_feedback.shape[0], 5), dtype=np.float64)
        if len(prefix) > 1:
            mean = minute_feedback[:, -2].astype(np.float64)
        predicted_means.append(mean.copy())
        return mean, np.full_like(mean, 0.001)

    result = generate_recursive_hierarchical_ohlcv_paths(
        predictor=predictor, future_minute_decisions_ns=decisions,
        future_hour_events=(False,) * len(decisions),
        future_day_events=(False,) * len(decisions),
        hour_availability_delay_ns=0, day_availability_delay_ns=0,
        minute_scaler=_scaler(), hour_scaler=_scaler(), day_scaler=_scaler(),
        observed_prefix=observed, paths=2, seed=39)

    assert len(seen_feedback) == len(decisions)
    assert result.ensemble.minute.shape == (2, 4, 5)
    assert result.ensemble.hour.shape == (2, 0, 5)
    assert result.ensemble.day.shape == (2, 0, 5)
    assert result.attempted_bar_samples == 8
    assert result.rejected_bar_samples == 0
    for step in range(1, len(decisions)):
        np.testing.assert_array_equal(
            seen_feedback[step][0][:, step - 1],
            result.ensemble.minute_features[:, step - 1],
        )
        np.testing.assert_array_equal(
            predicted_means[step], result.ensemble.minute_features[:, step - 1])
    assert not np.any(seen_feedback[0][0])


def test_recursive_path_generation_delivers_hour_aggregate_after_its_event():
    observed = _bars(59)
    decisions = np.arange(60, 63, dtype=np.int64) * MINUTE_NS
    seen_hour_feedback = []

    def predictor(prefix, minute_feedback, hour_feedback, day_feedback):
        seen_hour_feedback.append(hour_feedback.copy())
        mean = np.zeros((hour_feedback.shape[0], 5), dtype=np.float64)
        return mean, np.full_like(mean, 0.001)

    result = generate_recursive_hierarchical_ohlcv_paths(
        predictor=predictor, future_minute_decisions_ns=decisions,
        future_hour_events=(True, False, False),
        future_day_events=(False, False, False),
        hour_availability_delay_ns=0, day_availability_delay_ns=0,
        minute_scaler=_scaler(), hour_scaler=_scaler(), day_scaler=_scaler(),
        observed_prefix=observed, paths=2, seed=53)

    assert result.ensemble.hour_close_ns.tolist() == [60 * MINUTE_NS]
    np.testing.assert_array_equal(seen_hour_feedback[0], 0)
    np.testing.assert_array_equal(
        seen_hour_feedback[1][:, 0], result.ensemble.hour_feedback[:, 0])
    assert np.any(seen_hour_feedback[1][:, 0] != 0)


def test_recursive_path_generation_rejects_slow_event_without_closed_bucket():
    def predictor(prefix, minute_feedback, hour_feedback, day_feedback):
        mean = np.zeros((hour_feedback.shape[0], 5), dtype=np.float64)
        return mean, np.full_like(mean, 0.001)

    with pytest.raises(ValueError, match="no matching path-derived aggregate"):
        generate_recursive_hierarchical_ohlcv_paths(
            predictor=predictor,
            future_minute_decisions_ns=np.asarray([31 * MINUTE_NS], dtype=np.int64),
            future_hour_events=(True,), future_day_events=(False,),
            hour_availability_delay_ns=0, day_availability_delay_ns=0,
            minute_scaler=_scaler(), hour_scaler=_scaler(), day_scaler=_scaler(),
            observed_prefix=_bars(30), paths=1, seed=17)


def test_recursive_f2_wrapper_replays_each_generated_prefix(monkeypatch):
    replay = MarketSequence(_bars(4_320), "synthetic-recursive-f2", "train")
    tensors, scalers = build_train_hierarchy(
        replay, burn_in=2, score_start_ns=replay.candles[1].ts_close_ns,
        score_end_ns=replay.candles[-1].ts_close_ns + MINUTE_NS)
    model = RSSMConfig(deter=8, hidden=8, stoch=1, classes=2, blocks=1)
    model_config = HierarchicalModelConfig(model, model, model)
    training_config = HierarchicalTrainingConfig(
        seed=47, steps=1, batch_size=1, window_length=4,
        burn_in=1, stride=1, report_interval=1)
    calls = []

    def fake_forecast(**kwargs):
        calls.append(kwargs)
        feedback = kwargs["minute_feedback_features"]
        batch, horizon, _ = feedback.shape
        mean = np.zeros((batch, horizon, 5), dtype=np.float64)
        if horizon > 1:
            mean[:, -1] = feedback[:, -2]
        return {"minute_mean": mean,
                "minute_scale": np.full_like(mean, 0.001)}

    monkeypatch.setattr(
        "rles_world_model.routeb.jax_hierarchy_train.forecast_hierarchical_f2",
        fake_forecast)
    decisions = np.arange(4_321, 4_324, dtype=np.int64) * MINUTE_NS
    result = generate_recursive_hierarchical_f2_paths(
        tensors=tensors, train_scope_id="synthetic-train-only",
        dataset_id=replay.dataset_id, scaler_sha256="a" * 64,
        checkout="unused-test-checkout", checkpoint="unused-test-checkpoint",
        model_config=model_config, training_config=training_config,
        training_tensor_sha256="b" * 64, inference_context_sha256="c" * 64,
        future_minute_decisions_ns=decisions,
        hour_availability_delay_ns=0, day_availability_delay_ns=0,
        minute_scaler=scalers.minute, hour_scaler=scalers.hour,
        day_scaler=scalers.day, observed_prefix=replay.candles,
        paths=2, forecast_seed=73)

    assert len(calls) == len(decisions)
    assert result.ensemble.minute.shape == (2, len(decisions), 5)
    assert result.ensemble.hour.shape == (2, 0, 5)
    np.testing.assert_array_equal(
        calls[1]["minute_feedback_features"][:, 0],
        result.ensemble.minute_features[:, 0])
    np.testing.assert_array_equal(
        calls[2]["minute_feedback_features"][:, 1],
        result.ensemble.minute_features[:, 1])
