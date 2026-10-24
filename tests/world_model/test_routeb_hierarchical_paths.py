from __future__ import annotations

import numpy as np
import pytest

from rles_world_model.routeb.codec import TrainOnlyStandardizer
from rles_world_model.routeb.hierarchical_paths import (
    PathSamplingError,
    sample_hierarchical_ohlcv_paths,
)
from rles_world_model.routeb.market import Candle, MINUTE_NS


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
