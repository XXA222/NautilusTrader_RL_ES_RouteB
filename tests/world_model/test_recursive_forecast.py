from __future__ import annotations

import math

import numpy as np

from rles_data.core import MINUTE_NS
from rles_world_model.contracts import MarketFrame
from rles_world_model.features import hourly_features
from rles_world_model.forecast import generated_scale_state_updater
from rles_world_model.scales import DailyD0State, GRUParameters, HourlyGRUState
from rles_world_model.temporal import UtcCandleAggregator


def test_generated_complete_hour_and_day_advance_each_path_slow_state() -> None:
    frames = []
    previous = 100.0
    for index in range(1, 31 * 1440 + 1):
        close = 100.0 + index * 1e-6
        ts = index * MINUTE_NS
        frames.append(MarketFrame(ts, ts, "SYNTHETIC", previous,
                                  max(previous, close) + 0.01,
                                  min(previous, close) - 0.01,
                                  close, float(index % 17)))
        previous = close
    history = tuple(frames)
    origin = history[-1]

    aggregator = UtcCandleAggregator()
    hours = []
    days = []
    for frame in history:
        completed = aggregator.push(frame, decision_ns=origin.available_at_ns)
        if completed.hour is not None:
            hours.append(completed.hour)
        if completed.day is not None:
            days.append(completed.day)

    params = GRUParameters.initialize(11, 32, seed=20261018)
    hour_state = HourlyGRUState(params)
    for index in range(24, len(hours)):
        hour_state.update(hours[index], np.asarray(
            hourly_features(hours[:index + 1]).values, dtype=np.float32,
        ))
    day_state = DailyD0State()
    for candle in days:
        day_state = day_state.update(candle)
    initial_hour = hour_state.state.copy()
    initial_day = day_state.features()

    updater = generated_scale_state_updater(
        history, origin=origin, initial_hour_state=initial_hour,
        initial_day_state=initial_day, paths=2,
    )
    slow_hour = np.repeat(initial_hour[None, :], 2, axis=0)
    slow_day = np.repeat(initial_day[None, :], 2, axis=0)
    samples = np.zeros((2, 5), dtype=np.float32)
    samples[:, 4] = math.log1p(1.0 / 1440.0)
    first_day_context = None
    for index in range(1440):
        slow_hour, slow_day = updater(samples, index, slow_hour, slow_day)
        if index == 59:
            assert not np.array_equal(slow_hour[0], initial_hour)
        if index == 1439:
            first_day_context = slow_day.copy()
    assert first_day_context is not None
    assert np.allclose(slow_hour[0], slow_hour[1])
    assert np.allclose(slow_day[0], slow_day[1])
    assert not np.array_equal(first_day_context[0], initial_day)
