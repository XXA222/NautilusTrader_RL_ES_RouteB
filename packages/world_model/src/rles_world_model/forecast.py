"""Turn prior-only latent draws into legal minute and aggregated OHLCV paths."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from decimal import Decimal
import math
from typing import Callable

import numpy as np

from rles_data.core import DataError, MINUTE_NS, require_ns

from .contracts import MarketFrame
from .features import (
    OHLCVTarget, decode_ohlcv_target, hourly_features,
)
from .scales import DailyD0State, GRUParameters, HourlyGRUState
from .temporal import AggregatedCandle, UtcCandleAggregator


@dataclass(frozen=True, slots=True)
class MarketForecastPath:
    """One independent prior trajectory and its complete UTC aggregates."""

    minutes: tuple[MarketFrame, ...]
    hours: tuple[AggregatedCandle, ...]
    days: tuple[AggregatedCandle, ...]


ScaleStateUpdater = Callable[[np.ndarray, int, np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]


def generated_scale_state_updater(
    history: tuple[MarketFrame, ...],
    *,
    origin: MarketFrame,
    initial_hour_state: np.ndarray,
    initial_day_state: np.ndarray,
    paths: int,
    tick_size: float | None = None,
) -> ScaleStateUpdater:
    """Create isolated per-path UTC aggregators and advance slow states from samples only."""
    if type(history) is not tuple or not history or history[-1] != origin:
        raise DataError("recursive forecast requires causal history ending at its exact origin candle")
    if any(right.ts_close_ns - left.ts_close_ns != MINUTE_NS
           for left, right in zip(history, history[1:])):
        raise DataError("recursive forecast history must be contiguous 1m bars")
    if any(frame.instrument != origin.instrument or frame.available_at_ns > origin.available_at_ns
           for frame in history):
        raise DataError("recursive forecast history must be one instrument available by the origin")
    hour_initial = np.asarray(initial_hour_state, dtype=np.float32)
    day_initial = np.asarray(initial_day_state, dtype=np.float32)
    if hour_initial.shape != (32,) or day_initial.shape != (8,):
        raise DataError("recursive forecast slow-state snapshots must be 32-hour/8-day vectors")

    template = UtcCandleAggregator()
    hours: list[AggregatedCandle] = []
    days: list[AggregatedCandle] = []
    for frame in history:
        completed = template.push(frame, decision_ns=origin.available_at_ns)
        if completed.hour is not None:
            hours.append(completed.hour)
        if completed.day is not None:
            days.append(completed.day)
    if len(hours) < 25 or len(days) < 31:
        raise DataError("recursive forecast requires 24h/30d complete slow-state warmup")

    params = GRUParameters.initialize(11, 32, seed=20261018)
    check_hour = HourlyGRUState(params)
    for index in range(24, len(hours)):
        check_hour.update(hours[index], np.asarray(
            hourly_features(hours[:index + 1]).values, dtype=np.float32,
        ))
    if not np.allclose(check_hour.state, hour_initial, rtol=0, atol=1e-6):
        raise DataError("observed hourly GRU snapshot differs from its causal warmup history")
    day_tail = days[-30:]
    previous_days = days[-31:-1]
    day_state = DailyD0State(
        closes=tuple(candle.close for candle in day_tail),
        volumes=tuple(candle.volume for candle in day_tail),
        returns=tuple(math.log(candle.close / previous.close)
                      for candle, previous in zip(day_tail, previous_days, strict=True)),
        last_end_ns=day_tail[-1].end_ns,
    )
    if not np.allclose(day_state.features(), day_initial, rtol=0, atol=1e-6):
        raise DataError("observed daily D0 snapshot differs from its causal warmup history")

    aggregators = [deepcopy(template) for _ in range(paths)]
    hour_histories = [list(hours) for _ in range(paths)]
    hour_states = [HourlyGRUState(params, state=hour_initial,
                                  last_end_ns=hours[-1].end_ns) for _ in range(paths)]
    day_states = [day_state for _ in range(paths)]
    previous_closes = np.full(paths, origin.close, dtype=np.float64)

    def update(samples: np.ndarray, index: int, current_hours: np.ndarray,
               current_days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if samples.shape != (paths, 5):
            raise DataError("generated one-minute sample batch differs from path count")
        next_hours = np.asarray(current_hours, dtype=np.float32).copy()
        next_days = np.asarray(current_days, dtype=np.float32).copy()
        for path_index, sample in enumerate(samples):
            ts_close = origin.ts_close_ns + (index + 1) * MINUTE_NS
            available = max(ts_close, origin.available_at_ns)
            candle = decode_ohlcv_target(
                OHLCVTarget(*(float(value) for value in sample)),
                previous_close=float(previous_closes[path_index]),
                ts_close_ns=ts_close, available_at_ns=available,
                instrument=origin.instrument,
                tick_size=None if tick_size is None else Decimal(str(tick_size)),
            )
            completed = aggregators[path_index].push(candle, decision_ns=available)
            previous_closes[path_index] = candle.close
            if completed.hour is not None:
                hour_histories[path_index].append(completed.hour)
                vector = hourly_features(hour_histories[path_index][-25:])
                next_hours[path_index] = hour_states[path_index].update(
                    completed.hour, np.asarray(vector.values, dtype=np.float32),
                )
            if completed.day is not None:
                day_states[path_index] = day_states[path_index].update(completed.day)
                next_days[path_index] = day_states[path_index].features()
        return next_hours, next_days

    return update


def decode_prior_paths(
    samples: np.ndarray,
    *,
    previous_close: float,
    last_close_ns: int,
    available_from_ns: int,
    instrument: str,
    tick_size: float | None = None,
) -> tuple[MarketForecastPath, ...]:
    """Decode [path, minute, 5] samples; no observed future prices enter this call."""
    require_ns(last_close_ns)
    require_ns(available_from_ns)
    values = np.asarray(samples, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] != 5
            or not np.isfinite(values).all()):
        raise DataError("prior samples must be a finite [path, minute, 5] tensor")
    if last_close_ns % MINUTE_NS or available_from_ns < last_close_ns:
        raise DataError("forecast origin must be a closed, available minute")
    if not isinstance(instrument, str) or not instrument.strip():
        raise DataError("forecast instrument is required")
    if type(previous_close) not in (int, float) or not math.isfinite(previous_close) or previous_close <= 0:
        raise DataError("forecast origin close must be finite and positive")

    results = []
    for path_values in values:
        close = float(previous_close)
        aggregator = UtcCandleAggregator()
        minutes: list[MarketFrame] = []
        hours: list[AggregatedCandle] = []
        days: list[AggregatedCandle] = []
        for index, row in enumerate(path_values, start=1):
            ts_close = last_close_ns + index * MINUTE_NS
            available_at = max(ts_close, available_from_ns)
            candle = decode_ohlcv_target(
                OHLCVTarget(*(float(value) for value in row)),
                previous_close=close, ts_close_ns=ts_close,
                available_at_ns=available_at, instrument=instrument,
                tick_size=None if tick_size is None else Decimal(str(tick_size)),
            )
            frame = candle
            completed = aggregator.push(frame, decision_ns=frame.available_at_ns)
            minutes.append(frame)
            if completed.hour is not None:
                hours.append(completed.hour)
            if completed.day is not None:
                days.append(completed.day)
            close = candle.close
        results.append(MarketForecastPath(tuple(minutes), tuple(hours), tuple(days)))
    return tuple(results)
