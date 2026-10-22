"""Causal multi-scale supervised sequences with Train-only robust scaling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rles_data.core import DataError, require_ns

from .contracts import MarketReplay
from .features import (
    MarketFeatureSpec, MarketFeatureVector, TrainOnlyScaler, calendar_features,
    daily_features, encode_ohlcv_target, hourly_features, minute_features,
)
from .scales import DailyD0State, GRUParameters, HourlyGRUState
from .temporal import AggregatedCandle, UtcCandleAggregator

SCALES = ("1m", "1h", "1d", "calendar")


@dataclass(frozen=True, slots=True)
class RawTrainingRow:
    timestamp_ns: int
    features: tuple[MarketFeatureVector, ...]
    scale_state: tuple[float, ...]
    target: tuple[float, float, float, float, float]
    hourly_target: tuple[float, float, float, float, float] | None


@dataclass(frozen=True, slots=True)
class TrainSequences:
    observations: np.ndarray
    targets: np.ndarray
    hourly_targets: np.ndarray
    hourly_mask: np.ndarray
    reset: np.ndarray
    scalers: dict[str, TrainOnlyScaler]
    timestamps_ns: tuple[int, ...]
    clipped_values: int


def build_train_sequences(
    replay: MarketReplay,
    *,
    train_start_ns: int,
    train_end_ns: int,
    spec: MarketFeatureSpec | None = None,
    clip_sigma: float = 8.0,
) -> TrainSequences:
    """Construct next-minute supervision without reading outside the Train interval."""
    require_ns(train_start_ns)
    require_ns(train_end_ns)
    if train_end_ns <= train_start_ns:
        raise DataError("Train sequence interval must be positive")
    if not isinstance(replay, MarketReplay) or not replay.is_contiguous:
        raise DataError("world-model Train replay must be a contiguous frozen MarketReplay")
    if any(right.ts_close_ns <= left.ts_close_ns
           for left, right in zip(replay.frames, replay.frames[1:])):
        raise DataError("world-model Train replay cannot contain duplicate close times")
    spec = spec or MarketFeatureSpec()
    train_hours = _complete_train_hours(replay, train_end_ns)
    hour_auxiliary_by_end = {
        current.end_ns: _target_tuple(
            encode_ohlcv_target(following, previous_close=current.close),
        )
        for current, following in zip(train_hours, train_hours[1:])
        if following.end_ns < train_end_ns and following.available_at_ns <= train_end_ns
    }
    aggregator = UtcCandleAggregator()
    hour_state = HourlyGRUState(
        GRUParameters.initialize(len(spec.hour_names), 32, seed=20261018),
    )
    day_state = DailyD0State()
    hours: list[AggregatedCandle] = []
    days: list[AggregatedCandle] = []
    rows: list[RawTrainingRow] = []

    for index, frame in enumerate(replay.frames[:-1]):
        if frame.ts_close_ns >= train_end_ns or frame.available_at_ns > train_end_ns:
            # Do not advance state or fit against Validation/Test rows that may
            # coexist in a wider frozen Fold replay.
            break
        decision = frame.available_at_ns
        if frame.ts_close_ns < train_start_ns or frame.ts_close_ns >= train_end_ns:
            # Still advance causal period state through warmup; it is never used
            # to fit the scaler or optimize model weights.
            completed = aggregator.push(frame, decision_ns=decision)
            if completed.hour:
                hours.append(completed.hour)
                if len(hours) >= 25:
                    warm_hour = hourly_features(hours, spec=spec)
                    hour_state.update(completed.hour,
                                      np.asarray(warm_hour.values, dtype=np.float32))
            if completed.day:
                days.append(completed.day)
                day_state = day_state.update(completed.day)
            continue
        completed = aggregator.push(frame, decision_ns=decision)
        if completed.hour:
            hours.append(completed.hour)
            if len(hours) >= 25:
                hour_vector = hourly_features(hours, spec=spec)
                hour_state.update(completed.hour, np.asarray(hour_vector.values, dtype=np.float32))
        if completed.day:
            days.append(completed.day)
            day_state = day_state.update(completed.day)
        target_frame = replay.frames[index + 1]
        if target_frame.ts_close_ns >= train_end_ns or target_frame.available_at_ns > train_end_ns:
            continue
        visible_minute = replay.frames[max(0, index - 15):index + 1]
        if len(visible_minute) < 16:
            continue
        if (len(hours) < 25 or hours[-1].available_at_ns > decision
                or len(days) < 31 or days[-1].available_at_ns > decision
                or hour_state.last_end_ns is None or hour_state.last_end_ns > decision
                or day_state.last_end_ns is None or day_state.last_end_ns > decision):
            continue
        minute_vector = minute_features(visible_minute, spec=spec)
        hour_vector = hourly_features(hours, spec=spec)
        day_vector = daily_features(days, spec=spec)
        # The next UTC minute is deterministic and is the forecast target time.
        calendar_vector = calendar_features(target_frame.ts_close_ns, spec=spec)
        target = encode_ohlcv_target(target_frame, previous_close=frame.close)
        rows.append(RawTrainingRow(
            frame.ts_close_ns,
            (minute_vector, hour_vector, day_vector, calendar_vector),
            tuple(float(value) for value in
                  (*hour_state.state, *day_state.features())),
            (target.open_gap, target.body, target.upper_wick,
             target.lower_wick, target.log1p_volume),
            hour_auxiliary_by_end.get(frame.ts_close_ns),
        ))

    if not rows:
        raise DataError("Train interval has no samples after causal 1m/1h/1d warmup")
    scaler_map = {
        scale: TrainOnlyScaler.fit(
            [row.features[position] for row in rows],
            train_start_ns=train_start_ns, train_end_ns=train_end_ns,
            clip_sigma=clip_sigma,
        )
        for position, scale in enumerate(SCALES)
    }
    obs, clipped = _transform_rows(rows, scaler_map)
    targets = np.asarray([row.target for row in rows], dtype=np.float32)
    hourly_targets = np.asarray([
        (0.0, 0.0, 0.0, 0.0, 0.0) if row.hourly_target is None else row.hourly_target
        for row in rows
    ], dtype=np.float32)
    hourly_mask = np.asarray([row.hourly_target is not None for row in rows], dtype=np.bool_)
    reset = np.zeros((1, len(rows)), dtype=np.bool_)
    reset[0, 0] = True
    return TrainSequences(
        observations=obs[None, ...], targets=targets[None, ...],
        hourly_targets=hourly_targets[None, ...], hourly_mask=hourly_mask[None, ...],
        reset=reset,
        scalers=scaler_map, timestamps_ns=tuple(row.timestamp_ns for row in rows),
        clipped_values=clipped,
    )


def transform_feature_rows(rows: tuple[RawTrainingRow, ...],
                           scalers: dict[str, TrainOnlyScaler]) -> tuple[np.ndarray, int]:
    """Transform Validation/Test rows with already frozen Train scalers."""
    if set(scalers) != set(SCALES):
        raise DataError("frozen Route B scaler set must contain 1m/1h/1d/calendar")
    if any(scalers[key].scale != key for key in SCALES):
        raise DataError("frozen scaler key and scale differ")
    return _transform_rows(list(rows), scalers)


def _transform_rows(rows: list[RawTrainingRow], scalers: dict[str, TrainOnlyScaler]
                    ) -> tuple[np.ndarray, int]:
    vectors: list[list[float]] = []
    clipped = 0
    for row in rows:
        parts: list[float] = []
        for scale, vector in zip(SCALES, row.features, strict=True):
            scaled = scalers[scale].transform(vector)
            parts.extend(scaled.values)
            clipped += scaled.clipped_count
        parts.extend(row.scale_state)
        vectors.append(parts)
    return np.asarray(vectors, dtype=np.float32), clipped


def _complete_train_hours(replay: MarketReplay, train_end_ns: int) -> list[AggregatedCandle]:
    aggregator = UtcCandleAggregator()
    hours: list[AggregatedCandle] = []
    for frame in replay.frames:
        if frame.ts_close_ns >= train_end_ns or frame.available_at_ns > train_end_ns:
            break
        completed = aggregator.push(frame, decision_ns=frame.available_at_ns)
        if completed.hour is not None:
            hours.append(completed.hour)
    return hours


def _target_tuple(target) -> tuple[float, float, float, float, float]:
    return (target.open_gap, target.body, target.upper_wick,
            target.lower_wick, target.log1p_volume)
