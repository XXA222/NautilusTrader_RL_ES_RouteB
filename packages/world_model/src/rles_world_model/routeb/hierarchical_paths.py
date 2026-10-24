"""Sample legal minute OHLCV paths and reconcile their UTC slow buckets.

The RSSM feature heads are not slow-scale price forecasts. This module turns
each sampled minute feature trajectory into one coherent OHLCV world and
derives its completed hour/day candles only from that same path plus the
observed prefix. Callers may feed those generated candles into the next slow
RSSM update; they must not substitute the latent decoder diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clock import ClosedBucket, MultiScaleClock
from .codec import OHLCVCodec, TrainOnlyStandardizer
from .market import Candle, MINUTE_NS


class PathSamplingError(ValueError):
    """A fixed-size path ensemble could not be filled without clipping samples."""

    def __init__(self, *, requested: int, attempted: int, rejected: int) -> None:
        self.requested = requested
        self.attempted = attempted
        self.rejected = rejected
        super().__init__(
            f"only {attempted - rejected} of {requested} valid OHLCV paths after "
            f"{attempted} attempts ({rejected} rejected)"
        )


@dataclass(frozen=True, slots=True)
class HierarchicalOHLCVPaths:
    """One accepted minute-path ensemble and its path-derived slow candles."""

    minute: np.ndarray  # [paths,horizon,5] OHLCV
    minute_features: np.ndarray  # [paths,horizon,5] sampled, Train-standardized coordinates
    hour: np.ndarray  # [paths,completed future hours,5] OHLCV
    day: np.ndarray  # [paths,completed future days,5] OHLCV
    hour_feedback: np.ndarray  # [paths,completed future hours,5] Train-standardized aggregate features
    day_feedback: np.ndarray  # [paths,completed future days,5] Train-standardized aggregate features
    hour_close_ns: np.ndarray
    day_close_ns: np.ndarray
    attempted_paths: int
    rejected_paths: int

    def __post_init__(self) -> None:
        minute = np.array(self.minute, dtype=np.float64, copy=True)
        minute_features = np.array(self.minute_features, dtype=np.float32, copy=True)
        hour = np.array(self.hour, dtype=np.float64, copy=True)
        day = np.array(self.day, dtype=np.float64, copy=True)
        hour_feedback = np.array(self.hour_feedback, dtype=np.float32, copy=True)
        day_feedback = np.array(self.day_feedback, dtype=np.float32, copy=True)
        hour_close = np.array(self.hour_close_ns, dtype=np.int64, copy=True)
        day_close = np.array(self.day_close_ns, dtype=np.int64, copy=True)
        if (minute.ndim != 3 or minute.shape[0] < 1 or minute.shape[1] < 1
                or minute.shape[2] != 5 or hour.shape != (minute.shape[0], len(hour_close), 5)
                or minute_features.shape != minute.shape
                or day.shape != (minute.shape[0], len(day_close), 5)
                or hour_feedback.shape != hour.shape or day_feedback.shape != day.shape
                or hour_close.ndim != 1 or day_close.ndim != 1
                or any(not np.isfinite(values).all() for values in
                       (minute, minute_features, hour, day, hour_feedback, day_feedback))
                or type(self.attempted_paths) is not int
                or type(self.rejected_paths) is not int
                or self.attempted_paths < minute.shape[0]
                or not 0 <= self.rejected_paths <= self.attempted_paths
                or self.attempted_paths - self.rejected_paths != minute.shape[0]):
            raise ValueError("invalid path ensemble, bucket alignment or rejection accounting")
        for value in (minute, minute_features, hour, day, hour_feedback,
                      day_feedback, hour_close, day_close):
            value.setflags(write=False)
        object.__setattr__(self, "minute", minute)
        object.__setattr__(self, "minute_features", minute_features)
        object.__setattr__(self, "hour", hour)
        object.__setattr__(self, "day", day)
        object.__setattr__(self, "hour_feedback", hour_feedback)
        object.__setattr__(self, "day_feedback", day_feedback)
        object.__setattr__(self, "hour_close_ns", hour_close)
        object.__setattr__(self, "day_close_ns", day_close)

    @property
    def rejection_rate(self) -> float:
        return self.rejected_paths / self.attempted_paths


def _ohlcv(bucket: ClosedBucket) -> tuple[float, float, float, float, float]:
    return bucket.open, bucket.high, bucket.low, bucket.close, bucket.volume


def _verify_bucket_from_source(bucket: ClosedBucket, source: tuple[Candle, ...]) -> None:
    first_close = bucket.close_ns - (bucket.timeframe_minutes - 1) * MINUTE_NS
    rows = tuple(bar for bar in source
                 if first_close <= bar.ts_close_ns <= bucket.close_ns)
    if (len(rows) != bucket.timeframe_minutes
            or rows[0].ts_close_ns != first_close
            or rows[-1].ts_close_ns != bucket.close_ns):
        raise RuntimeError("completed slow bucket lacks its full minute source interval")
    expected = (rows[0].open, max(bar.high for bar in rows),
                min(bar.low for bar in rows), rows[-1].close,
                sum(bar.volume for bar in rows))
    if expected != _ohlcv(bucket):
        raise RuntimeError("slow OHLCV bucket does not exactly conserve its minute path")


def _standardized_bucket_feedback(bucket: ClosedBucket, source: tuple[Candle, ...],
                                  scaler: TrainOnlyStandardizer) -> np.ndarray:
    previous_close_ns = bucket.close_ns - bucket.timeframe_minutes * MINUTE_NS
    prior = next((bar for bar in reversed(source)
                  if bar.ts_close_ns == previous_close_ns), None)
    # Match hierarchy_dataset._coords: if the captured source begins with the
    # first full slow bucket, its open is the only valid initial reference.
    previous_close = bucket.open if prior is None else prior.close
    candle = Candle(bucket.close_ns, bucket.available_at_ns, bucket.instrument,
                    bucket.open, bucket.high, bucket.low, bucket.close, bucket.volume)
    coordinates = OHLCVCodec.encode((candle,), previous_close=previous_close)
    return scaler.transform(coordinates)[0]


def sample_hierarchical_ohlcv_paths(*, minute_mean: np.ndarray,
        minute_scale: np.ndarray, minute_scaler: TrainOnlyStandardizer,
        hour_scaler: TrainOnlyStandardizer, day_scaler: TrainOnlyStandardizer,
        observed_prefix: tuple[Candle, ...], paths: int, seed: int,
        max_attempts_per_path: int = 20) -> HierarchicalOHLCVPaths:
    """Sample/decode complete paths, rejecting unsafe draws without clipping.

    ``minute_mean`` and ``minute_scale`` are the standardized-coordinate
    conditional heads for one forecast anchor, shaped ``[horizon,5]``. The
    scalers must be frozen on the same Train dataset. Returned
    ``hour_feedback``/``day_feedback`` are aggregate-derived features that a
    caller can feed to the matching slow RSSM posterior at each emitted close.
    Any coordinate/geometry rejected by
    ``OHLCVCodec`` consumes an attempt and is counted; it is never repaired.
    Hour/day outputs contain only completed buckets whose close is after the
    observed anchor, and every returned bucket is re-derived and checked from
    the same observed-prefix + simulated-minute sequence.
    """
    mean = np.asarray(minute_mean, dtype=np.float64)
    scale = np.asarray(minute_scale, dtype=np.float64)
    if (mean.ndim != 2 or mean.shape[1] != 5 or scale.shape != mean.shape
            or mean.shape[0] < 1 or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("finite [horizon,5] standardized minute distributions are required")
    if (not isinstance(minute_scaler, TrainOnlyStandardizer)
            or not isinstance(hour_scaler, TrainOnlyStandardizer)
            or not isinstance(day_scaler, TrainOnlyStandardizer)
            or any(scaler.train_scope != "train" for scaler in
                   (minute_scaler, hour_scaler, day_scaler))
            or len({scaler.dataset_id for scaler in
                    (minute_scaler, hour_scaler, day_scaler)}) != 1):
        raise ValueError("frozen Train-only minute/hour/day scalers for one dataset are required")
    if (not isinstance(observed_prefix, tuple) or not observed_prefix
            or any(not isinstance(bar, Candle) for bar in observed_prefix)
            or type(paths) is not int or paths < 1 or type(seed) is not int or seed < 0
            or type(max_attempts_per_path) is not int or max_attempts_per_path < 1):
        raise ValueError("invalid observed prefix or path sampling configuration")
    anchor = observed_prefix[-1].ts_close_ns
    if any(bar.ts_close_ns > anchor or bar.available_at_ns > anchor
           for bar in observed_prefix):
        raise ValueError("observed prefix contains a future or unavailable candle")
    # This validates continuity, instrument identity, and availability order.
    MultiScaleClock(observed_prefix)

    rng = np.random.default_rng(seed)
    accepted_minute: list[np.ndarray] = []
    accepted_minute_features: list[np.ndarray] = []
    accepted_hour: list[np.ndarray] = []
    accepted_day: list[np.ndarray] = []
    accepted_hour_feedback: list[np.ndarray] = []
    accepted_day_feedback: list[np.ndarray] = []
    hour_close_ns: np.ndarray | None = None
    day_close_ns: np.ndarray | None = None
    attempted = rejected = 0
    max_attempts = paths * max_attempts_per_path
    while len(accepted_minute) < paths and attempted < max_attempts:
        attempted += 1
        standardized = rng.normal(mean, scale)
        coordinates = minute_scaler.inverse(standardized)
        try:
            simulated = OHLCVCodec.decode(
                coordinates, previous_close=observed_prefix[-1].close,
                start_close_ns=anchor + MINUTE_NS,
                instrument=observed_prefix[-1].instrument,
            )
        except ValueError:
            rejected += 1
            continue

        source = observed_prefix + simulated
        clock = MultiScaleClock(source)
        end_ns = simulated[-1].ts_close_ns
        hour_buckets = tuple(bucket for bucket in clock.buckets(60, as_of_ns=end_ns)
                             if anchor < bucket.close_ns <= end_ns)
        day_buckets = tuple(bucket for bucket in clock.buckets(1440, as_of_ns=end_ns)
                            if anchor < bucket.close_ns <= end_ns)
        for bucket in (*hour_buckets, *day_buckets):
            _verify_bucket_from_source(bucket, source)
        current_hour_closes = np.asarray([bucket.close_ns for bucket in hour_buckets],
                                         dtype=np.int64)
        current_day_closes = np.asarray([bucket.close_ns for bucket in day_buckets],
                                        dtype=np.int64)
        if hour_close_ns is None or day_close_ns is None:
            hour_close_ns, day_close_ns = current_hour_closes, current_day_closes
        else:
            if (not np.array_equal(hour_close_ns, current_hour_closes)
                    or not np.array_equal(day_close_ns, current_day_closes)):
                raise RuntimeError("path sampling changed UTC bucket boundaries")
        accepted_minute.append(np.asarray(
            [[bar.open, bar.high, bar.low, bar.close, bar.volume] for bar in simulated],
            dtype=np.float64))
        accepted_minute_features.append(np.asarray(standardized, dtype=np.float32))
        accepted_hour.append(np.asarray([_ohlcv(bucket) for bucket in hour_buckets],
                                        dtype=np.float64).reshape((-1, 5)))
        accepted_day.append(np.asarray([_ohlcv(bucket) for bucket in day_buckets],
                                       dtype=np.float64).reshape((-1, 5)))
        accepted_hour_feedback.append(np.asarray([
            _standardized_bucket_feedback(bucket, source, hour_scaler)
            for bucket in hour_buckets], dtype=np.float32).reshape((-1, 5)))
        accepted_day_feedback.append(np.asarray([
            _standardized_bucket_feedback(bucket, source, day_scaler)
            for bucket in day_buckets], dtype=np.float32).reshape((-1, 5)))

    if len(accepted_minute) != paths:
        raise PathSamplingError(requested=paths, attempted=attempted, rejected=rejected)
    assert hour_close_ns is not None and day_close_ns is not None
    return HierarchicalOHLCVPaths(
        np.stack(accepted_minute), np.stack(accepted_minute_features),
        np.stack(accepted_hour), np.stack(accepted_day),
        np.stack(accepted_hour_feedback), np.stack(accepted_day_feedback),
        hour_close_ns, day_close_ns, attempted, rejected,
    )


def align_bucket_feedback(*, close_ns: np.ndarray, feedback: np.ndarray,
        minute_decisions_ns: np.ndarray, availability_delay_ns: int,
        expected_events: tuple[bool, ...]) -> np.ndarray:
    """Place each path-derived slow feature at its causal future event step.

    Forecast event plans are computed from expected close times plus a frozen
    availability delay. This helper aligns only generated aggregate features;
    it does not alter the event plan or expose the feature before its event.
    """
    closes = np.asarray(close_ns)
    values = np.asarray(feedback, dtype=np.float32)
    decisions = np.asarray(minute_decisions_ns)
    events = np.asarray(expected_events)
    if (closes.ndim != 1 or closes.dtype.kind not in "iu"
            or np.any(closes <= 0) or np.any(np.diff(closes) <= 0)
            or values.ndim != 3 or values.shape[1] != len(closes)
            or values.shape[2] != 5 or not np.isfinite(values).all()
            or decisions.ndim != 1 or decisions.dtype.kind not in "iu"
            or not len(decisions) or np.any(decisions <= 0)
            or np.any(np.diff(decisions) <= 0)
            or events.dtype.kind != "b" or events.shape != decisions.shape
            or type(availability_delay_ns) is not int or availability_delay_ns < 0):
        raise ValueError("invalid path-feedback alignment inputs")
    aligned = np.zeros((values.shape[0], len(decisions), 5), dtype=np.float32)
    assigned = np.zeros(len(decisions), dtype=np.bool_)
    for index, close in enumerate(closes):
        available_at = int(close) + availability_delay_ns
        step = int(np.searchsorted(decisions, available_at, side="left"))
        if step == len(decisions):
            continue
        if not events[step] or assigned[step]:
            raise ValueError("path-derived slow close does not match its availability event")
        aligned[:, step] = values[:, index]
        assigned[step] = True
    if not np.array_equal(assigned, events):
        raise ValueError("future slow event plan has no matching path-derived aggregate")
    return aligned
