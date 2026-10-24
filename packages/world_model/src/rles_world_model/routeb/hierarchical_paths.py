"""Sample legal minute OHLCV paths and reconcile their UTC slow buckets.

The RSSM feature heads are not slow-scale price forecasts. This module turns
each sampled minute feature trajectory into one coherent OHLCV world and
derives its completed hour/day candles only from that same path plus the
observed prefix. Callers may feed those generated candles into the next slow
RSSM update; they must not substitute the latent decoder diagnostics.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

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


@dataclass(frozen=True, slots=True)
class RecursiveHierarchicalOHLCVPaths:
    """Recursively generated OHLCV paths and per-bar rejection accounting."""

    ensemble: HierarchicalOHLCVPaths
    attempted_bar_samples: int
    rejected_bar_samples: int

    def __post_init__(self) -> None:
        if not isinstance(self.ensemble, HierarchicalOHLCVPaths):
            raise ValueError("recursive path result requires a validated OHLCV ensemble")
        accepted = self.ensemble.minute.shape[0] * self.ensemble.minute.shape[1]
        if (type(self.attempted_bar_samples) is not int
                or type(self.rejected_bar_samples) is not int
                or self.attempted_bar_samples < accepted
                or self.rejected_bar_samples < 0
                or self.attempted_bar_samples - self.rejected_bar_samples != accepted):
            raise ValueError("recursive per-bar sampling accounting is inconsistent")

    @property
    def rejection_rate(self) -> float:
        return self.rejected_bar_samples / self.attempted_bar_samples


class StatefulRecursivePredictor(Protocol):
    """Cached predictor that consumes one sampled bar before its next forecast."""

    def predict_step(self, *, step: int, minute_feedback: np.ndarray,
                     hour_feedback: np.ndarray, day_feedback: np.ndarray,
                     hour_event: bool, day_event: bool
                     ) -> tuple[np.ndarray, np.ndarray]: ...


RecursivePredictor = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
] | StatefulRecursivePredictor

MAX_TOTAL_BAR_ATTEMPTS = 1_000_000


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


def generate_recursive_hierarchical_ohlcv_paths(*, predictor: RecursivePredictor,
        future_minute_decisions_ns: np.ndarray,
        future_hour_events: tuple[bool, ...], future_day_events: tuple[bool, ...],
        hour_availability_delay_ns: int, day_availability_delay_ns: int,
        minute_scaler: TrainOnlyStandardizer, hour_scaler: TrainOnlyStandardizer,
        day_scaler: TrainOnlyStandardizer, observed_prefix: tuple[Candle, ...],
        paths: int, seed: int, max_attempts_per_bar: int = 20,
        max_total_bar_attempts: int = 250_000
        ) -> RecursiveHierarchicalOHLCVPaths:
    """Generate one legal minute bar at a time and feed it into the next forecast.

    ``predictor`` receives the decision prefix and path-aligned minute/hour/day
    feature feedback, then returns the conditional distribution for the final
    decision as ``[paths,5]`` mean and scale arrays. Feedback at the current
    decision is zero until that decision's minute candle has been sampled; its
    features can therefore affect only subsequent forecasts. Slow feedback is
    rebuilt from completed UTC candles on each path and aligned to its actual
    availability event. This is a correctness-first reference orchestration;
    callers with long horizons should provide a stateful predictor to avoid
    replaying the observed history for every decision.
    """
    decisions = np.asarray(future_minute_decisions_ns)
    if (not (callable(predictor) or callable(getattr(predictor, "predict_step", None)))
            or decisions.ndim != 1
            or decisions.dtype.kind not in "iu" or len(decisions) < 1
            or np.any(decisions <= 0) or np.any(np.diff(decisions) != MINUTE_NS)
            or not isinstance(observed_prefix, tuple) or not observed_prefix
            or any(not isinstance(bar, Candle) for bar in observed_prefix)
            or type(hour_availability_delay_ns) is not int
            or hour_availability_delay_ns < 0
            or type(day_availability_delay_ns) is not int
            or day_availability_delay_ns < 0
            or type(paths) is not int or paths < 1
            or type(seed) is not int or seed < 0
            or type(max_attempts_per_bar) is not int or max_attempts_per_bar < 1
            or type(max_total_bar_attempts) is not int
            or not 1 <= max_total_bar_attempts <= MAX_TOTAL_BAR_ATTEMPTS):
        raise ValueError("invalid recursive OHLCV generation inputs")
    maximum_attempts = paths * len(decisions) * max_attempts_per_bar
    if maximum_attempts > max_total_bar_attempts:
        raise ValueError(
            "recursive OHLCV worst-case bar attempts exceed the explicit total budget")
    anchor = observed_prefix[-1].ts_close_ns
    if observed_prefix[-1].available_at_ns >= decisions[0]:
        raise ValueError("recursive decisions must follow the observed availability barrier")
    if anchor > np.iinfo(np.int64).max - len(decisions) * MINUTE_NS:
        raise ValueError("recursive decision horizon overflows int64 timestamps")
    expected_decisions = anchor + np.arange(1, len(decisions) + 1, dtype=np.int64) * MINUTE_NS
    if not np.array_equal(decisions, expected_decisions):
        raise ValueError("recursive decisions must match the contiguous generated minute closes")
    if (len(future_hour_events) != len(decisions)
            or len(future_day_events) != len(decisions)
            or any(type(event) is not bool
                   for event in (*future_hour_events, *future_day_events))):
        raise ValueError("recursive slow event plans must align with every minute decision")
    scalers = (minute_scaler, hour_scaler, day_scaler)
    if (any(not isinstance(scaler, TrainOnlyStandardizer) for scaler in scalers)
            or any(scaler.train_scope != "train" for scaler in scalers)
            or len({scaler.dataset_id for scaler in scalers}) != 1):
        raise ValueError("recursive generation requires frozen same-dataset Train scalers")
    MultiScaleClock(observed_prefix)

    minute_bars: list[list[np.ndarray]] = [[] for _ in range(paths)]
    minute_features: list[list[np.ndarray]] = [[] for _ in range(paths)]
    path_sources: list[list[Candle]] = [list(observed_prefix) for _ in range(paths)]
    hour_bars: list[list[np.ndarray]] = [[] for _ in range(paths)]
    day_bars: list[list[np.ndarray]] = [[] for _ in range(paths)]
    hour_features: list[list[np.ndarray]] = [[] for _ in range(paths)]
    day_features: list[list[np.ndarray]] = [[] for _ in range(paths)]
    hour_closes: list[int] = []
    day_closes: list[int] = []
    attempted_samples = rejected_samples = 0

    for step in range(len(decisions)):
        decision_prefix = decisions[:step + 1]
        minute_feedback = np.zeros((paths, step + 1, 5), dtype=np.float32)
        hour_feedback = np.zeros_like(minute_feedback)
        day_feedback = np.zeros_like(minute_feedback)
        if step:
            for path_index in range(paths):
                minute_feedback[path_index, :step] = np.stack(
                    minute_features[path_index])
                if hour_closes:
                    hour_feedback[path_index, :step] = align_bucket_feedback(
                        close_ns=np.asarray(hour_closes, dtype=np.int64),
                        feedback=np.asarray(hour_features[path_index], dtype=np.float32)[None],
                        minute_decisions_ns=decisions[:step],
                        availability_delay_ns=hour_availability_delay_ns,
                        expected_events=future_hour_events[:step])[0]
                elif any(future_hour_events[:step]):
                    raise ValueError("a prior hour event has no generated path aggregate")
                if day_closes:
                    day_feedback[path_index, :step] = align_bucket_feedback(
                        close_ns=np.asarray(day_closes, dtype=np.int64),
                        feedback=np.asarray(day_features[path_index], dtype=np.float32)[None],
                        minute_decisions_ns=decisions[:step],
                        availability_delay_ns=day_availability_delay_ns,
                        expected_events=future_day_events[:step])[0]
                elif any(future_day_events[:step]):
                    raise ValueError("a prior day event has no generated path aggregate")

        stateful_step = getattr(predictor, "predict_step", None)
        if callable(stateful_step):
            if step == 0:
                prior_minute = np.zeros((paths, 5), dtype=np.float32)
                prior_hour = np.zeros_like(prior_minute)
                prior_day = np.zeros_like(prior_minute)
                hour_event = day_event = False
            else:
                prior_minute = minute_feedback[:, step - 1]
                prior_hour = hour_feedback[:, step - 1]
                prior_day = day_feedback[:, step - 1]
                hour_event = future_hour_events[step - 1]
                day_event = future_day_events[step - 1]
            mean, scale = stateful_step(
                step=step, minute_feedback=prior_minute,
                hour_feedback=prior_hour, day_feedback=prior_day,
                hour_event=hour_event, day_event=day_event)
        elif callable(predictor):
            mean, scale = predictor(decision_prefix, minute_feedback,
                                    hour_feedback, day_feedback)
        else:
            raise ValueError("recursive predictor must be callable or implement predict_step")
        mean = np.asarray(mean, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        if (mean.shape != (paths, 5) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise ValueError("recursive predictor must return finite positive [paths,5] heads")

        sampled_at_step = []
        for path_index in range(paths):
            step_seed = int(np.random.SeedSequence(
                [seed, path_index, step]).generate_state(1, dtype=np.uint64)[0])
            sampled = sample_hierarchical_ohlcv_paths(
                minute_mean=mean[path_index][None],
                minute_scale=scale[path_index][None],
                minute_scaler=minute_scaler, hour_scaler=hour_scaler,
                day_scaler=day_scaler,
                observed_prefix=tuple(path_sources[path_index]), paths=1,
                seed=step_seed, max_attempts_per_path=max_attempts_per_bar)
            attempted_samples += sampled.attempted_paths
            rejected_samples += sampled.rejected_paths
            minute = sampled.minute[0, 0]
            features = sampled.minute_features[0, 0]
            close_ns = int(decisions[step])
            bar = Candle(close_ns, close_ns, path_sources[path_index][-1].instrument,
                         *map(float, minute))
            path_sources[path_index].append(bar)
            minute_bars[path_index].append(minute)
            minute_features[path_index].append(features)
            hour_bars[path_index].extend(sampled.hour[0])
            day_bars[path_index].extend(sampled.day[0])
            hour_features[path_index].extend(sampled.hour_feedback[0])
            day_features[path_index].extend(sampled.day_feedback[0])
            sampled_at_step.append(sampled)

        reference = sampled_at_step[0]
        for sampled in sampled_at_step[1:]:
            if (not np.array_equal(sampled.hour_close_ns, reference.hour_close_ns)
                    or not np.array_equal(sampled.day_close_ns, reference.day_close_ns)):
                raise RuntimeError("recursive paths produced different UTC bucket boundaries")
        for close in reference.hour_close_ns:
            if int(close) not in hour_closes:
                hour_closes.append(int(close))
        for close in reference.day_close_ns:
            if int(close) not in day_closes:
                day_closes.append(int(close))

    for path_index in range(paths):
        align_bucket_feedback(
            close_ns=np.asarray(hour_closes, dtype=np.int64),
            feedback=np.asarray(hour_features[path_index], dtype=np.float32).reshape(
                (1, len(hour_closes), 5)),
            minute_decisions_ns=decisions,
            availability_delay_ns=hour_availability_delay_ns,
            expected_events=future_hour_events)
        align_bucket_feedback(
            close_ns=np.asarray(day_closes, dtype=np.int64),
            feedback=np.asarray(day_features[path_index], dtype=np.float32).reshape(
                (1, len(day_closes), 5)),
            minute_decisions_ns=decisions,
            availability_delay_ns=day_availability_delay_ns,
            expected_events=future_day_events)

    def stack_path_rows(rows: list[list[np.ndarray]]) -> np.ndarray:
        if not rows or not rows[0]:
            return np.empty((paths, 0, 5), dtype=np.float64)
        return np.stack([np.stack(path_rows) for path_rows in rows])

    ensemble = HierarchicalOHLCVPaths(
        stack_path_rows(minute_bars),
        np.asarray([[row for row in path_rows] for path_rows in minute_features],
                   dtype=np.float32),
        stack_path_rows(hour_bars), stack_path_rows(day_bars),
        np.asarray([[row for row in path_rows] for path_rows in hour_features],
                   dtype=np.float32).reshape((paths, len(hour_closes), 5)),
        np.asarray([[row for row in path_rows] for path_rows in day_features],
                   dtype=np.float32).reshape((paths, len(day_closes), 5)),
        np.asarray(hour_closes, dtype=np.int64), np.asarray(day_closes, dtype=np.int64),
        paths, 0,
    )
    return RecursiveHierarchicalOHLCVPaths(
        ensemble, attempted_samples, rejected_samples)
