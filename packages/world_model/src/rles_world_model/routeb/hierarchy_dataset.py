"""Train-only causal tensor projection for the experimental B3 three-scale learner.

This module prepares ONE contiguous approved Train sequence. It intentionally
is not a general sampler, model-selection service, or authorization bypass.
Raw 1m candles are the sole source of 1h/1d candles. A slow latent is referenced
only after its final constituent 1m candle becomes available.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .clock import ClosedBucket, MultiScaleClock
from .codec import OHLCVCodec, TrainOnlyStandardizer
from .hierarchy import build_plan
from .market import Candle, MarketSequence, MINUTE_NS


@dataclass(frozen=True, slots=True)
class HierarchyScalers:
    minute: TrainOnlyStandardizer
    hour: TrainOnlyStandardizer
    day: TrainOnlyStandardizer


@dataclass(frozen=True, slots=True)
class HierarchyTensors:
    minute_x: np.ndarray          # [1,T,5], float32
    hour_x: np.ndarray            # [1,H,5], float32
    day_x: np.ndarray             # [1,D,5], float32
    hour_index: np.ndarray        # [1,T], int32, -1 before first available H
    day_index: np.ndarray         # [1,T], int32, -1 before first available D
    minute_mask: np.ndarray       # [1,T], float32, targets after burn-in
    hour_mask: np.ndarray         # [1,H], float32
    day_mask: np.ndarray          # [1,D], float32
    minute_available_at: np.ndarray  # [T]
    hour_available_at: np.ndarray    # [H]
    day_available_at: np.ndarray     # [D]
    hour_close_at: np.ndarray        # [H]
    day_close_at: np.ndarray         # [D]

    def __post_init__(self) -> None:
        if (self.minute_x.ndim != 3 or self.minute_x.shape[0] != 1
                or self.minute_x.shape[2] != 5 or self.hour_x.ndim != 3
                or self.hour_x.shape[0] != 1 or self.hour_x.shape[2] != 5
                or self.day_x.ndim != 3 or self.day_x.shape[0] != 1
                or self.day_x.shape[2] != 5 or self.minute_x.shape[1] < 3
                or self.hour_x.shape[1] < 2 or self.day_x.shape[1] < 2):
            raise ValueError('hierarchy features need one nonempty [time,5] sequence per scale')
        if any(not np.isfinite(x).all() for x in (self.minute_x,self.hour_x,self.day_x)):
            raise ValueError('hierarchy features must be finite')
        t, h, d = self.minute_x.shape[1], self.hour_x.shape[1], self.day_x.shape[1]
        if (self.hour_index.shape != (1,t) or self.day_index.shape != (1,t)
                or self.minute_mask.shape != (1,t) or self.hour_mask.shape != (1,h)
                or self.day_mask.shape != (1,d)
                or self.minute_available_at.shape != (t,)
                or self.hour_available_at.shape != (h,)
                or self.day_available_at.shape != (d,)
                or self.hour_close_at.shape != (h,)
                or self.day_close_at.shape != (d,)):
            raise ValueError('hierarchy index, mask and availability shapes differ')
        if (self.hour_index.dtype.kind not in 'iu' or self.day_index.dtype.kind not in 'iu'
                or any(x.dtype.kind not in 'iu' for x in (
                    self.minute_available_at,self.hour_available_at,self.day_available_at,
                    self.hour_close_at,self.day_close_at))):
            raise ValueError('hierarchy indices and availability timestamps must be integers')
        if (any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1))
                for x in (self.minute_mask,self.hour_mask,self.day_mask))
                or (self.hour_index < -1).any() or (self.hour_index >= h).any()
                or (self.day_index < -1).any() or (self.day_index >= d).any()):
            raise ValueError('hierarchy masks or visible-state indices are invalid')
        if (np.any(np.diff(self.minute_available_at) < 0)
                or np.any(np.diff(self.hour_available_at) < 0)
                or np.any(np.diff(self.day_available_at) < 0)
                or np.any(np.diff(self.hour_close_at) <= 0)
                or np.any(np.diff(self.day_close_at) <= 0)):
            raise ValueError('hierarchy availability must be monotonic')
        if (np.any(self.hour_close_at <= 0) or np.any(self.day_close_at <= 0)
                or np.any(self.hour_available_at < self.hour_close_at)
                or np.any(self.day_available_at < self.day_close_at)):
            raise ValueError('slow-bar availability cannot precede its UTC close')
        for indices, availability, name in (
                (self.hour_index[0],self.hour_available_at,'hour'),
                (self.day_index[0],self.day_available_at,'day')):
            visible = indices >= 0
            if visible.any() and np.any(availability[indices[visible]]
                                         > self.minute_available_at[visible]):
                raise ValueError(f'{name} hierarchy state is not yet available')


@dataclass(frozen=True, slots=True)
class HierarchyWindowBatch:
    """Minute windows with causal absolute indices into the Train slow streams."""

    minute_x: np.ndarray       # [N,L,5]
    hour_index: np.ndarray     # [N,L]
    day_index: np.ndarray      # [N,L]
    minute_mask: np.ndarray    # [N,L]
    reset: np.ndarray          # [N,L]
    starts: np.ndarray         # [N]

    def __post_init__(self) -> None:
        if (self.minute_x.ndim != 3 or self.minute_x.shape[0] < 1
                or self.minute_x.shape[1] < 3 or self.minute_x.shape[2] != 5):
            raise ValueError('hierarchy windows require finite [N,L,5] features')
        n, length, _ = self.minute_x.shape
        if (not np.isfinite(self.minute_x).all()
                or self.hour_index.shape != (n,length)
                or self.day_index.shape != (n,length)
                or self.minute_mask.shape != (n,length)
                or self.reset.shape != (n,length)
                or self.starts.shape != (n,)
                or self.hour_index.dtype.kind not in 'iu' or self.day_index.dtype.kind not in 'iu'
                or self.starts.dtype.kind not in 'iu'
                or self.reset.dtype != np.dtype('bool')
                or np.any(~self.reset[:,0]) or np.any(self.reset[:,1:])
                or (self.hour_index < -1).any() or (self.day_index < -1).any()
                or not np.isfinite(self.minute_mask).all()
                or np.any((self.minute_mask < 0) | (self.minute_mask > 1))
                or np.any(self.minute_mask[:,1:].sum(axis=1) <= 0)):
            raise ValueError('invalid hierarchy window indices, masks or reset boundaries')


def _bucket_candles(buckets: tuple[ClosedBucket, ...]) -> tuple[Candle, ...]:
    return tuple(Candle(b.close_ns, b.available_at_ns, b.instrument,
                        b.open, b.high, b.low, b.close, b.volume) for b in buckets)


def _coords(bars: tuple[Candle, ...]) -> np.ndarray:
    if len(bars) < 2:
        raise ValueError('hierarchy requires at least two completed bars of each scale')
    # First completed slow bar has no prior full slow close inside this input;
    # use its own open as initial reference. It is only referenced AFTER close.
    return OHLCVCodec.encode(bars, previous_close=bars[0].open)


def build_train_hierarchy(replay: MarketSequence, *, burn_in: int,
                          score_start_ns: int, score_end_ns: int,
                          frozen_scalers: HierarchyScalers | None = None
                          ) -> tuple[HierarchyTensors, HierarchyScalers]:
    """Prepare B3 Train-only tensors, optionally reusing previously fitted Train scalers.

    The returned data are not a B3 trained model and do not authorize access to
    any independent Validation/Test/Holdout. Slow states after a minute
    prediction anchor are invisible via the explicit -1/bounded index mapping.
    """
    if replay.scope != 'train' or not replay.candles:
        raise ValueError('only project-approved in-memory Train replay accepted')
    bars = replay.candles
    if type(burn_in) is not int or burn_in < 1 or burn_in >= len(bars)-1:
        raise ValueError('invalid minute burn-in')
    if (type(score_start_ns) is not int or type(score_end_ns) is not int
            or score_start_ns >= score_end_ns
            or score_start_ns < bars[1].ts_close_ns
            or score_end_ns > bars[-1].ts_close_ns + MINUTE_NS):
        raise ValueError('Train score must be an explicit half-open interval inside the replay')
    clock = MultiScaleClock(bars)
    barrier = bars[-1].available_at_ns
    hourly = _bucket_candles(clock.buckets(60, as_of_ns=barrier))
    daily = _bucket_candles(clock.buckets(1440, as_of_ns=barrier))
    # Match B2 exactly: the first minute bar is the causal close anchor,
    # never a scored feature token. Every minute feature starts at bar[1].
    minute_coords = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    hour_coords = _coords(hourly)
    day_coords = _coords(daily)
    minute_score = np.asarray([
        score_start_ns <= bar.ts_close_ns < score_end_ns for bar in bars[1:]
    ], dtype=np.bool_)
    hour_score = np.asarray([
        score_start_ns <= bar.ts_close_ns - 59 * MINUTE_NS
        and bar.ts_close_ns < score_end_ns for bar in hourly
    ], dtype=np.bool_)
    day_score = np.asarray([
        score_start_ns <= bar.ts_close_ns - 1_439 * MINUTE_NS
        and bar.ts_close_ns < score_end_ns for bar in daily
    ], dtype=np.bool_)
    if (minute_score.sum() < 2 or hour_score.sum() < 2 or day_score.sum() < 2):
        raise ValueError('Train score has too few completed bars to fit every hierarchy scaler')
    if frozen_scalers is None:
        scalers = HierarchyScalers(*(
            TrainOnlyStandardizer.fit(x, dataset_id=replay.dataset_id, scope='train')
            for x in (minute_coords[minute_score], hour_coords[hour_score],
                      day_coords[day_score])
        ))
    else:
        scalers = frozen_scalers
        if any(x.dataset_id != replay.dataset_id or x.train_scope != 'train'
               for x in (scalers.minute, scalers.hour, scalers.day)):
            raise ValueError('slow and minute scalers must have the same Train dataset identity')
    plan = build_plan(bars)
    hour_idx = np.asarray(plan.hour_last_visible[1:], dtype=np.int32)
    day_idx = np.asarray(plan.day_last_visible[1:], dtype=np.int32)
    decisions = np.asarray([bar.available_at_ns for bar in bars[1:]], dtype=np.int64)
    for idx, visible_at, name in ((hour_idx, plan.hour_available_at, 'hour'),
                                  (day_idx, plan.day_available_at, 'day')):
        if (idx.shape != decisions.shape or (idx < -1).any()
            or (idx >= len(visible_at)).any() or (np.diff(idx) < 0).any()):
            raise ValueError(f'{name} slow index violates shape/monotonicity')
        selected = idx >= 0
        if selected.any() and np.any(visible_at[idx[selected]] > decisions[selected]):
            raise ValueError(f'{name} slow latent lookahead detected')
    mask = minute_score.astype(np.float32)
    mask[:burn_in] = 0.0
    return HierarchyTensors(
        scalers.minute.transform(minute_coords)[None],
        scalers.hour.transform(hour_coords)[None],
        scalers.day.transform(day_coords)[None],
        hour_idx[None], day_idx[None], mask[None],
        hour_score.astype(np.float32)[None],
        day_score.astype(np.float32)[None],
        decisions,
        np.asarray(plan.hour_available_at, dtype=np.int64),
        np.asarray(plan.day_available_at, dtype=np.int64),
        np.asarray([bar.ts_close_ns for bar in hourly], dtype=np.int64),
        np.asarray([bar.ts_close_ns for bar in daily], dtype=np.int64),
    ), scalers


def build_inference_hierarchy(replay: MarketSequence, *,
        scalers: HierarchyScalers, decision_ts_ns: int
        ) -> tuple[HierarchyTensors, tuple[Candle, ...]]:
    """Build one causal B3 context at an authorized Validation decision.

    The input sequence must already come from the registered Train+Validation
    reader. This function does not authorize access: it rejects Test/Holdout
    scopes, clips to bars both closed and available by ``decision_ts_ns``,
    uses a fixed 4,320-minute lookback with enough complete slow-scale history,
    and applies (but never fits) the frozen Train scalers.
    """
    if (not isinstance(replay, MarketSequence)
            or replay.scope not in {"validation", "inference_visible"}
            or not isinstance(scalers, HierarchyScalers)
            or type(decision_ts_ns) is not int or decision_ts_ns <= 0
            or decision_ts_ns % MINUTE_NS):
        raise ValueError("B3 inference requires an authorized visible sequence and aligned decision")
    if any(scaler.dataset_id != replay.dataset_id or scaler.train_scope != "train"
           for scaler in (scalers.minute, scalers.hour, scalers.day)):
        raise ValueError("B3 inference must reuse matching frozen Train scalers")

    visible = replay.as_of(decision_ts_ns)
    if not visible or visible[-1].ts_close_ns != decision_ts_ns:
        raise ValueError("authorized inference sequence must contain the decision-time candle")
    # Fixed length keeps every anchor on one JAX context shape. Starting three
    # complete UTC days back at the same minute-of-day leaves at least two full
    # completed daily buckets while allowing the current day to remain partial.
    context_bars = 3 * 1440
    context_start = decision_ts_ns - (context_bars - 1) * MINUTE_NS
    first_index = next((index for index, bar in enumerate(visible)
                        if bar.ts_close_ns >= context_start), None)
    if first_index is None or visible[first_index].ts_close_ns != context_start:
        raise ValueError("inference context lacks a complete UTC-aligned three-day prefix")
    bars = visible[first_index:]
    if len(bars) != context_bars:
        raise ValueError("B3 inference context requires the complete fixed 4,320-bar lookback")

    minute_coords = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    clock = MultiScaleClock(bars)
    hourly = _bucket_candles(clock.buckets(60, as_of_ns=decision_ts_ns))
    daily = _bucket_candles(clock.buckets(1440, as_of_ns=decision_ts_ns))
    if len(hourly) < 2 or len(daily) < 2:
        raise ValueError("B3 inference context needs two complete bars at every slow scale")
    hour_coords = _coords(hourly)
    day_coords = _coords(daily)
    plan = build_plan(bars)
    hour_index = np.asarray(plan.hour_last_visible[1:], dtype=np.int32)
    day_index = np.asarray(plan.day_last_visible[1:], dtype=np.int32)
    minute_available = np.asarray([bar.available_at_ns for bar in bars[1:]],
                                  dtype=np.int64)
    for indices, availability, name in (
            (hour_index, plan.hour_available_at, "hour"),
            (day_index, plan.day_available_at, "day")):
        if (indices.shape != minute_available.shape or (indices < -1).any()
                or (indices >= len(availability)).any()
                or (np.diff(indices) < 0).any()):
            raise ValueError(f"{name} inference visibility map is invalid")
        selected = indices >= 0
        if selected.any() and np.any(availability[indices[selected]]
                                     > minute_available[selected]):
            raise ValueError(f"{name} inference context contains a future slow bar")

    tensors = HierarchyTensors(
        scalers.minute.transform(minute_coords)[None],
        scalers.hour.transform(hour_coords)[None],
        scalers.day.transform(day_coords)[None],
        hour_index[None], day_index[None],
        np.zeros((1, len(minute_coords)), dtype=np.float32),
        np.zeros((1, len(hourly)), dtype=np.float32),
        np.zeros((1, len(daily)), dtype=np.float32),
        minute_available,
        np.asarray(plan.hour_available_at, dtype=np.int64),
        np.asarray(plan.day_available_at, dtype=np.int64),
        np.asarray([bar.ts_close_ns for bar in hourly], dtype=np.int64),
        np.asarray([bar.ts_close_ns for bar in daily], dtype=np.int64),
    )
    return tensors, bars


def make_hierarchy_windows(tensors: HierarchyTensors, *, length: int,
                           burn_in: int, stride: int) -> HierarchyWindowBatch:
    """Window only minute inputs; slow streams remain shared and causally indexed."""
    total = tensors.minute_x.shape[1]
    if (type(length) is not int or type(burn_in) is not int or type(stride) is not int
            or length < 3 or not 1 <= burn_in < length - 1 or stride < 1):
        raise ValueError('invalid hierarchy window length, burn-in or stride')
    if total < length:
        raise ValueError('Train hierarchy is shorter than one requested minute window')
    features: list[np.ndarray] = []
    hours: list[np.ndarray] = []
    days: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    starts: list[int] = []
    for start in range(0,total-length+1,stride):
        end = start + length
        mask = np.array(tensors.minute_mask[0,start:end],dtype=np.float32,copy=True)
        mask[:burn_in] = 0.0
        if not np.any(mask[1:]):
            continue
        features.append(tensors.minute_x[0,start:end])
        hours.append(tensors.hour_index[0,start:end])
        days.append(tensors.day_index[0,start:end])
        masks.append(mask)
        starts.append(start)
    if not starts:
        raise ValueError('no hierarchy windows contain scored minute targets')
    reset = np.zeros((len(starts),length),dtype=np.bool_)
    reset[:,0] = True
    return HierarchyWindowBatch(
        np.asarray(features,dtype=np.float32),
        np.asarray(hours,dtype=np.int32),
        np.asarray(days,dtype=np.int32),
        np.asarray(masks,dtype=np.float32),
        reset,
        np.asarray(starts,dtype=np.int64),
    )
