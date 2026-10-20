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
from .market import Candle, MarketSequence


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
    clock = MultiScaleClock(bars)
    barrier = bars[-1].available_at_ns
    hourly = _bucket_candles(clock.buckets(60, as_of_ns=barrier))
    daily = _bucket_candles(clock.buckets(1440, as_of_ns=barrier))
    # Match B2 exactly: the first minute bar is the causal close anchor,
    # never a scored feature token. Every minute feature starts at bar[1].
    minute_coords = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    hour_coords = _coords(hourly)
    day_coords = _coords(daily)
    if frozen_scalers is None:
        scalers = HierarchyScalers(*(
            TrainOnlyStandardizer.fit(x, dataset_id=replay.dataset_id, scope='train')
            for x in (minute_coords, hour_coords, day_coords)
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
    mask = np.ones(len(bars)-1, dtype=np.float32)
    mask[:burn_in] = 0.0
    return HierarchyTensors(
        scalers.minute.transform(minute_coords)[None],
        scalers.hour.transform(hour_coords)[None],
        scalers.day.transform(day_coords)[None],
        hour_idx[None], day_idx[None], mask[None],
        np.ones((1, len(hourly)), dtype=np.float32),
        np.ones((1, len(daily)), dtype=np.float32),
        decisions,
        np.asarray(plan.hour_available_at, dtype=np.int64),
        np.asarray(plan.day_available_at, dtype=np.int64),
    ), scalers
