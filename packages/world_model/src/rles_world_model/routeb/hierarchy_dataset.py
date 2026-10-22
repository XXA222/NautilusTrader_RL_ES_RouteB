"""Train-only causal tensor projection for the experimental B3 three-scale learner.

This module prepares ONE contiguous approved Train sequence. It intentionally
is not a general sampler, model-selection service, or authorization bypass.
Raw 1m candles are the sole source of 1h/1d candles. A slow latent is referenced
only after its final constituent 1m candle becomes available.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .clock import ClosedBucket, MultiScaleClock
from .codec import OHLCVCodec, TrainOnlyStandardizer
from .hierarchy import build_plan
from .market import MINUTE_NS, Candle, MarketSequence
from .risk_targets import estimate_train_barrier_fractions


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
    hour_day_index: np.ndarray | None = None  # [1,H], latest day actually available to each hour
    dataset_id: str | None = None
    scaler_fingerprints: tuple[str, str, str] | None = None
    scope: str = "train"
    score_start_ns: int | None = None
    score_end_ns: int | None = None
    risk_targets: np.ndarray | None = None  # [1,T,H,6], Train-only labels
    risk_masks: np.ndarray | None = None    # [1,T,H], complete in-scope paths
    risk_horizons: tuple[int, ...] = (1, 5, 15, 30, 60)
    risk_barrier_fractions: tuple[float, ...] | None = None
    risk_barrier_quantile: float = 0.75


def _train_risk_targets(bars: tuple[Candle, ...], *, score_start_ns: int,
                        score_end_ns: int, horizons: tuple[int, ...],
                        volatility_quantile: float = 0.75
                        ) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Build E2-compatible risk labels using Train rows and Train-fitted barriers."""
    if not horizons or tuple(sorted(set(horizons))) != horizons or any(
            type(value) is not int or value < 1 for value in horizons):
        raise ValueError("risk horizons must be strictly increasing positive integers")
    scored = [bar for bar in bars if score_start_ns <= bar.ts_close_ns < score_end_ns]
    if len(scored) < 2:
        raise ValueError("Train risk barrier fitting requires at least two score bars")
    barriers = estimate_train_barrier_fractions(
        np.asarray([bar.close for bar in scored], dtype=np.float64),
        train_timestamps_ns=np.asarray([bar.ts_close_ns for bar in scored], dtype=np.int64),
        horizons=horizons, volatility_quantile=volatility_quantile)
    timestamps = np.asarray([bar.ts_close_ns for bar in bars], dtype=np.int64)
    closes = np.asarray([bar.close for bar in bars], dtype=np.float64)
    log_close = np.log(closes)
    log_high = np.log(np.asarray([bar.high for bar in bars], dtype=np.float64))
    log_low = np.log(np.asarray([bar.low for bar in bars], dtype=np.float64))
    returns = np.zeros(len(bars), dtype=np.float64)
    returns[1:] = np.diff(log_close)
    prefix_return = np.concatenate(([0.0], np.cumsum(returns)))
    prefix_square = np.concatenate(([0.0], np.cumsum(np.square(returns))))
    # Feature token row r is candle r+1. Its future horizon starts at candle r+2.
    token_count = len(bars) - 1
    targets = np.zeros((1, token_count, len(horizons), 6), dtype=np.float32)
    masks = np.zeros((1, token_count, len(horizons)), dtype=np.float32)
    scored_anchor = ((timestamps[1:] >= score_start_ns)
                     & (timestamps[1:] < score_end_ns))
    for h_index, horizon in enumerate(horizons):
        anchor_indices = np.arange(1, len(bars) - horizon, dtype=np.int64)
        if not anchor_indices.size:
            continue
        rows = anchor_indices - 1
        valid = (scored_anchor[rows]
                 & (timestamps[anchor_indices + horizon] < score_end_ns))
        if not np.any(valid):
            continue
        anchor_indices = anchor_indices[valid]
        rows = rows[valid]
        total_return = (prefix_return[anchor_indices + horizon + 1]
                        - prefix_return[anchor_indices + 1])
        realized_volatility = np.sqrt(
            prefix_square[anchor_indices + horizon + 1]
            - prefix_square[anchor_indices + 1])
        future_highs = sliding_window_view(log_high[2:], horizon)[:len(scored_anchor)]
        future_lows = sliding_window_view(log_low[2:], horizon)[:len(scored_anchor)]
        max_high = np.max(future_highs[rows], axis=1) - log_close[anchor_indices]
        min_low = np.min(future_lows[rows], axis=1) - log_close[anchor_indices]
        barrier = float(barriers[h_index])
        targets[0, rows, h_index, 0] = total_return
        targets[0, rows, h_index, 1] = np.log(realized_volatility + 1e-8)
        targets[0, rows, h_index, 2] = max_high
        targets[0, rows, h_index, 3] = min_low
        targets[0, rows, h_index, 4] = max_high >= np.log1p(barrier)
        targets[0, rows, h_index, 5] = min_low <= np.log1p(-barrier)
        masks[0, rows, h_index] = 1.0
    return targets, masks, tuple(float(value) for value in barriers)


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
                          frozen_scalers: HierarchyScalers | None = None,
                          score_start_ns: int | None = None,
                          score_end_ns: int | None = None,
                          risk_horizons: tuple[int, ...] = (1, 5, 15, 30, 60),
                          risk_barrier_quantile: float = 0.75
                          ) -> tuple[HierarchyTensors, HierarchyScalers]:
    """Prepare Train tensors, fitting scalers and losses only on score rows.

    The returned data are not a B3 trained model and do not authorize access to
    any independent Validation/Test/Holdout. Slow states after a minute
    prediction anchor are invisible via the explicit -1/bounded index mapping.
    Optional leading replay rows are read-only causal warmup and are excluded
    from scaler fitting and all scale-specific loss masks.
    """
    if replay.scope != 'train' or not replay.candles:
        raise ValueError('only project-approved in-memory Train replay accepted')
    bars = replay.candles
    if type(burn_in) is not int or burn_in < 1 or burn_in >= len(bars)-1:
        raise ValueError('invalid minute burn-in')
    score_start_ns = bars[1].ts_close_ns if score_start_ns is None else score_start_ns
    score_end_ns = bars[-1].ts_close_ns + MINUTE_NS if score_end_ns is None else score_end_ns
    if (type(score_start_ns) is not int or type(score_end_ns) is not int
            or score_start_ns % MINUTE_NS or score_end_ns % MINUTE_NS
            or score_start_ns >= score_end_ns
            or score_start_ns < bars[1].ts_close_ns
            or score_end_ns > bars[-1].ts_close_ns + MINUTE_NS):
        raise ValueError('Train score interval is outside the supplied causal replay')
    clock = MultiScaleClock(bars)
    barrier = bars[-1].available_at_ns
    hourly = _bucket_candles(clock.buckets(60, as_of_ns=barrier))
    daily = _bucket_candles(clock.buckets(1440, as_of_ns=barrier))
    # Match B2 exactly: the first minute bar is the causal close anchor,
    # never a scored feature token. Every minute feature starts at bar[1].
    minute_coords = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    hour_coords = _coords(hourly)
    day_coords = _coords(daily)
    minute_score = np.asarray(
        [score_start_ns <= bar.ts_close_ns < score_end_ns for bar in bars[1:]],
        dtype=bool)
    hour_score = np.asarray(
        [score_start_ns <= bar.ts_close_ns < score_end_ns for bar in hourly], dtype=bool)
    day_score = np.asarray(
        [score_start_ns <= bar.ts_close_ns < score_end_ns for bar in daily], dtype=bool)
    if (minute_score.sum() <= burn_in or hour_score.sum() < 2 or day_score.sum() < 2):
        raise ValueError('Train score interval has insufficient minute/hour/day observations')
    if frozen_scalers is None:
        scalers = HierarchyScalers(*(
            TrainOnlyStandardizer.fit(x[mask], dataset_id=replay.dataset_id, scope='train')
            for x, mask in ((minute_coords, minute_score), (hour_coords, hour_score),
                            (day_coords, day_score))
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
    hour_day_idx = np.searchsorted(
        plan.day_available_at, plan.hour_available_at, side="right") - 1
    hour_day_idx = np.asarray(hour_day_idx, dtype=np.int32)
    selected_hour_day = hour_day_idx >= 0
    if (np.any(hour_day_idx < -1) or np.any(hour_day_idx >= len(plan.day_available_at))
            or (selected_hour_day.any() and np.any(
                plan.day_available_at[hour_day_idx[selected_hour_day]] >
                plan.hour_available_at[selected_hour_day]))):
        raise ValueError("hour-to-day slow index violates actual availability ordering")
    for idx, visible_at, name in ((hour_idx, plan.hour_available_at, 'hour'),
                                  (day_idx, plan.day_available_at, 'day')):
        if (idx.shape != decisions.shape or (idx < -1).any()
            or (idx >= len(visible_at)).any() or (np.diff(idx) < 0).any()):
            raise ValueError(f'{name} slow index violates shape/monotonicity')
        selected = idx >= 0
        if selected.any() and np.any(visible_at[idx[selected]] > decisions[selected]):
            raise ValueError(f'{name} slow latent lookahead detected')
    mask = minute_score.astype(np.float32)
    first_score = int(np.flatnonzero(minute_score)[0])
    mask[first_score:first_score + burn_in] = 0.0
    risk_targets, risk_masks, barriers = _train_risk_targets(
        bars, score_start_ns=score_start_ns, score_end_ns=score_end_ns,
        horizons=risk_horizons, volatility_quantile=risk_barrier_quantile)
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
        hour_day_idx[None],
        replay.dataset_id,
        (scalers.minute.fingerprint, scalers.hour.fingerprint,
         scalers.day.fingerprint),
        "train", score_start_ns, score_end_ns,
        risk_targets, risk_masks, risk_horizons, barriers, risk_barrier_quantile,
    ), scalers


def build_validation_hierarchy(replay: MarketSequence, *, scalers: HierarchyScalers,
                               score_start_ns: int, score_end_ns: int
                               ) -> HierarchyTensors:
    """Project a separately authorized Validation sequence with frozen Train scalers.

    This path never fits statistics. It marks Validation-score decision bars in
    ``minute_mask`` and preserves actual close/availability indexes for each
    slow state. The caller must pass only the registered Validation context and
    score window; Test/Holdout sequences are rejected.
    """
    if replay.scope != "validation" or not replay.candles:
        raise ValueError("only registered Validation replay is accepted")
    if (type(score_start_ns) is not int or type(score_end_ns) is not int
            or score_start_ns <= 0 or score_start_ns >= score_end_ns
            or score_start_ns % MINUTE_NS or score_end_ns % MINUTE_NS):
        raise ValueError("invalid Validation score interval")
    if any(s.dataset_id != replay.dataset_id or s.train_scope != "train"
           for s in (scalers.minute, scalers.hour, scalers.day)):
        raise ValueError("Validation must use frozen Train scalers from the same dataset")
    bars = replay.candles
    if bars[0].ts_close_ns >= score_start_ns or bars[-1].ts_close_ns < score_end_ns - MINUTE_NS:
        raise ValueError("Validation sequence does not cover warmup and score interval")

    clock = MultiScaleClock(bars)
    barrier = bars[-1].available_at_ns
    hourly = _bucket_candles(clock.buckets(60, as_of_ns=barrier))
    daily = _bucket_candles(clock.buckets(1440, as_of_ns=barrier))
    minute_coords = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    hour_coords = _coords(hourly)
    day_coords = _coords(daily)
    plan = build_plan(bars)
    hour_idx = np.asarray(plan.hour_last_visible[1:], dtype=np.int32)
    day_idx = np.asarray(plan.day_last_visible[1:], dtype=np.int32)
    decisions = np.asarray([bar.available_at_ns for bar in bars[1:]], dtype=np.int64)
    hour_day_idx = np.searchsorted(
        plan.day_available_at, plan.hour_available_at, side="right") - 1
    hour_day_idx = np.asarray(hour_day_idx, dtype=np.int32)
    for idx, visible_at, name in ((hour_idx, plan.hour_available_at, "hour"),
                                  (day_idx, plan.day_available_at, "day")):
        if (idx.shape != decisions.shape or np.any(idx < -1)
                or np.any(idx >= len(visible_at)) or np.any(np.diff(idx) < 0)):
            raise ValueError(f"Validation {name} index violates shape/monotonicity")
        selected = idx >= 0
        if selected.any() and np.any(visible_at[idx[selected]] > decisions[selected]):
            raise ValueError(f"Validation {name} index leaks a future slow state")
    selected = hour_day_idx >= 0
    if (np.any(hour_day_idx < -1) or np.any(hour_day_idx >= len(plan.day_available_at))
            or (selected.any() and np.any(
                plan.day_available_at[hour_day_idx[selected]] > plan.hour_available_at[selected]))):
        raise ValueError("Validation hour-to-day index violates availability ordering")

    score_mask = np.asarray(
        [score_start_ns <= bar.ts_close_ns < score_end_ns for bar in bars[1:]],
        dtype=np.float32)
    if np.count_nonzero(score_mask) < 2:
        raise ValueError("Validation score interval has insufficient minute anchors")
    fingerprints = (scalers.minute.fingerprint, scalers.hour.fingerprint,
                    scalers.day.fingerprint)
    return HierarchyTensors(
        scalers.minute.transform(minute_coords)[None],
        scalers.hour.transform(hour_coords)[None],
        scalers.day.transform(day_coords)[None],
        hour_idx[None], day_idx[None], score_mask[None],
        np.ones((1, len(hourly)), dtype=np.float32),
        np.ones((1, len(daily)), dtype=np.float32),
        decisions, np.asarray(plan.hour_available_at, dtype=np.int64),
        np.asarray(plan.day_available_at, dtype=np.int64), hour_day_idx[None],
        replay.dataset_id, fingerprints, "validation", score_start_ns, score_end_ns)
