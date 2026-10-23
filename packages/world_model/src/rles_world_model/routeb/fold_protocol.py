"""Shared validation and timestamp rules for registered Route B Fold runs."""
from __future__ import annotations

import numpy as np

from rles_data.core import MINUTE_NS, FoldManifest


def validate_fold_experiment_protocol(*, horizons: tuple[int, ...],
        anchor_interval_minutes: int, context_lookback: int,
        volatility_window: int, block_length: int,
        bootstrap_replicates: int, seed: int) -> None:
    """Reject an invalid evaluation protocol before a fresh Fold is consumed."""
    if (not horizons or tuple(sorted(set(horizons))) != horizons
            or any(type(horizon) is not int or not 1 <= horizon <= 60
                   for horizon in horizons)
            or type(anchor_interval_minutes) is not int or anchor_interval_minutes < 1
            or type(context_lookback) is not int or context_lookback < 3
            or type(volatility_window) is not int or volatility_window < 2
            or volatility_window > horizons[-1]
            or type(block_length) is not int or block_length < 1
            or type(bootstrap_replicates) is not int or bootstrap_replicates < 1
            or type(seed) is not int or seed < 0):
        raise ValueError(
            "invalid fold evaluation protocol: horizons must be ordered in [1, 60] "
            "and volatility_window must not exceed the maximum horizon"
        )


def validation_anchor_timestamps(*, fold: FoldManifest, max_horizon: int,
        anchor_interval_minutes: int) -> np.ndarray:
    """Return common anchors whose complete future is inside Validation score.

    ``score_end_ns`` is exclusive. A forecast anchored at ``t`` has targets at
    ``t + 1m`` through ``t + Hm``, so its final target must be strictly before
    the Validation end boundary.
    """
    if (not isinstance(fold, FoldManifest)
            or type(max_horizon) is not int or not 1 <= max_horizon <= 60
            or type(anchor_interval_minutes) is not int or anchor_interval_minutes < 1):
        raise ValueError("invalid Fold anchor protocol")
    first_anchor = fold.validation.score_start_ns - MINUTE_NS
    last_anchor = fold.validation.score_end_ns - (max_horizon + 1) * MINUTE_NS
    stride = anchor_interval_minutes * MINUTE_NS
    if last_anchor < first_anchor:
        return np.empty(0, dtype=np.int64)
    return np.arange(first_anchor, last_anchor + 1, stride, dtype=np.int64)
