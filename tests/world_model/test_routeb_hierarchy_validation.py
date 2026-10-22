from __future__ import annotations

import numpy as np
import pytest

from rles_world_model.routeb.hierarchy_dataset import (
    HierarchyScalers,
    build_train_hierarchy,
    build_validation_hierarchy,
)
from rles_world_model.routeb.market import Candle, MINUTE_NS, MarketSequence


def _sequence(*, start_close: int, count: int, scope: str, dataset: str = "r4-source") -> MarketSequence:
    rows = []
    previous = 100.0
    for index in range(count):
        close = previous + 0.001
        rows.append(Candle(
            start_close + index * MINUTE_NS,
            start_close + index * MINUTE_NS,
            "AVAXUSDT-PERP.BINANCE",
            previous, close + 0.01, previous - 0.01, close, 5.0,
        ))
        previous = close
    return MarketSequence(tuple(rows), dataset, scope)


def _train_scalers() -> tuple[HierarchyScalers, int]:
    train = _sequence(start_close=MINUTE_NS, count=3 * 1440 + 1, scope="train")
    _, scalers = build_train_hierarchy(train, burn_in=60)
    return scalers, train.candles[-1].ts_close_ns


def test_validation_hierarchy_reuses_train_scalers_and_masks_only_score_anchors() -> None:
    scalers, train_end = _train_scalers()
    validation = _sequence(
        start_close=train_end + MINUTE_NS, count=3 * 1440 + 1, scope="validation")
    score_start = validation.candles[1440].ts_close_ns
    score_end = score_start + 60 * MINUTE_NS
    tensors = build_validation_hierarchy(
        validation, scalers=scalers, score_start_ns=score_start, score_end_ns=score_end)

    assert tensors.dataset_id == validation.dataset_id
    assert tensors.scaler_fingerprints == (
        scalers.minute.fingerprint, scalers.hour.fingerprint, scalers.day.fingerprint)
    assert np.count_nonzero(tensors.minute_mask) == 60
    assert tensors.scope == "validation"
    assert (tensors.score_start_ns, tensors.score_end_ns) == (score_start, score_end)
    assert tensors.minute_mask[0, 1438] == 0
    assert tensors.minute_mask[0, 1439] == 1
    assert tensors.minute_mask[0, 1498] == 1
    assert tensors.minute_mask[0, 1499] == 0
    selected = tensors.hour_index[0] >= 0
    assert np.all(tensors.hour_available_at[tensors.hour_index[0, selected]]
                  <= tensors.minute_available_at[selected])
    selected = tensors.day_index[0] >= 0
    assert np.all(tensors.day_available_at[tensors.day_index[0, selected]]
                  <= tensors.minute_available_at[selected])


def test_validation_hierarchy_rejects_non_validation_and_wrong_dataset_scalers() -> None:
    scalers, train_end = _train_scalers()
    validation = _sequence(
        start_close=train_end + MINUTE_NS, count=3 * 1440 + 1, scope="validation")
    score_start = validation.candles[1440].ts_close_ns
    kwargs = dict(score_start_ns=score_start, score_end_ns=score_start + 60 * MINUTE_NS)
    with pytest.raises(ValueError, match="registered Validation"):
        build_validation_hierarchy(
            MarketSequence(validation.candles, validation.dataset_id, "test"),
            scalers=scalers, **kwargs)
    mismatched = HierarchyScalers(*(
        type(s).fit(np.ones((3, 5), dtype=np.float32),
                    dataset_id="other-source", scope="train")
        for s in (scalers.minute, scalers.hour, scalers.day)))
    with pytest.raises(ValueError, match="same dataset"):
        build_validation_hierarchy(validation, scalers=mismatched, **kwargs)


def test_validation_hierarchy_requires_context_before_the_score_interval() -> None:
    scalers, train_end = _train_scalers()
    validation = _sequence(
        start_close=train_end + MINUTE_NS, count=3 * 1440 + 1, scope="validation")
    score_start = validation.candles[0].ts_close_ns
    with pytest.raises(ValueError, match="cover warmup"):
        build_validation_hierarchy(
            validation, scalers=scalers, score_start_ns=score_start,
            score_end_ns=score_start + 60 * MINUTE_NS)


def test_train_hierarchy_excludes_read_only_warmup_from_scalers_and_losses() -> None:
    source = _sequence(start_close=MINUTE_NS, count=5 * 1440 + 1, scope="train")
    score_start = source.candles[2 * 1440].ts_close_ns
    bars = tuple(Candle(
        bar.ts_close_ns, bar.available_at_ns, bar.instrument,
        bar.open, bar.high, bar.low, bar.close,
        1000.0 if bar.ts_close_ns < score_start else 5.0)
        for bar in source.candles)
    replay = MarketSequence(bars, source.dataset_id, source.scope)
    tensors, scalers = build_train_hierarchy(
        replay, burn_in=60, score_start_ns=score_start,
        score_end_ns=bars[-1].ts_close_ns + MINUTE_NS)

    assert np.isclose(scalers.minute.mean[4], np.log1p(5.0))
    score_mask = tensors.minute_available_at >= score_start
    assert np.all(tensors.minute_mask[0, ~score_mask] == 0)
    assert np.count_nonzero(tensors.minute_mask) == np.count_nonzero(score_mask) - 60
    assert np.all(tensors.hour_mask[0, tensors.hour_available_at < score_start] == 0)
    assert np.all(tensors.day_mask[0, tensors.day_available_at < score_start] == 0)
