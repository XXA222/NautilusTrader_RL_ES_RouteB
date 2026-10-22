from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import math

import pytest

from rles_data.core import DataError, MINUTE_NS
from rles_world_model.contracts import MarketFrame
from rles_world_model.features import (
    DAY_FEATURE_NAMES,
    HOUR_FEATURE_NAMES,
    MINUTE_FEATURE_NAMES,
    MarketFeatureSpec,
    MarketFeatureVector,
    OHLCVTarget,
    TrainOnlyScaler,
    calendar_features,
    daily_features,
    decode_ohlcv_target,
    encode_ohlcv_target,
    hourly_features,
    minute_features,
)
from rles_world_model.temporal import AggregatedCandle, DAY_NS, HOUR_NS


def _minute(index: int) -> MarketFrame:
    timestamp = index * MINUTE_NS
    opening = 100 + index / 100
    close = opening + 0.02
    return MarketFrame(timestamp, timestamp, "ETHUSDT-PERP.SIM", opening,
                       close + 0.1, opening - 0.1, close, float(index))


def _aggregate(index: int, period_ns: int) -> AggregatedCandle:
    end = index * period_ns
    opening = 100 + index / 100
    close = opening + 0.02
    return AggregatedCandle(end - period_ns, end, end, "ETHUSDT-PERP.SIM",
                            opening, close + 0.1, opening - 0.1, close,
                            float(index), period_ns // MINUTE_NS)


def test_market_feature_schema_is_versioned_and_frozen() -> None:
    spec = MarketFeatureSpec()
    assert len(spec.hash) == 64
    assert spec.names("1m") == MINUTE_FEATURE_NAMES
    assert spec.names("1h") == HOUR_FEATURE_NAMES
    assert spec.names("1d") == DAY_FEATURE_NAMES
    with pytest.raises(DataError, match="windows are frozen"):
        MarketFeatureSpec(minute_windows=(3, 9))


def test_minute_features_are_finite_and_contain_only_named_market_inputs() -> None:
    vector = minute_features(tuple(_minute(index) for index in range(1, 17)))
    assert vector.names == MINUTE_FEATURE_NAMES
    assert len(vector.values) == 10
    assert all(math.isfinite(value) for value in vector.values)
    assert vector.as_dict()["log_upper_wick"] >= 0
    assert vector.as_dict()["log_lower_wick"] >= 0


def test_minute_features_reject_incomplete_warmup_and_gapped_history() -> None:
    with pytest.raises(DataError, match="15 prior bars"):
        minute_features(tuple(_minute(index) for index in range(1, 16)))
    gapped = tuple(_minute(index) for index in (*range(1, 15), 16, 17))
    with pytest.raises(DataError, match="contiguous"):
        minute_features(gapped)


def test_hourly_and_daily_features_require_complete_contiguous_aggregates() -> None:
    hours = tuple(_aggregate(index, HOUR_NS) for index in range(1, 26))
    days = tuple(_aggregate(index, DAY_NS) for index in range(1, 32))
    assert hourly_features(hours).names == HOUR_FEATURE_NAMES
    assert len(hourly_features(hours).values) == len(HOUR_FEATURE_NAMES)
    assert daily_features(days).names == DAY_FEATURE_NAMES
    assert len(daily_features(days).values) == len(DAY_FEATURE_NAMES)
    with pytest.raises(DataError, match="contiguous"):
        hourly_features((*hours[:-1], _aggregate(27, HOUR_NS)))


def test_calendar_features_use_utc_and_mark_only_real_utc_boundaries() -> None:
    ts = int(datetime(2026, 10, 18, 0, 0, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    vector = calendar_features(ts)
    values = vector.as_dict()
    assert vector.asof_ns == ts
    assert values["is_complete_hour"] == values["is_complete_day"] == 1.0
    assert values["utc_minute_sin"] == pytest.approx(0.0, abs=1e-12)
    assert values["utc_minute_cos"] == pytest.approx(1.0)
    after = calendar_features(ts + MINUTE_NS).as_dict()
    assert after["is_complete_hour"] == after["is_complete_day"] == 0.0


def test_ohlcv_target_roundtrips_and_tick_quantization_preserves_envelope() -> None:
    frame = MarketFrame(60 * MINUTE_NS, 60 * MINUTE_NS, "ETHUSDT-PERP.SIM",
                        101.0, 105.0, 98.0, 103.0, 12.0)
    target = encode_ohlcv_target(frame, previous_close=100.0)
    restored = decode_ohlcv_target(
        target, previous_close=100.0, ts_close_ns=frame.ts_close_ns,
        available_at_ns=frame.available_at_ns, instrument=frame.instrument,
    )
    assert restored.open == pytest.approx(frame.open)
    assert restored.high == pytest.approx(frame.high)
    assert restored.low == pytest.approx(frame.low)
    assert restored.close == pytest.approx(frame.close)
    assert restored.volume == pytest.approx(frame.volume)

    quantized = decode_ohlcv_target(
        target, previous_close=100.003, ts_close_ns=frame.ts_close_ns,
        available_at_ns=frame.available_at_ns, instrument=frame.instrument,
        tick_size=Decimal("0.01"),
    )
    assert all((Decimal(str(value)) / Decimal("0.01")) % 1 == 0
               for value in (quantized.open, quantized.high, quantized.low, quantized.close))
    assert quantized.low <= min(quantized.open, quantized.close)
    assert quantized.high >= max(quantized.open, quantized.close)


def test_zero_wick_and_zero_volume_target_is_legal() -> None:
    flat = MarketFrame(MINUTE_NS, MINUTE_NS, "ETHUSDT-PERP.SIM",
                       100.0, 100.0, 100.0, 100.0, 0.0)
    target = encode_ohlcv_target(flat, previous_close=100.0)
    assert target == OHLCVTarget(0.0, 0.0, 0.0, 0.0, 0.0)
    restored = decode_ohlcv_target(target, previous_close=100.0,
                                   ts_close_ns=MINUTE_NS, available_at_ns=MINUTE_NS,
                                   instrument=flat.instrument)
    assert (restored.open, restored.high, restored.low, restored.close, restored.volume) == (
        100.0, 100.0, 100.0, 100.0, 0.0,
    )


def test_scaler_is_train_score_only_and_records_clipped_values() -> None:
    vectors = tuple(MarketFeatureVector("1m", ts, ("x", "y"), (float(ts), float(ts % 7)))
                    for ts in (100, 200, 300))
    scaler = TrainOnlyScaler.fit(vectors, train_start_ns=100, train_end_ns=400)
    transformed = scaler.transform(MarketFeatureVector("1m", 400, ("x", "y"), (1e9, 3.0)))
    assert scaler.fit_count == 3
    assert scaler.fit_start_ns == 100 and scaler.fit_end_ns == 400
    assert transformed.clipped_count == 1
    assert max(abs(value) for value in transformed.values) <= scaler.clip_sigma
    with pytest.raises(DataError, match="inside Train score interval"):
        TrainOnlyScaler.fit((*vectors, MarketFeatureVector("1m", 400, ("x", "y"), (4, 4))),
                            train_start_ns=100, train_end_ns=400)
