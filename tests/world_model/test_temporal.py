from __future__ import annotations

from dataclasses import replace

import pytest

from rles_data.core import DataError, MINUTE_NS
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.temporal import (
    DAY_NS,
    HOUR_NS,
    CausalMarketProvider,
    TemporalScheduler,
    UtcCandleAggregator,
)


def _frame(index: int, *, delay_ns: int = 0, close: float | None = None) -> MarketFrame:
    ts = index * MINUTE_NS
    opening = 100.0 + index / 100
    closing = opening + 0.02 if close is None else close
    return MarketFrame(
        ts, ts + delay_ns, "ETHUSDT-PERP.SIM", opening,
        max(opening, closing) + 0.1, min(opening, closing) - 0.1,
        closing, float(index),
    )


def test_provider_obeys_availability_and_half_open_stream_windows() -> None:
    replay = MarketReplay((_frame(1), _frame(2, delay_ns=MINUTE_NS), _frame(3)), "a" * 64)
    provider = CausalMarketProvider(replay, feature_spec_hash="b" * 64)
    assert tuple(f.ts_close_ns for f in provider.stream(MINUTE_NS, 3 * MINUTE_NS)) == (
        MINUTE_NS, 2 * MINUTE_NS,
    )
    assert tuple(f.ts_close_ns for f in provider.visible_as_of(2 * MINUTE_NS)) == (MINUTE_NS,)
    assert provider.feature_spec_hash() == "b" * 64


def test_provider_rejects_duplicate_timestamps() -> None:
    replay = MarketReplay((_frame(1), _frame(1)), "a" * 64)
    with pytest.raises(DataError, match="duplicate or unordered"):
        CausalMarketProvider(replay, feature_spec_hash="b" * 64)


def test_hour_state_updates_only_after_all_sixty_utc_minute_bars() -> None:
    aggregator = UtcCandleAggregator()
    for index in range(1, 60):
        assert aggregator.push(_frame(index), decision_ns=index * MINUTE_NS).hour is None
    complete = aggregator.push(_frame(60), decision_ns=60 * MINUTE_NS)
    assert complete.hour is not None
    assert complete.hour.start_ns == 0
    assert complete.hour.end_ns == HOUR_NS
    assert complete.hour.minute_count == 60
    assert complete.hour.open == _frame(1).open
    assert complete.hour.close == _frame(60).close
    assert complete.hour.volume == sum(float(index) for index in range(1, 61))


def test_day_state_updates_only_after_all_1440_utc_minute_bars() -> None:
    aggregator = UtcCandleAggregator()
    for index in range(1, 1440):
        completed = aggregator.push(_frame(index), decision_ns=index * MINUTE_NS)
        assert completed.day is None
    completed = aggregator.push(_frame(1440), decision_ns=DAY_NS)
    assert completed.hour is not None
    assert completed.day is not None
    assert completed.day.start_ns == 0
    assert completed.day.end_ns == DAY_NS
    assert completed.day.minute_count == 1440


def test_gap_fails_closed_and_restart_requires_explicit_contiguous_warmup() -> None:
    aggregator = UtcCandleAggregator()
    aggregator.push(_frame(1), decision_ns=MINUTE_NS)
    with pytest.raises(DataError, match="missing minute"):
        aggregator.push(_frame(3), decision_ns=3 * MINUTE_NS)
    with pytest.raises(DataError, match="configured amount of warmup"):
        aggregator.reset_with_warmup((_frame(4),), decision_ns=4 * MINUTE_NS,
                                     min_warmup_minutes=2)
    aggregator.reset_with_warmup((_frame(4), _frame(5)), decision_ns=5 * MINUTE_NS,
                                 min_warmup_minutes=2)
    assert aggregator.last_close_ns == 5 * MINUTE_NS


def test_rejected_warmup_reset_preserves_existing_aggregator_state() -> None:
    aggregator = UtcCandleAggregator()
    original = _frame(1)
    aggregator.push(original, decision_ns=MINUTE_NS)
    with pytest.raises(DataError, match="missing minute"):
        aggregator.reset_with_warmup(
            (_frame(3), _frame(5)), decision_ns=5 * MINUTE_NS,
            min_warmup_minutes=2,
        )
    assert aggregator.last_close_ns == original.ts_close_ns


def test_late_available_bar_cannot_advance_aggregator_or_slow_state() -> None:
    aggregator = UtcCandleAggregator()
    delayed = _frame(1, delay_ns=1)
    with pytest.raises(DataError, match="not available"):
        aggregator.push(delayed, decision_ns=MINUTE_NS)
    assert aggregator.last_close_ns is None


def test_scheduler_forms_prior_before_observation_and_updates_slow_state_after() -> None:
    scheduler = TemporalScheduler(0)
    observed_prior_inputs: list[tuple[int, int]] = []

    def predict(state: int, close_ns: int) -> tuple[int, int]:
        observed_prior_inputs.append((state, close_ns))
        return (state, close_ns)

    def observe(state: int, _frame: MarketFrame) -> int:
        return state + 1

    step = None
    for index in range(1, 61):
        step = scheduler.advance_one_minute(
            _frame(index), decision_ns=index * MINUTE_NS,
            predict_prior=predict, observe=observe,
            update_hour=lambda state, _bar: state + 1000,
            update_day=lambda state, _bar: state + 10000,
        )
    assert step is not None
    assert observed_prior_inputs[-1] == (59, HOUR_NS)
    assert step.prior == (59, HOUR_NS)
    assert step.state == 1060
    assert step.completed.hour is not None


def test_daily_and_hourly_states_update_together_only_after_midnight_bar() -> None:
    scheduler = TemporalScheduler(0)
    update_order: list[str] = []
    step = None
    for index in range(1, 1441):
        step = scheduler.advance_one_minute(
            _frame(index), decision_ns=index * MINUTE_NS,
            predict_prior=lambda state, _ts: state,
            observe=lambda state, _bar: state,
            update_hour=lambda state, _bar: update_order.append("hour") or state + 1,
            update_day=lambda state, _bar: update_order.append("day") or state + 10,
        )
    assert step is not None
    assert step.prior == 23
    assert step.state == 34
    assert step.completed.hour is not None and step.completed.day is not None
    assert update_order[-2:] == ["day", "hour"]


def test_scheduler_prior_does_not_depend_on_current_or_future_bar_values() -> None:
    initial = _frame(1)
    changed = replace(initial, open=500.0, high=501.0, low=499.0, close=500.0)

    def run(frame: MarketFrame) -> object:
        scheduler = TemporalScheduler(("history",))
        result = scheduler.advance_one_minute(
            frame, decision_ns=frame.available_at_ns,
            predict_prior=lambda state, close_ns: (state, close_ns),
            observe=lambda state, current: state + (current.close,),
            update_hour=lambda state, _bar: state,
            update_day=lambda state, _bar: state,
        )
        return result.prior

    assert run(initial) == run(changed)
