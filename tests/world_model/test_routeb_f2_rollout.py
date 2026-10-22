from __future__ import annotations

import pytest

from rles_world_model.routeb.market import Candle, MINUTE_NS
from rles_world_model.routeb.synthetic_feedback import (
    SyntheticFeedback, rollout_hierarchical_f2,
)


def _prefix(count: int) -> tuple[Candle, ...]:
    rows = []
    price = 100.0
    for minute in range(1, count + 1):
        opening = price
        price += 0.01
        rows.append(Candle(minute * MINUTE_NS, minute * MINUTE_NS,
                           "ETHUSDT-PERP.SIM", opening, price + 0.02,
                           opening - 0.02, price, 10.0))
    return tuple(rows)


def _runner(*, mode: str, prefix: tuple[Candle, ...], steps: int,
            minute_posterior=None):
    observations = []

    def minute_prior(state, hour, day, close_ns, seed):
        # Assertions prove a new hour posterior becomes visible on the next
        # minute and callback arguments contain no trade action/state.
        observations.append((state, hour, day, close_ns, seed))
        # Keep deterministic valid OHLCV and contiguous synthetic close times.
        last_close = (prefix[-1].close + 0.01 * (len(observations) - 1))
        close = last_close + 0.01
        bar = Candle(close_ns, close_ns, prefix[-1].instrument,
                     last_close, close + 0.01, last_close - 0.01, close, 10.0)
        return state + 1, bar

    def hour_posterior(state, bucket, day, seed):
        assert bucket.timeframe_minutes == 60
        return state + 1

    def day_posterior(state, bucket, seed):
        assert bucket.timeframe_minutes == 1440
        return state + 1

    result = rollout_hierarchical_f2(
        feedback=SyntheticFeedback(prefix, anchor_ns=prefix[-1].ts_close_ns),
        steps=steps, minute_state=0, hour_state=10, day_state=20,
        minute_prior=minute_prior, hour_posterior=hour_posterior,
        day_posterior=day_posterior, mode=mode,
        minute_posterior=minute_posterior, seed=31)
    return result, observations


def test_f2_updates_hour_state_only_after_complete_generated_hour() -> None:
    result, seen = _runner(mode="F2_SLOW_ONLY", prefix=_prefix(59), steps=2)
    assert result.mode == "F2_SLOW_ONLY"
    assert result.score_capability == "SCENARIO_ONLY"
    assert result.hour_updates == 1 and result.day_updates == 0
    assert seen[0][1] == 10
    assert seen[1][1] == 11
    assert result.minute_states == (1, 2)
    assert len(result.bars) == 2


def test_f2_full_filter_is_a_separate_minute_state_update_mode() -> None:
    filtered = []

    def minute_posterior(prior_state, synthetic, hour_state, day_state):
        filtered.append((synthetic.ts_close_ns, hour_state, day_state))
        return prior_state + 100

    result, _ = _runner(mode="F2_FULL_FILTER", prefix=_prefix(59), steps=2,
                        minute_posterior=minute_posterior)
    assert result.mode == "F2_FULL_FILTER"
    assert result.minute_states == (101, 202)
    assert filtered == [(60 * MINUTE_NS, 11, 20),
                        (61 * MINUTE_NS, 11, 20)]


def test_f2_rejects_incomplete_observed_prefix_at_hour_boundary() -> None:
    with pytest.raises(ValueError, match="needs sufficient observed prefix"):
        _runner(mode="F2_SLOW_ONLY", prefix=_prefix(58)[1:], steps=2)


def test_f2_slow_only_rejects_an_unrequested_minute_filter() -> None:
    with pytest.raises(ValueError, match="must not receive a minute posterior"):
        _runner(mode="F2_SLOW_ONLY", prefix=_prefix(59), steps=1,
                minute_posterior=lambda *args: None)


def test_f2_applies_same_time_day_update_before_hour_update() -> None:
    prefix = _prefix(1439)
    anchor = prefix[-1].ts_close_ns

    order = []

    def minute_prior(state, hour, day, close_ns, seed):
        last_close = prefix[-1].close
        bar = Candle(close_ns, close_ns, prefix[-1].instrument,
                     last_close, last_close + 0.01, last_close - 0.01,
                     last_close, 1)
        return state + 1, bar

    def day_posterior(state, bucket, seed):
        order.append("day")
        return state + 1

    def hour_posterior(state, bucket, day, seed):
        order.append(("hour", day))
        return state + 1

    result = rollout_hierarchical_f2(
        feedback=SyntheticFeedback(prefix, anchor_ns=anchor), steps=1,
        minute_state=0, hour_state=10,
        day_state=20, minute_prior=minute_prior, hour_posterior=hour_posterior,
        day_posterior=day_posterior, mode="F2_SLOW_ONLY", seed=1)
    assert order == ["day", ("hour", 21)]
    assert result.day_updates == 1 and result.hour_updates == 1
