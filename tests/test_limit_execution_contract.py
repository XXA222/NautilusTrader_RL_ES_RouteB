from decimal import Decimal

import pytest

from rles_gym_bridge.limit_book import (
    LimitEventKind,
    LimitOrderBook,
    LimitOrderState,
    PriceBar,
)
from rles_contracts.limit_execution import (
    LimitInstrumentRules,
    LimitOrderIntent,
    OrderIntentKind,
)


D = Decimal
RULES = LimitInstrumentRules("ETHUSDT-PERP.SIM", D("0.1"), D("0.01"), D("0.01"), D("5"))


def limit_order(*, order_id="order-1", qty="1", price="99", decision=100,
                effective=110, expiry=None):
    return LimitOrderIntent(
        intent_id=f"intent-{order_id}", instrument_id=RULES.instrument_id,
        kind=OrderIntentKind.LIMIT, signed_qty=D(qty), decision_ts_ns=decision,
        effective_ts_ns=effective, limit_price=D(price), pending_id=order_id,
        expiry_ts_ns=expiry,
    )


def cancel_order(*, order_id="order-1", decision=120, effective=130, cancel_id="cancel-1"):
    return LimitOrderIntent(
        intent_id=f"intent-{cancel_id}", instrument_id=RULES.instrument_id,
        kind=OrderIntentKind.CANCEL, signed_qty=D(0), decision_ts_ns=decision,
        effective_ts_ns=effective, pending_id=order_id, cancel_id=cancel_id,
    )


def bar(start, *, end=None, open="100", high="101", low="99", close="100"):
    return PriceBar(
        RULES.instrument_id, start, end if end is not None else start + 10,
        D(open), D(high), D(low), D(close),
    )


def test_l01_touch_before_effective_time_does_not_fill():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="100", effective=120))
    events = book.advance(bar(100, end=110, low="99"))
    assert not any(event.event_kind is LimitEventKind.FILLED for event in events)
    assert book.state("order-1") is LimitOrderState.PENDING


def test_l02_effective_next_bar_touch_fills_full_quantity_at_limit():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="99", effective=110))
    events = book.advance(bar(110, high="100", low="99"))
    fill = next(event for event in events if event.event_kind is LimitEventKind.FILLED)
    assert fill.signed_qty == D("1")
    assert fill.price == D("99")
    assert fill.fee_class == "SCENARIO_REQUIRED"
    assert book.state("order-1") is LimitOrderState.FILLED


def test_l03_gtc_order_survives_thirty_untouched_bars_without_economic_events():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="90", decision=90, effective=100))
    for i in range(30):
        book.advance(bar(100 + 10 * i, low="99", high="101"))
    assert book.open_orders == ("order-1",)
    assert not any(event.event_kind is LimitEventKind.FILLED for event in book.events)


def test_l04_cancel_effective_before_trigger_prevents_fill():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="99", effective=105))
    book.cancel(cancel_order(decision=105, effective=110))
    events = book.advance(bar(110, low="98"))
    assert any(event.event_kind is LimitEventKind.CANCELLED for event in events)
    assert not any(event.event_kind is LimitEventKind.FILLED for event in events)


def test_l05_fill_is_not_reversed_by_later_cancel():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="99", decision=90, effective=100))
    book.advance(bar(100, low="99"))
    book.cancel(cancel_order(decision=110, effective=120))
    assert book.state("order-1") is LimitOrderState.FILLED
    assert book.events[-1].event_kind is LimitEventKind.CANCEL_IGNORED_CLOSED


def test_l06_trigger_and_cancel_in_same_path_unknown_bar_is_ambiguous():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="99", decision=90, effective=100))
    book.cancel(cancel_order(decision=100, effective=105))
    events = book.advance(bar(100, end=110, low="98"))
    assert [event.event_kind for event in events] == [LimitEventKind.AMBIGUOUS_PATH]
    assert book.state("order-1") is LimitOrderState.AMBIGUOUS


def test_l07_marketable_limit_at_open_is_tagged_taker_and_uses_limit_price():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="99", decision=90, effective=100))
    events = book.advance(bar(100, open="98", high="100", low="97", close="99"))
    fill = next(event for event in events if event.event_kind is LimitEventKind.FILLED)
    assert fill.fee_class == "TAKER"
    assert fill.price == D("99")


def test_marketable_open_fill_precedes_a_later_intrabar_cancel():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="101", decision=90, effective=100))
    book.cancel(cancel_order(decision=110, effective=130))
    events = book.advance(bar(100, end=140, open="100", high="101", low="99"))
    fill = next(event for event in events if event.event_kind is LimitEventKind.FILLED)
    assert fill.ts_ns == 100
    assert fill.fee_class == "TAKER"
    assert book.state("order-1") is LimitOrderState.FILLED


@pytest.mark.parametrize(
    ("expiry", "cancel_ts", "expected", "expected_ts"),
    [
        (130, 150, LimitEventKind.EXPIRED, 130),
        (150, 130, LimitEventKind.CANCELLED, 130),
        (130, 130, LimitEventKind.EXPIRED, 130),
    ],
)
def test_earliest_intrabar_terminal_event_wins_without_a_trigger(
    expiry, cancel_ts, expected, expected_ts,
):
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="90", decision=90, effective=100, expiry=expiry))
    book.cancel(cancel_order(decision=cancel_ts - 10, effective=cancel_ts))
    events = book.advance(bar(100, end=200, low="99", high="101"))
    terminal = events[-1]
    assert terminal.event_kind is expected
    assert terminal.ts_ns == expected_ts


def test_intrabar_trigger_race_is_anchored_to_the_first_terminal_event():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="99", decision=90, effective=100, expiry=130))
    book.cancel(cancel_order(decision=140, effective=150))
    events = book.advance(bar(100, end=200, low="98"))
    assert events[-1].event_kind is LimitEventKind.AMBIGUOUS_PATH
    assert events[-1].ts_ns == 130
    assert "expiry" in events[-1].reason
    assert book.state("order-1") is LimitOrderState.AMBIGUOUS


def test_l08_unfilled_gtc_order_remains_pending_across_funding_barrier():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="90", decision=90, effective=100))
    book.advance(bar(100, end=200, low="99", high="101"))
    # The matching prototype has no account or funding ledger; an order alone
    # cannot create a fill or funding event. Native account accrual is still TBD.
    assert book.state("order-1") is LimitOrderState.PENDING
    assert all(event.event_kind is not LimitEventKind.FILLED for event in book.events)


def test_l09_episode_end_cancels_every_unfilled_order():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="90", decision=90, effective=100))
    events = book.terminate(200)
    assert len(events) == 1
    assert events[0].event_kind is LimitEventKind.TERMINAL_CANCELLED
    assert book.open_orders == ()


@pytest.mark.parametrize(
    ("price", "qty", "message"),
    [("99.05", "1", "price tick"), ("99", "1.005", "quantity step"),
     ("99", "0.04", "below instrument minimum")],
)
def test_l10_invalid_tick_step_or_minimum_is_rejected(price, qty, message):
    intent = limit_order(price=price, qty=qty)
    with pytest.raises(ValueError, match=message):
        LimitOrderBook(RULES).submit(intent)


def test_l10_quantity_below_instrument_minimum_is_rejected():
    rules = LimitInstrumentRules(RULES.instrument_id, D("0.1"), D("0.01"), D("0.02"), D("1"))
    with pytest.raises(ValueError, match="below minimum quantity"):
        LimitOrderBook(rules).submit(limit_order(price="99", qty="0.01"))


def test_activation_inside_triggered_bar_fails_closed_as_ambiguous():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="99", effective=105))
    events = book.advance(bar(100, end=110, low="98"))
    assert events[-1].event_kind is LimitEventKind.AMBIGUOUS_PATH
    assert book.state("order-1") is LimitOrderState.AMBIGUOUS


def test_explicit_expiry_before_next_bar_prevents_trigger():
    book = LimitOrderBook(RULES)
    book.submit(limit_order(price="99", decision=90, effective=100, expiry=120))
    events = book.advance(bar(120, low="98"))
    assert events[-1].event_kind is LimitEventKind.EXPIRED
    assert book.state("order-1") is LimitOrderState.EXPIRED
