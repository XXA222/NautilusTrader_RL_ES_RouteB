from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from nautilus_trader.model import (
    AccountId,
    AccountType,
    Bar,
    BarType,
    Currency,
    CurrencyPair,
    InstrumentId,
    Money,
    OmsType,
    Price,
    Quantity,
    StandardMarginModel,
    Symbol,
    Venue,
)
from nautilus_trader.execution import DefaultFillModel, MakerTakerFeeModel, StaticLatencyModel
from nautilus_trader.trading import Strategy, StrategyConfig

from rles_contracts.limit_execution import (
    LimitInstrumentRules,
    LimitOrderIntent,
    OrderIntentKind,
)
from rles_contracts.dto import (
    ExecutionPlan,
    OrderType,
    PendingOrderPolicy,
    TimeInForce,
)
from rles_gym_bridge.limit_book import LimitOrderBook, LimitOrderState, PriceBar
from rles_nautilus_adapter.bootstrap import NativeBootConfiguration, PublicBacktestBuilder
from rles_nautilus_adapter.limit_submitter import NativeLimitSubmitter
from rles_nautilus_adapter.session import EngineBoundaryError, EngineSession, NativeFactsReader


pytestmark = pytest.mark.micro_golden


def _instrument() -> CurrencyPair:
    return CurrencyPair(
        instrument_id=InstrumentId.from_str("BTC/USD.SIM"),
        raw_symbol=Symbol("BTC/USD"), base_currency=Currency.from_str("BTC"),
        quote_currency=Currency.from_str("USD"), price_precision=2, size_precision=4,
        price_increment=Price.from_str("0.01"), size_increment=Quantity.from_str("0.0001"),
        ts_event=1, ts_init=1,
    )


class _Strategy(Strategy):
    def __init__(self) -> None:
        super().__init__(StrategyConfig())


def _batch(number: int, *, open_price: float = 100, high: float = 100,
           low: float = 100, close: float = 100):
    instrument = _instrument()
    ts = number * 60_000_000_000
    bar = Bar(
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
        open=instrument.make_price(open_price), high=instrument.make_price(high),
        low=instrument.make_price(low), close=instrument.make_price(close),
        volume=instrument.make_qty(100), ts_event=ts, ts_init=ts,
    )
    event = SimpleNamespace(
        event_id=f"native-limit-bar-{number}", source_hash="a" * 64,
        ts_event_ns=ts, ts_init_ns=ts, available_at_ns=ts, native_bar=bar,
    )
    from rles_nautilus_adapter.session import VisibleBatch

    return VisibleBatch(ts, (event,), True)


def _fixture():
    instrument = _instrument()
    rules = LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"),
        min_notional=Decimal("1"),
    )
    submitter = NativeLimitSubmitter(rules=rules, quantity_unit="BTC")
    builder = PublicBacktestBuilder(
        plan_submitter=submitter,
        config=NativeBootConfiguration(
            venue_kwargs=({
                "venue": Venue("SIM"), "oms_type": OmsType.NETTING,
                "account_type": AccountType.MARGIN,
                "starting_balances": [Money(100_000, Currency.from_str("USD"))],
                "base_currency": Currency.from_str("USD"),
                "margin_model": StandardMarginModel(),
                "fee_model": MakerTakerFeeModel(Decimal("0.001"), Decimal("0.002")),
                "fill_model": DefaultFillModel(prob_fill_on_limit=1.0, prob_slippage=0.0,
                                               random_seed=0),
                "latency_model": StaticLatencyModel(),
                "bar_execution": True, "liquidation_enabled": False,
            },),
            instrument_builders=(_instrument,), strategy_builder=_Strategy,
        ),
    )
    session = EngineSession(
        build_native=builder,
        submit_plan=lambda _engine, _plan: None,
        decode_event=lambda event: event.native_bar,
        facts_reader=NativeFactsReader(account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
                                       currency="USD"),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
        submit_limit_intent=lambda engine, intent, ts: submitter(
            engine, intent, decision_ts_ns=ts,
        ),
        cancel_limit_orders=submitter.cancel_open_orders,
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    return session, submitter, instrument


def test_native_limit_submit_and_cancel_follow_native_order_events() -> None:
    session, submitter, instrument = _fixture()
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        order = LimitOrderIntent(
            intent_id="limit-1", instrument_id=str(instrument.id), kind=OrderIntentKind.LIMIT,
            signed_qty=Decimal("1.0000"), decision_ts_ns=60_000_000_000,
            effective_ts_ns=60_000_000_001, limit_price=Decimal("50.00"), pending_id="pending-1",
        )
        pending = session.advance_limit(order, batches=(_batch(2),),
                                        next_decision_ts_ns=120_000_000_000)
        assert len(pending.orders) == 1
        assert "LIMIT" in str(pending.orders[0]), pending.orders[0]
        assert "50.00" in str(pending.orders[0]), pending.orders[0]
        assert pending.positions == pending.fills == ()

        cancel = LimitOrderIntent(
            intent_id="cancel-1", instrument_id=str(instrument.id), kind=OrderIntentKind.CANCEL,
            signed_qty=Decimal(0), decision_ts_ns=120_000_000_000,
            effective_ts_ns=180_000_000_000, pending_id="pending-1", cancel_id="cancel-event-1",
        )
        closed = session.advance_limit(cancel, batches=(_batch(3),),
                                       next_decision_ts_ns=180_000_000_000)
        assert closed.orders == closed.positions == closed.fills == ()
        assert closed.all_orders[0]["status"] == "CANCELED"
        assert closed.equity == Decimal("100000.00")
    finally:
        session.close()


def test_native_gtd_expiry_wins_equal_timestamp_touch_like_ohlc_contract() -> None:
    session, submitter, instrument = _fixture()
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        intent = LimitOrderIntent(
            intent_id="expiring", instrument_id=str(instrument.id), kind=OrderIntentKind.LIMIT,
            signed_qty=Decimal("1.0000"), decision_ts_ns=60_000_000_000,
            effective_ts_ns=60_000_000_001, limit_price=Decimal("99.00"), pending_id="pending-expire",
            expiry_ts_ns=180_000_000_000,
        )
        book = LimitOrderBook(LimitInstrumentRules(
            instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
            qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"),
            min_notional=Decimal("1"),
        ))
        book.submit(intent)
        expected = book.advance(PriceBar(
            str(instrument.id), 120_000_000_000, 180_000_000_000,
            Decimal("100.00"), Decimal("101.00"), Decimal("100.00"), Decimal("100.00"),
        ))
        expected += book.advance(PriceBar(
            str(instrument.id), 180_000_000_000, 240_000_000_000,
            Decimal("100.00"), Decimal("101.00"), Decimal("99.00"), Decimal("100.00"),
        ))
        result = session.advance_limit(
            intent,
            batches=(_batch(2), _batch(3, open_price=100, high=101, low=99, close=100)),
            next_decision_ts_ns=180_000_000_000,
        )
        assert expected[-1].event_kind.value == "EXPIRED"
        assert result.fills == result.positions == ()
        assert result.all_orders[0]["status"] == "EXPIRED"
        assert result.all_orders[0]["expire_time_ns"] == 180_000_000_000
    finally:
        if session.state != "CLOSED":
            session.close()


def test_native_gtc_survives_thirty_untouched_bars_without_economic_events() -> None:
    session, _submitter, instrument = _fixture()
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        intent = LimitOrderIntent(
            intent_id="native-l03", instrument_id=str(instrument.id),
            kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.0000"),
            decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
            limit_price=Decimal("99.00"), pending_id="native-l03-pending",
        )
        result = session.advance_limit(
            intent, batches=tuple(_batch(number) for number in range(2, 32)),
            next_decision_ts_ns=31 * 60_000_000_000,
        )
        assert len(result.orders) == 1
        assert result.orders[0]["status"] == "ACCEPTED"
        assert result.fills == result.positions == ()
        assert result.equity == Decimal("100000.00")
    finally:
        session.close()


def test_native_marketable_limit_at_open_matches_ohlc_taker_fill() -> None:
    session, _submitter, instrument = _fixture()
    intent = LimitOrderIntent(
        intent_id="native-l07", instrument_id=str(instrument.id),
        kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.0000"),
        decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
        limit_price=Decimal("99.00"), pending_id="native-l07-pending",
    )
    book = LimitOrderBook(LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"),
        min_notional=Decimal("1"),
    ))
    book.submit(intent)
    expected = book.advance(PriceBar(
        str(instrument.id), 120_000_000_000, 180_000_000_000,
        Decimal("98.00"), Decimal("100.00"), Decimal("97.00"), Decimal("99.00"),
    ))
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        native = session.advance_limit(
            intent, batches=(_batch(2, open_price=98, high=100, low=97, close=99),),
            next_decision_ts_ns=120_000_000_000,
        )
        ohlc_fill = next(event for event in expected if event.event_kind.value == "FILLED")
        assert len(native.fills) == 1
        assert native.fills[0]["liquidity_side"] == "TAKER"
        assert Decimal(native.fills[0]["last_qty"]) == ohlc_fill.signed_qty
        assert Decimal(native.fills[0]["last_px"]) == ohlc_fill.price == Decimal("99.00")
        assert native.all_orders[0]["status"] == "FILLED"
    finally:
        session.close()


def test_native_same_bar_trigger_cancel_fails_closed_when_path_is_unordered() -> None:
    session, _submitter, instrument = _fixture()
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        order = LimitOrderIntent(
            intent_id="native-l06-order", instrument_id=str(instrument.id),
            kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.0000"),
            decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
            limit_price=Decimal("99.00"), pending_id="native-l06-pending",
        )
        session.advance_limit(
            order, batches=(_batch(2),), next_decision_ts_ns=120_000_000_000,
        )
        cancel = LimitOrderIntent(
            intent_id="native-l06-cancel", instrument_id=str(instrument.id),
            kind=OrderIntentKind.CANCEL, signed_qty=Decimal(0),
            decision_ts_ns=120_000_000_000, effective_ts_ns=150_000_000_000,
            pending_id="native-l06-pending", cancel_id="native-l06-cancel-id",
        )
        with pytest.raises(EngineBoundaryError, match="must align with a replay batch"):
            session.advance_limit(
                cancel,
                batches=(_batch(3, open_price=100, high=101, low=99, close=100),),
                next_decision_ts_ns=180_000_000_000,
            )
        assert session.state == "CLOSED"
    finally:
        if session.state != "CLOSED":
            session.close()


@pytest.mark.parametrize(
    ("price", "quantity"),
    (
        (Decimal("99.005"), Decimal("1.0000")),  # price is off tick
        (Decimal("99.00"), Decimal("1.00015")),  # quantity is off step
        (Decimal("0.50"), Decimal("1.0000")),  # below minimum notional
    ),
)
def test_native_l10_rejects_invalid_tick_step_and_minimum_before_submit(
    price: Decimal, quantity: Decimal,
) -> None:
    session, _submitter, instrument = _fixture()
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        intent = LimitOrderIntent(
            intent_id=f"native-l10-{price}-{quantity}", instrument_id=str(instrument.id),
            kind=OrderIntentKind.LIMIT, signed_qty=quantity,
            decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
            limit_price=price, pending_id="native-l10-pending",
        )
        with pytest.raises(EngineBoundaryError, match="violates native instrument representation"):
            session.advance_limit(
                intent, batches=(_batch(2),), next_decision_ts_ns=120_000_000_000,
            )
        assert session.state == "CLOSED"
    finally:
        if session.state != "CLOSED":
            session.close()


def test_native_l09_terminal_episode_cancels_unfilled_limit_before_close() -> None:
    session, _submitter, instrument = _fixture()
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        intent = LimitOrderIntent(
            intent_id="native-l09-order", instrument_id=str(instrument.id),
            kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.0000"),
            decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
            limit_price=Decimal("50.00"), pending_id="native-l09-pending",
        )
        pending = session.advance_limit(
            intent, batches=(_batch(2),), next_decision_ts_ns=120_000_000_000,
        )
        native_order_id = pending.orders[0]["client_order_id"]
        cancel_plan = ExecutionPlan(
            decision_id="native-l09-cancel", plan_id="native-l09-cancel",
            delta_qty=Decimal(0), quantity_unit="BTC", quantity_increment=Decimal("0.0001"),
            order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
            cancel_policy=PendingOrderPolicy.CANCEL_REPLACE, latency_model_id="latency-1",
            instrument_id=str(instrument.id), execution_spec_hash="d" * 64,
            idempotency_key="native-l09-cancel", action="CANCEL", reason="episode_end",
            cancel_order_ids=(native_order_id,), requires_cancel_ack=True,
        )
        flatten_plan = ExecutionPlan(
            decision_id="native-l09-flat", plan_id="native-l09-flat",
            delta_qty=Decimal(0), quantity_unit="BTC", quantity_increment=Decimal("0.0001"),
            order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
            cancel_policy=PendingOrderPolicy.NONE, latency_model_id="latency-1",
            instrument_id=str(instrument.id), execution_spec_hash="d" * 64,
            idempotency_key="native-l09-flat", action="NOOP", reason="account already flat",
        )
        terminal = session.drain_and_close(
            "force_flatten_with_cost", flatten_plan=flatten_plan,
            cancel_plan=cancel_plan, cancel_batches=(_batch(3),),
            cancel_ts_ns=180_000_000_000,
            drain_batches=(_batch(4),), terminal_ts_ns=240_000_000_000,
        )
        assert terminal.orders == terminal.positions == terminal.fills == ()
        assert terminal.all_orders[0]["status"] == "CANCELED"
        assert terminal.equity == Decimal("100000.00")
        assert session.state == "CLOSED"
    finally:
        session.close()


def test_native_limit_session_rejects_batch_before_effective_timestamp() -> None:
    session, _submitter, instrument = _fixture()
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        intent = LimitOrderIntent(
            intent_id="late-effective", instrument_id=str(instrument.id), kind=OrderIntentKind.LIMIT,
            signed_qty=Decimal("1.0000"), decision_ts_ns=60_000_000_000,
            effective_ts_ns=120_000_000_001, limit_price=Decimal("50.00"), pending_id="pending-late",
        )
        with pytest.raises(EngineBoundaryError, match="precedes LIMIT intent effectiveness"):
            session.advance_limit(intent, batches=(_batch(2), _batch(3)),
                                  next_decision_ts_ns=180_000_000_000)
        assert session.state == "CLOSED"
    finally:
        session.close()


def test_native_touch_fill_matches_ohlc_limit_path_price_quantity_and_terminal_state() -> None:
    session, _submitter, instrument = _fixture()
    rules = LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"),
        min_notional=Decimal("1"),
    )
    intent = LimitOrderIntent(
        intent_id="limit-touch-parity", instrument_id=str(instrument.id),
        kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.0000"),
        decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
        limit_price=Decimal("99.00"), pending_id="touch-parity",
    )
    book = LimitOrderBook(rules)
    book.submit(intent)
    ohlc_events = book.advance(PriceBar(
        str(instrument.id), 120_000_000_000, 180_000_000_000,
        Decimal("100.00"), Decimal("101.00"), Decimal("100.00"), Decimal("100.00"),
    ))
    ohlc_events += book.advance(PriceBar(
        str(instrument.id), 180_000_000_000, 240_000_000_000,
        Decimal("100.00"), Decimal("101.00"), Decimal("99.00"), Decimal("100.00"),
    ))
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        native = session.advance_limit(
            intent,
            batches=(_batch(2), _batch(3, open_price=100, high=101, low=99, close=100)),
            next_decision_ts_ns=180_000_000_000,
        )
        ohlc_fill = next(event for event in ohlc_events
                         if event.event_kind.value == "FILLED")
        assert book.state("touch-parity") is LimitOrderState.FILLED
        assert len(native.fills) == 1
        assert Decimal(native.fills[0]["last_qty"]) == ohlc_fill.signed_qty
        assert Decimal(native.fills[0]["last_px"]) == ohlc_fill.price
        assert Decimal(native.fills[0]["last_px"]) == Decimal("99.00")
        assert native.fills[0]["liquidity_side"] == "MAKER"
        assert native.fills[0]["commission"] == "0.10 USD"
        assert ohlc_fill.fee_class == "SCENARIO_REQUIRED"
        assert Decimal(native.positions[0]["quantity"]) == abs(ohlc_fill.signed_qty)
        assert native.equity == Decimal("100000.90")
    finally:
        session.close()


def test_native_scheduled_cancel_before_touch_matches_ohlc_l04() -> None:
    session, _submitter, instrument = _fixture()
    rules = LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"),
        min_notional=Decimal("1"),
    )
    order = LimitOrderIntent(
        intent_id="cancel-l04-order", instrument_id=str(instrument.id),
        kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.0000"),
        decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
        limit_price=Decimal("99.00"), pending_id="cancel-l04",
    )
    cancel = LimitOrderIntent(
        intent_id="cancel-l04-event", instrument_id=str(instrument.id),
        kind=OrderIntentKind.CANCEL, signed_qty=Decimal(0),
        decision_ts_ns=120_000_000_000, effective_ts_ns=180_000_000_000,
        pending_id="cancel-l04", cancel_id="cancel-l04-id",
    )
    book = LimitOrderBook(rules)
    book.submit(order)
    book.advance(PriceBar(str(instrument.id), 120_000_000_000, 180_000_000_000,
                          Decimal("100"), Decimal("101"), Decimal("100"), Decimal("100")))
    book.cancel(cancel)
    expected = book.advance(PriceBar(str(instrument.id), 180_000_000_000, 240_000_000_000,
                                     Decimal("100"), Decimal("101"), Decimal("99"), Decimal("100")))
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance_limit(order, batches=(_batch(2),), next_decision_ts_ns=120_000_000_000)
        native = session.advance_limit(
            cancel, batches=(_batch(3), _batch(4, open_price=100, high=101, low=99, close=100)),
            next_decision_ts_ns=240_000_000_000,
        )
        assert expected[-1].event_kind.value == "CANCELLED"
        assert native.fills == native.positions == ()
        assert native.all_orders[0]["status"] == "CANCELED"
        assert native.equity == Decimal("100000.00")
    finally:
        session.close()


def test_native_fill_precedes_later_scheduled_cancel_matches_ohlc_l05() -> None:
    session, _submitter, instrument = _fixture()
    rules = LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.0001"), min_qty=Decimal("0.0001"),
        min_notional=Decimal("1"),
    )
    order = LimitOrderIntent(
        intent_id="cancel-l05-order", instrument_id=str(instrument.id),
        kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.0000"),
        decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
        limit_price=Decimal("99.00"), pending_id="cancel-l05",
    )
    cancel = LimitOrderIntent(
        intent_id="cancel-l05-event", instrument_id=str(instrument.id),
        kind=OrderIntentKind.CANCEL, signed_qty=Decimal(0),
        decision_ts_ns=120_000_000_000, effective_ts_ns=240_000_000_000,
        pending_id="cancel-l05", cancel_id="cancel-l05-id",
    )
    book = LimitOrderBook(rules)
    book.submit(order)
    expected = book.advance(PriceBar(str(instrument.id), 120_000_000_000, 180_000_000_000,
                                     Decimal("100"), Decimal("101"), Decimal("99"), Decimal("100")))
    book.cancel(cancel)
    expected += book.advance(PriceBar(str(instrument.id), 180_000_000_000, 240_000_000_000,
                                      Decimal("100"), Decimal("101"), Decimal("100"), Decimal("100")))
    book.advance(PriceBar(str(instrument.id), 240_000_000_000, 300_000_000_000,
                          Decimal("100"), Decimal("101"), Decimal("100"), Decimal("100")))
    try:
        session.bootstrap((_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance_limit(order, batches=(_batch(2),), next_decision_ts_ns=120_000_000_000)
        native = session.advance_limit(
            cancel,
            batches=(_batch(3, open_price=100, high=101, low=99, close=100), _batch(4)),
            next_decision_ts_ns=240_000_000_000,
        )
        ohlc_fill = next(event for event in expected if event.event_kind.value == "FILLED")
        assert book.state("cancel-l05") is LimitOrderState.FILLED
        assert len(native.fills) == 1
        assert Decimal(native.fills[0]["last_qty"]) == ohlc_fill.signed_qty
        assert Decimal(native.fills[0]["last_px"]) == ohlc_fill.price
        assert native.all_orders[0]["status"] == "FILLED"
    finally:
        session.close()
