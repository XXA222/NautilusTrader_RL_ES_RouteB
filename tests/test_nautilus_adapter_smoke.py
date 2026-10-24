from __future__ import annotations

import re
from decimal import Decimal
from types import SimpleNamespace

import pytest

from nautilus_trader.execution import (
    DefaultFillModel, LimitOrderPartialFillModel, MakerTakerFeeModel, StaticLatencyModel,
)
from nautilus_trader.model import (
    AccountType,
    AccountId,
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
from nautilus_trader.trading import Strategy, StrategyConfig

from rles_nautilus_adapter.bootstrap import NativeBootConfiguration, PublicBacktestBuilder
from rles_nautilus_adapter.session import EngineBoundaryError, EngineSession, NativeFactsReader
from rles_contracts.dto import DataQualityGrade, ExecutionPlan, MarketEvent, OrderType, PendingOrderPolicy, TimeInForce
from rles_contracts.frozen import FrozenFields
from rles_nautilus_adapter import NativeMarketEventDecoder, NativePlanSubmitter, VisibleBatch
from rles_nautilus_adapter.limit_submitter import NativeLimitSubmitter
from rles_contracts.limit_execution import LimitInstrumentRules, LimitOrderIntent, OrderIntentKind

pytestmark = pytest.mark.micro_golden


def _instrument() -> CurrencyPair:
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol("BTC/USD"), Venue("SIM")),
        raw_symbol=Symbol("BTC/USD"),
        base_currency=Currency.from_str("BTC"),
        quote_currency=Currency.from_str("USD"),
        price_precision=2,
        size_precision=4,
        price_increment=Price(0.01, 2),
        size_increment=Quantity(0.0001, 4),
        ts_event=1,
        ts_init=1,
    )


class _NoopStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__(StrategyConfig())


def _builder(submitter: NativePlanSubmitter | None = None) -> PublicBacktestBuilder:
    return PublicBacktestBuilder(
        plan_submitter=submitter,
        config=NativeBootConfiguration(
            venue_kwargs=(
                {
                    "venue": Venue("SIM"),
                    "oms_type": OmsType.NETTING,
                    "account_type": AccountType.MARGIN,
                    "starting_balances": [Money(100_000, Currency.from_str("USD"))],
                    "base_currency": Currency.from_str("USD"),
                    "margin_model": StandardMarginModel(),
                    "fee_model": MakerTakerFeeModel(Decimal("0.001"), Decimal("0.001")),
                    "fill_model": DefaultFillModel(),
                    "latency_model": StaticLatencyModel(),
                    "bar_execution": True,
                    "liquidation_enabled": False,
                },
            ),
            instrument_builders=(_instrument,),
            strategy_builder=_NoopStrategy,
        )
    )


def test_engine_session_opens_and_disposes_a_real_pinned_backtest_engine() -> None:
    instrument = _instrument()
    builder = _builder()
    session = EngineSession(
        build_native=builder,
        submit_plan=lambda _engine, _plan: None,
        decode_event=lambda event: event,
        facts_reader=NativeFactsReader(
            account_id="SIM-ACCOUNT",
            instrument_id=instrument.id,
            currency="USD",
        ),
        contract_snapshot_factory=lambda **_facts: None,
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )

    fingerprint = session.open(SimpleNamespace(
        dataset_hash="a" * 64,
        fold_hash="b" * 64,
        config_hash="c" * 64,
    ))
    try:
        assert session.state == "READY"
        assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        assert session.fingerprint == fingerprint
    finally:
        session.close()
    assert session.state == "CLOSED"


def test_native_builder_refuses_cross_fold_execution_model_reuse() -> None:
    builder = _builder()
    engine = builder(SimpleNamespace())
    engine.dispose()
    with pytest.raises(EngineBoundaryError, match="models cannot be reused"):
        builder(SimpleNamespace())


def test_real_rc6_engine_streams_bars_and_reads_native_account_facts(capfd) -> None:
    instrument = _instrument()
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")

    def event(number: int) -> SimpleNamespace:
        timestamp = number * 60_000_000_000
        bar = Bar(
            bar_type=bar_type,
            open=instrument.make_price(100.0 + number),
            high=instrument.make_price(101.0 + number),
            low=instrument.make_price(99.0 + number),
            close=instrument.make_price(100.5 + number),
            volume=instrument.make_qty(100),
            ts_event=timestamp,
            ts_init=timestamp,
        )
        return SimpleNamespace(
            event_id=f"native-bar-{number}", source_hash="a" * 64,
            ts_event_ns=timestamp, ts_init_ns=timestamp,
            available_at_ns=timestamp, native_bar=bar,
        )

    from rles_nautilus_adapter.session import VisibleBatch

    session = EngineSession(
        build_native=_builder(),
        submit_plan=lambda _engine, _plan: pytest.fail("NOOP must not submit a native order"),
        decode_event=lambda item: item.native_bar,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"),
            instrument_id=instrument.id,
            currency="USD",
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    first = session.bootstrap((VisibleBatch(60_000_000_000, (event(1),), True),),
                              decision_ts_ns=60_000_000_000)
    assert first.equity == Decimal("100000.00")
    assert first.currency == "USD"
    assert first.positions == ()
    with pytest.raises(EngineBoundaryError, match="native equity currency differs"):
        NativeFactsReader(account_id=AccountId.from_str("SIM-001"),
                          instrument_id=instrument.id, currency="EUR").read(
                              session._engine, 60_000_000_000)
    plan = ExecutionPlan(
        decision_id="decision-1", plan_id="noop-1", instrument_id=str(instrument.id),
        quantity_unit="BTC", quantity_increment=Decimal("0.0001"),
        latency_model_id="latency-1", execution_spec_hash="d" * 64,
        order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
        cancel_policy=PendingOrderPolicy.NONE,
        action="NOOP", reason="native account already flat",
        idempotency_key="noop-1", delta_qty=Decimal(0),
    )
    second = session.advance(plan,
                             batches=(VisibleBatch(120_000_000_000, (event(2),), True),),
                             next_decision_ts_ns=120_000_000_000)
    assert second.equity == first.equity
    assert second.decision_ts_ns == 120_000_000_000
    session.close()
    assert session.state == "CLOSED"
    assert "[INFO]" not in capfd.readouterr().out


def test_engine_session_rejects_unhashed_transition_identity_before_build() -> None:
    session = EngineSession(
        build_native=lambda _episode: pytest.fail("invalid identity must not build native engine"),
        submit_plan=lambda _engine, _plan: None,
        decode_event=lambda event: event,
        facts_reader=NativeFactsReader(
            account_id="SIM-ACCOUNT",
            instrument_id=_instrument().id,
            currency="USD",
        ),
        contract_snapshot_factory=lambda **_facts: None,
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    with pytest.raises(EngineBoundaryError, match="lowercase SHA-256"):
        session.open(SimpleNamespace(
            dataset_hash="dataset-fixture",
            fold_hash="b" * 64,
            config_hash="c" * 64,
        ))
    assert session.state == "CLOSED"


def _native_order_session(submitter: NativePlanSubmitter) -> EngineSession:
    session = EngineSession(
        build_native=_builder(submitter), submit_plan=submitter,
        decode_event=lambda item: item.native_bar,
        facts_reader=NativeFactsReader(account_id=AccountId.from_str("SIM-001"),
                                       instrument_id=_instrument().id, currency="USD"),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    return session


def _order_batch(number: int) -> VisibleBatch:
    instrument = _instrument()
    timestamp = number * 60_000_000_000
    bar = Bar(
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
        open=instrument.make_price(100), high=instrument.make_price(100),
        low=instrument.make_price(100), close=instrument.make_price(100),
        volume=instrument.make_qty(100), ts_event=timestamp, ts_init=timestamp,
    )
    return VisibleBatch(timestamp, (SimpleNamespace(
        event_id=f"order-bar-{number}", source_hash="a" * 64,
        ts_event_ns=timestamp, ts_init_ns=timestamp, available_at_ns=timestamp,
        native_bar=bar),), True)


def _submitter() -> NativePlanSubmitter:
    return NativePlanSubmitter(instrument_id=str(_instrument().id), quantity_unit="BTC",
                               quantity_increment=Decimal("0.0001"),
                               execution_spec_hash="d" * 64, latency_model_id="latency-1")


def _order_plan(key: str, quantity: str) -> ExecutionPlan:
    return ExecutionPlan(
        decision_id=key, plan_id=key, instrument_id=str(_instrument().id),
        quantity_unit="BTC", quantity_increment=Decimal("0.0001"),
        latency_model_id="latency-1", execution_spec_hash="d" * 64,
        order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
        cancel_policy=PendingOrderPolicy.NONE, action="SUBMIT", reason="native order test",
        idempotency_key=key, delta_qty=Decimal(quantity),
    )


def _perpetual_instrument():
    from nautilus_trader.model import CryptoPerpetual

    usdt = Currency.from_str("USDT")
    return CryptoPerpetual(
        instrument_id=InstrumentId.from_str("ETHUSDT-PERP.SIM"),
        raw_symbol=Symbol("ETHUSDT"), base_currency=Currency.from_str("ETH"),
        quote_currency=usdt, settlement_currency=usdt, is_inverse=False,
        price_precision=2, size_precision=3, price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.001"), ts_event=1, ts_init=1,
        min_notional=Money.from_str("10 USDT"), margin_init=Decimal("1.0"),
        margin_maint=Decimal("0.35"),
    )


def _perpetual_builder(submitter: NativePlanSubmitter, *, bar_execution: bool = True,
                       fill_model: object | None = None) -> PublicBacktestBuilder:
    instrument = _perpetual_instrument()
    usdt = Currency.from_str("USDT")
    return PublicBacktestBuilder(
        plan_submitter=submitter,
        config=NativeBootConfiguration(
            venue_kwargs=({
                "venue": Venue("SIM"), "oms_type": OmsType.NETTING,
                "account_type": AccountType.MARGIN,
                "starting_balances": [Money(100_000, usdt)], "base_currency": usdt,
                "margin_model": StandardMarginModel(),
                "fee_model": MakerTakerFeeModel(Decimal("0.001"), Decimal("0.001")),
                "fill_model": (DefaultFillModel() if fill_model is None else fill_model),
                "latency_model": StaticLatencyModel(),
                "bar_execution": bar_execution, "liquidity_consumption": not bar_execution,
                "liquidation_enabled": False,
            },),
            instrument_builders=(lambda: instrument,), strategy_builder=_NoopStrategy,
        ),
    )


def _perpetual_batch(number: int, *extra_events: tuple[str, dict],
                     price: str = "1000.00",
                     ohlc: tuple[str, str, str, str] | None = None) -> VisibleBatch:
    instrument = _perpetual_instrument()
    timestamp = number * 60_000_000_000
    open_price, high_price, low_price, close_price = (
        ohlc if ohlc is not None else (price, price, price, price))
    payloads = [("ohlcv_1m", {
        "instrument": str(instrument.id), "open": open_price, "high": high_price,
        "low": low_price, "close": close_price, "volume": "100.000",
    }), *extra_events]
    events = tuple(MarketEvent(
        event_id=f"perpetual-{number}-{index}", ts_event_ns=timestamp,
        ts_init_ns=timestamp, available_at_ns=timestamp,
        source_hash=f"{index + 1:064x}", quality_grade=DataQualityGrade.B,
        event_type=kind, payload=FrozenFields.from_mapping(payload),
    ) for index, (kind, payload) in enumerate(payloads))
    return VisibleBatch(timestamp, events, True)


def _quote_payload(bid: str, bid_qty: str, ask: str, ask_qty: str) -> tuple[str, dict]:
    return "best_quote", {"instrument": "ETHUSDT-PERP.SIM", "bid_price": bid,
                           "bid_qty": bid_qty, "ask_price": ask, "ask_qty": ask_qty,
                           "update_id": 1, "transaction_time_ns": 60_000_000_000}


def _perpetual_plan(key: str, quantity: str) -> ExecutionPlan:
    instrument = _perpetual_instrument()
    return ExecutionPlan(
        decision_id=key, plan_id=key, instrument_id=str(instrument.id),
        quantity_unit="ETH", quantity_increment=Decimal("0.001"),
        latency_model_id="latency-1", execution_spec_hash="e" * 64,
        order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
        cancel_policy=PendingOrderPolicy.NONE, action="SUBMIT", reason="funding golden",
        idempotency_key=key, delta_qty=Decimal(quantity),
    )


@pytest.mark.parametrize(
    ("entry_qty", "exit_qty", "exit_price"),
    [("1.000", "-1.000", "1010.00"),
     ("-1.000", "1.000", "990.00")],
)
def test_imagined_account_matches_native_perpetual_market_roundtrip_baseline(
    entry_qty: str, exit_qty: str, exit_price: str,
) -> None:
    """Only certify the shared fixed-price, market-order, taker-fee subcase."""
    from rles_world_model.routeb.imagined_account import ImaginedAccount
    from rles_world_model.routeb.market import Candle, MINUTE_NS

    instrument = _perpetual_instrument()
    submitter = NativePlanSubmitter(
        instrument_id=str(instrument.id), quantity_unit="ETH",
        quantity_increment=Decimal("0.001"), execution_spec_hash="e" * 64,
        latency_model_id="latency-1",
    )
    decoder = NativeMarketEventDecoder(
        instrument=instrument,
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
    )
    session = EngineSession(
        build_native=_perpetual_builder(submitter), submit_plan=submitter,
        decode_event=decoder,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT",
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    account = ImaginedAccount(
        initial_equity=100_000, fee_maker=0.001, fee_taker=0.001)

    def candle(number: int, price: str) -> Candle:
        value = float(price)
        timestamp = number * MINUTE_NS
        return Candle(timestamp, timestamp, str(instrument.id), value, value,
                      value, value, 100.0)

    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    try:
        session.bootstrap(
            (_perpetual_batch(1),), decision_ts_ns=MINUTE_NS)
        account.on_bar(candle(1, "1000.00"))

        account.submit(target_qty=float(entry_qty), decision_ns=MINUTE_NS)
        entered = session.advance(
            _perpetual_plan("parity-entry", entry_qty),
            batches=(_perpetual_batch(2),), next_decision_ts_ns=2 * MINUTE_NS,
        )
        simulated_entry = account.on_bar(candle(2, "1000.00"))
        assert simulated_entry is not None
        assert len(entered.fills) == 1
        native_entry = entered.fills[0]
        native_entry_sign = (Decimal("1") if native_entry["order_side"] == "BUY"
                             else Decimal("-1"))
        assert native_entry_sign * Decimal(native_entry["last_qty"]) == Decimal(entry_qty)
        assert Decimal(native_entry["last_px"]) == Decimal("1000.00")
        assert Money.from_str(native_entry["commission"]).as_decimal() == Decimal(
            str(simulated_entry.fee))
        assert Decimal(str(simulated_entry.signed_qty)) == Decimal(entry_qty)
        assert Decimal(str(simulated_entry.price)) == Decimal("1000.00")
        assert entered.equity == Decimal(str(account.equity))

        account.submit(target_qty=0.0, decision_ns=2 * MINUTE_NS)
        terminal = session.drain_and_close(
            "force_flatten_with_cost", flatten_plan=_perpetual_plan(
                "parity-exit", exit_qty),
            drain_batches=(_perpetual_batch(3, price=exit_price),),
            terminal_ts_ns=3 * MINUTE_NS,
        )
        simulated_exit = account.on_bar(candle(3, exit_price))
        assert simulated_exit is not None
        assert len(terminal.fills) == 2
        native_exit = terminal.fills[-1]
        native_exit_sign = (Decimal("1") if native_exit["order_side"] == "BUY"
                            else Decimal("-1"))
        assert native_exit_sign * Decimal(native_exit["last_qty"]) == Decimal(exit_qty)
        assert Decimal(native_exit["last_px"]) == Decimal(exit_price)
        assert Money.from_str(native_exit["commission"]).as_decimal() == Decimal(
            str(simulated_exit.fee))
        assert Decimal(str(simulated_exit.signed_qty)) == Decimal(exit_qty)
        assert Decimal(str(simulated_exit.price)) == Decimal(exit_price)
        assert terminal.equity == Decimal(str(account.equity))
        assert terminal.equity == (Decimal("100007.99") if entry_qty.startswith("1")
                                   else Decimal("100008.01"))
    finally:
        session.close()


@pytest.mark.parametrize(
    ("signed_qty", "limit_price", "bar_ohlc", "expected_side"),
    [("1.000", "995.00", ("1000.00", "1005.00", "990.00", "1000.00"), "BUY"),
     ("-1.000", "1005.00", ("1000.00", "1010.00", "995.00", "1000.00"), "SELL")],
)
def test_imagined_account_matches_native_perpetual_intrabar_limit_touch(
    signed_qty: str, limit_price: str, bar_ohlc: tuple[str, str, str, str],
    expected_side: str,
) -> None:
    """Compare only a resting limit touched after the next bar opens."""
    from dataclasses import replace
    from rles_world_model.routeb.imagined_account import ImaginedAccount
    from rles_world_model.routeb.market import Candle, MINUTE_NS

    instrument = _perpetual_instrument()
    rules = LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.001"), min_qty=Decimal("0.001"),
        min_notional=Decimal("10"),
    )
    submitter = NativeLimitSubmitter(rules=rules, quantity_unit="ETH")
    decoder = NativeMarketEventDecoder(
        instrument=instrument,
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
    )
    session = EngineSession(
        build_native=_perpetual_builder(submitter),
        submit_plan=lambda _engine, _plan: None,
        decode_event=decoder,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT",
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
        submit_limit_intent=lambda engine, intent, timestamp: submitter(
            engine, intent, decision_ts_ns=timestamp),
    )
    account = ImaginedAccount(
        initial_equity=100_000, fee_maker=0.001, fee_taker=0.001)

    def candle(number: int, ohlc: tuple[str, str, str, str]) -> Candle:
        open_, high, low, close = (float(value) for value in ohlc)
        timestamp = number * MINUTE_NS
        return Candle(timestamp, timestamp, str(instrument.id), open_, high, low,
                      close, 100.0)

    session.open(SimpleNamespace(dataset_hash="d" * 64, fold_hash="e" * 64,
                                 config_hash="f" * 64))
    try:
        session.bootstrap((_perpetual_batch(1),), decision_ts_ns=MINUTE_NS)
        account.on_bar(candle(1, ("1000.00",) * 4))
        intent = LimitOrderIntent(
            intent_id="imagined-native-maker-parity", instrument_id=str(instrument.id),
            kind=OrderIntentKind.LIMIT, signed_qty=Decimal(signed_qty),
            decision_ts_ns=MINUTE_NS, effective_ts_ns=2 * MINUTE_NS,
            limit_price=Decimal(limit_price), pending_id="imagined-native-maker-pending",
        )
        account.submit(target_qty=float(signed_qty), decision_ns=MINUTE_NS,
                       kind="limit", limit_price=float(limit_price), delay_ns=MINUTE_NS)
        pending = session.advance_limit(
            intent, batches=(_perpetual_batch(2),), next_decision_ts_ns=2 * MINUTE_NS)
        account.on_bar(candle(2, ("1000.00",) * 4))
        assert pending.fills == ()

        touched = session.advance(
            replace(_perpetual_plan("maker-touch-observe", "1.000"),
                    action="NOOP", delta_qty=Decimal(0)),
            batches=(_perpetual_batch(3, ohlc=bar_ohlc),),
            next_decision_ts_ns=3 * MINUTE_NS,
        )
        simulated_fill = account.on_bar(candle(3, bar_ohlc))
        assert simulated_fill is not None and simulated_fill.maker
        assert len(touched.fills) == 1
        native_fill = touched.fills[0]
        assert native_fill["order_side"] == expected_side
        assert Decimal(native_fill["last_qty"]) == abs(Decimal(signed_qty))
        assert Decimal(native_fill["last_px"]) == Decimal(limit_price)
        assert float(Money.from_str(native_fill["commission"]).as_decimal()) == pytest.approx(
            simulated_fill.fee, rel=0, abs=1e-8)
        assert Decimal(str(simulated_fill.price)) == Decimal(limit_price)
        assert float(touched.equity) == pytest.approx(account.equity, rel=0, abs=1e-8)
    finally:
        session.close()


def test_imagined_account_is_not_economics_eligible_for_native_partial_limit_fill() -> None:
    """Pin a native partial-fill counterexample to the all-or-none prototype."""
    from dataclasses import replace
    from rles_world_model.routeb.imagined_account import ImaginedAccount
    from rles_world_model.routeb.market import Candle, MINUTE_NS

    instrument = _perpetual_instrument()
    submitter = NativeLimitSubmitter(rules=LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.001"), min_qty=Decimal("0.001"), min_notional=Decimal("10"),
    ), quantity_unit="ETH")
    decoder = NativeMarketEventDecoder(
        instrument=instrument,
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
    )
    session = EngineSession(
        build_native=lambda _scope: _perpetual_builder(
            submitter, fill_model=LimitOrderPartialFillModel(random_seed=0))(_scope),
        submit_plan=lambda _engine, _plan: None,
        decode_event=decoder,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT",
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
        submit_limit_intent=lambda engine, intent, timestamp: submitter(
            engine, intent, decision_ts_ns=timestamp),
    )
    account = ImaginedAccount(initial_equity=100_000, fee_maker=0.001, fee_taker=0.001)

    def candle(number: int, ohlc: tuple[str, str, str, str]) -> Candle:
        open_, high, low, close = (float(value) for value in ohlc)
        timestamp = number * MINUTE_NS
        return Candle(timestamp, timestamp, str(instrument.id), open_, high, low,
                      close, 100.0)

    session.open(SimpleNamespace(dataset_hash="8" * 64, fold_hash="9" * 64,
                                 config_hash="a" * 64))
    try:
        session.bootstrap((_perpetual_batch(1),), decision_ts_ns=MINUTE_NS)
        account.on_bar(candle(1, ("1000.00",) * 4))
        order = LimitOrderIntent(
            intent_id="partial-native-limit", instrument_id=str(instrument.id),
            kind=OrderIntentKind.LIMIT, signed_qty=Decimal("10.000"),
            decision_ts_ns=MINUTE_NS, effective_ts_ns=2 * MINUTE_NS,
            limit_price=Decimal("995.00"), pending_id="partial-native-pending",
        )
        account.submit(target_qty=10.0, decision_ns=MINUTE_NS, kind="limit",
                       limit_price=995.0, delay_ns=MINUTE_NS)
        session.advance_limit(
            order, batches=(_perpetual_batch(2),), next_decision_ts_ns=2 * MINUTE_NS,
        )
        account.on_bar(candle(2, ("1000.00",) * 4))
        touched = session.advance(
            replace(_perpetual_plan("partial-native-touch", "1.000"),
                    action="NOOP", delta_qty=Decimal(0)),
            batches=(_perpetual_batch(
                3, ohlc=("1000.00", "1005.00", "990.00", "1000.00"),
            ),),
            next_decision_ts_ns=3 * MINUTE_NS,
        )
        simulated_fill = account.on_bar(candle(
            3, ("1000.00", "1005.00", "990.00", "1000.00")))
        native_filled = sum(Decimal(fill["last_qty"]) for fill in touched.fills)
        assert len(touched.fills) == 2
        assert all(Decimal(fill["last_qty"]) < Decimal("10.000")
                   for fill in touched.fills)
        assert native_filled == Decimal("10.000")
        assert simulated_fill is not None
        assert len(account.fills) == 1
        assert Decimal(str(simulated_fill.signed_qty)) == Decimal("10.0")
        native_fee = sum(
            (Money.from_str(fill["commission"]).as_decimal() for fill in touched.fills),
            start=Decimal(0),
        )
        assert native_fee != Decimal(str(simulated_fill.fee))
        assert account.economics_eligible is False
    finally:
        session.close()


def test_imagined_account_matches_native_limit_cancel_before_later_touch() -> None:
    """A canceled resting order must remain unfilled when a later bar touches it."""
    from dataclasses import replace
    from rles_world_model.routeb.imagined_account import ImaginedAccount
    from rles_world_model.routeb.market import Candle, MINUTE_NS

    instrument = _perpetual_instrument()
    submitter = NativeLimitSubmitter(rules=LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.001"), min_qty=Decimal("0.001"), min_notional=Decimal("10"),
    ), quantity_unit="ETH")
    decoder = NativeMarketEventDecoder(
        instrument=instrument,
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
    )
    session = EngineSession(
        build_native=_perpetual_builder(submitter), submit_plan=lambda _engine, _plan: None,
        decode_event=decoder,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT",
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
        submit_limit_intent=lambda engine, intent, timestamp: submitter(
            engine, intent, decision_ts_ns=timestamp),
    )
    account = ImaginedAccount(initial_equity=100_000, fee_maker=0.001, fee_taker=0.001)

    def candle(number: int, ohlc: tuple[str, str, str, str]) -> Candle:
        open_, high, low, close = (float(value) for value in ohlc)
        timestamp = number * MINUTE_NS
        return Candle(timestamp, timestamp, str(instrument.id), open_, high, low,
                      close, 100.0)

    session.open(SimpleNamespace(dataset_hash="d" * 64, fold_hash="e" * 64,
                                 config_hash="f" * 64))
    try:
        session.bootstrap((_perpetual_batch(1),), decision_ts_ns=MINUTE_NS)
        account.on_bar(candle(1, ("1000.00",) * 4))
        order = LimitOrderIntent(
            intent_id="cancel-parity-limit", instrument_id=str(instrument.id),
            kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.000"),
            decision_ts_ns=MINUTE_NS, effective_ts_ns=MINUTE_NS + 1,
            limit_price=Decimal("995.00"), pending_id="cancel-parity-pending",
        )
        account.submit(target_qty=1.0, decision_ns=MINUTE_NS, kind="limit",
                       limit_price=995.0, delay_ns=1)
        pending = session.advance_limit(
            order, batches=(_perpetual_batch(2),), next_decision_ts_ns=2 * MINUTE_NS,
        )
        account.on_bar(candle(2, ("1000.00",) * 4))
        assert pending.fills == ()

        account.cancel()
        cancel = LimitOrderIntent(
            intent_id="cancel-parity-action", instrument_id=str(instrument.id),
            kind=OrderIntentKind.CANCEL, signed_qty=Decimal(0),
            decision_ts_ns=2 * MINUTE_NS, effective_ts_ns=3 * MINUTE_NS,
            pending_id="cancel-parity-pending", cancel_id="cancel-parity-event",
        )
        canceled = session.advance_limit(
            cancel, batches=(_perpetual_batch(3),), next_decision_ts_ns=3 * MINUTE_NS,
        )
        account.on_bar(candle(3, ("1000.00",) * 4))
        assert canceled.all_orders[0]["status"] == "CANCELED"
        assert canceled.fills == ()

        touched = session.advance(
            replace(_perpetual_plan("cancel-parity-touch", "1.000"),
                    action="NOOP", delta_qty=Decimal(0)),
            batches=(_perpetual_batch(
                4, ohlc=("1000.00", "1005.00", "990.00", "1000.00"),
            ),),
            next_decision_ts_ns=4 * MINUTE_NS,
        )
        assert account.on_bar(candle(4, ("1000.00", "1005.00", "990.00", "1000.00"))) is None
        assert touched.fills == ()
        assert touched.positions == ()
        assert touched.equity == Decimal("100000.00")
        assert touched.equity == Decimal(str(account.equity))
    finally:
        session.close()


@pytest.mark.parametrize("entry_qty,exit_qty", [("1.0000", "-1.0000"), ("-1.0000", "1.0000")])
def test_native_submitter_executes_market_fees_and_terminal_flatten(entry_qty: str, exit_qty: str) -> None:
    session = _native_order_session(_submitter())
    try:
        initial = session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        entered = session.advance(_order_plan("entry", entry_qty), batches=(_order_batch(2),),
                                  next_decision_ts_ns=120_000_000_000)
        assert len(entered.positions) == 1
        assert entered.orders == ()
        assert len(entered.fills) == 1
        assert entered.equity == initial.equity - Decimal("0.10")
        terminal = session.drain_and_close(
            "force_flatten_with_cost", flatten_plan=_order_plan("exit", exit_qty),
            drain_batches=(_order_batch(3),), terminal_ts_ns=180_000_000_000,
        )
        assert terminal.positions == terminal.orders == ()
        assert len(terminal.fills) == 2
        assert len(terminal.all_orders) == 2
        assert sum(order["is_reduce_only"] for order in terminal.all_orders) == 1
        assert terminal.equity == initial.equity - Decimal("0.20")
        assert session.state == "CLOSED"
    finally:
        session.close()


@pytest.mark.parametrize("sign", [1, -1])
def test_native_hard_risk_partial_reduction_is_reduce_only(sign: int) -> None:
    from dataclasses import replace

    session = _native_order_session(_submitter())
    try:
        initial = session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance(_order_plan("entry", str(sign)), batches=(_order_batch(2),),
                        next_decision_ts_ns=120_000_000_000)
        reduced = session.advance(
            replace(_order_plan("hard-reduce", str(Decimal("-0.5") * sign)), reason="hard_risk_reduce"),
            batches=(_order_batch(3),), next_decision_ts_ns=180_000_000_000,
        )
        assert len(reduced.positions) == 1
        assert Decimal(reduced.positions[0]["signed_qty"]) == Decimal("0.5") * sign
        assert len(reduced.fills) == 2
        assert reduced.equity == initial.equity - Decimal("0.15")
        assert sum(order["is_reduce_only"] for order in reduced.all_orders) == 1
        terminal = session.drain_and_close(
            "force_flatten_with_cost", flatten_plan=_order_plan("exit", str(Decimal("-0.5") * sign)),
            drain_batches=(_order_batch(4),), terminal_ts_ns=240_000_000_000,
        )
        assert terminal.positions == terminal.orders == ()
        assert terminal.equity == initial.equity - Decimal("0.20")
    finally:
        session.close()


@pytest.mark.parametrize("quantity", ["1", "-2"])
def test_native_hard_risk_rejects_expansion_and_reversal_before_submission(quantity: str) -> None:
    from dataclasses import replace

    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance(_order_plan("entry", "1"), batches=(_order_batch(2),),
                        next_decision_ts_ns=120_000_000_000)
        with pytest.raises(EngineBoundaryError, match="must reduce the owned"):
            submitter(session._engine, replace(_order_plan("bad-reduce", quantity), reason="hard_risk_reduce"))
        assert len(session.snapshot().all_orders) == 1
    finally:
        session.close()


@pytest.mark.parametrize("sign", [-1, 1])
def test_native_voluntary_flat_is_a_full_reduce_only_exit(sign):
    from dataclasses import replace
    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        initial = session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance(_order_plan("entry", str(sign)), batches=(_order_batch(2),),
                        next_decision_ts_ns=120_000_000_000)
        with pytest.raises(EngineBoundaryError, match="quantity differs from native position"):
            submitter(session._engine, replace(_order_plan("bad-flat", str(-0.5 * sign)), reason="policy_flat"))
        flat = session.advance(replace(_order_plan("policy-exit", str(-sign)), reason="policy_flat"),
                               batches=(_order_batch(3),), next_decision_ts_ns=180_000_000_000)
        assert flat.positions == ()
        assert flat.equity == initial.equity - Decimal("0.20")
        assert sum(order["is_reduce_only"] for order in flat.all_orders) == 1
    finally:
        session.close()


@pytest.mark.parametrize("quantity", ["-2.0000", "-0.5000", "1.0000"])
def test_native_terminal_rejects_wrong_close_quantity_before_submitting(quantity: str) -> None:
    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance(_order_plan("entry", "1.0000"), batches=(_order_batch(2),),
                        next_decision_ts_ns=120_000_000_000)
        with pytest.raises(EngineBoundaryError, match="quantity differs from native position"):
            submitter.submit_flatten(session._engine, _order_plan("wrong-exit", quantity))
        assert len(session.snapshot().all_orders) == 1
    finally:
        session.close()


@pytest.mark.parametrize("field,value", [
    ("execution_spec_hash", "e" * 64), ("latency_model_id", "other-latency"),
    ("quantity_unit", "ETH"), ("quantity_increment", Decimal("0.001")),
    ("instrument_id", "ETH/USD.SIM"),
])
def test_native_submitter_rejects_contract_drift_before_order(field: str, value: object) -> None:
    from dataclasses import replace

    session = _native_order_session(_submitter())
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        with pytest.raises(EngineBoundaryError, match="frozen native execution contract"):
            session.advance(replace(_order_plan("drift", "1.0000"), **{field: value}),
                            batches=(_order_batch(2),), next_decision_ts_ns=120_000_000_000)
        assert session.state == "CLOSED"
    finally:
        session.close()


@pytest.mark.parametrize("terminal", [False, True])
def test_native_submitter_cancels_live_strategy_order_and_observes_ack(terminal: bool) -> None:
    from dataclasses import replace
    from nautilus_trader.model import ClientOrderId, OrderSide

    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        # Native limit order creates a deterministic pending command without
        # emulating an exchange or introducing another production order type.
        order = submitter._strategy.order_factory.limit(
            instrument_id=_instrument().id, order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.0000"), price=Price.from_str("50.00"),
            client_order_id=ClientOrderId("test-pending-native"),
        )
        submitter._strategy.submit_order(order)
        wait = replace(_order_plan("wait-pending", "1.0000"), action="WAIT",
                       delta_qty=Decimal(0), cancel_policy=PendingOrderPolicy.WAIT)
        pending = session.advance(wait, batches=(_order_batch(2),),
                                  next_decision_ts_ns=120_000_000_000)
        assert len(pending.orders) == 1
        cancel = replace(_order_plan("cancel-pending", "1.0000"), action="CANCEL",
                         delta_qty=Decimal(0), cancel_policy=PendingOrderPolicy.CANCEL_REPLACE,
                         cancel_order_ids=("test-pending-native",), requires_cancel_ack=True)
        if terminal:
            flat = replace(_order_plan("already-flat", "1.0000"), action="NOOP", delta_qty=Decimal(0))
            acknowledged = session.drain_and_close(
                "force_flatten_with_cost", cancel_plan=cancel,
                cancel_batches=(_order_batch(3),), cancel_ts_ns=180_000_000_000,
                flatten_plan=flat, drain_batches=(_order_batch(4),), terminal_ts_ns=240_000_000_000,
            )
            assert session.state == "CLOSED"
        else:
            acknowledged = session.advance(cancel, batches=(_order_batch(3),),
                                           next_decision_ts_ns=180_000_000_000)
        assert acknowledged.orders == ()
        assert acknowledged.positions == acknowledged.fills == ()
        assert acknowledged.all_orders[0]["status"] == "CANCELED"
        assert acknowledged.equity == Decimal("100000.00")
    finally:
        session.close()


def test_native_submitter_rejects_reused_action_and_cross_episode_binding() -> None:
    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        plan = _order_plan("once", "1.0000")
        session.advance(plan, batches=(_order_batch(2),), next_decision_ts_ns=120_000_000_000)
        with pytest.raises(EngineBoundaryError, match="already used"):
            submitter(session._engine, plan)
        assert len(session.snapshot().all_orders) == 1
        with pytest.raises(EngineBoundaryError, match="cannot be rebound"):
            submitter.bind(session._engine, submitter._strategy)
    finally:
        session.close()


def test_native_fills_retain_closed_roundtrip_after_reopening_position() -> None:
    session = _native_order_session(_submitter())
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        for number, quantity in [(2, "1.0000"), (3, "-1.0000"), (4, "1.0000")]:
            snapshot = session.advance(_order_plan(f"roundtrip-{number}", quantity),
                                       batches=(_order_batch(number),),
                                       next_decision_ts_ns=number * 60_000_000_000)
            assert len(snapshot.fills) == number - 1
        terminal = session.drain_and_close(
            "force_flatten_with_cost", flatten_plan=_order_plan("roundtrip-exit", "-1.0000"),
            drain_batches=(_order_batch(5),), terminal_ts_ns=300_000_000_000,
        )
        assert len(terminal.fills) == len(terminal.all_orders) == 4
        assert terminal.equity == Decimal("99999.60")
    finally:
        session.close()


@pytest.mark.parametrize(
    ("quantity", "funding_change", "equity"),
    [("1.000", Decimal("-1.00"), Decimal("99998.00")),
     ("-1.000", Decimal("1.00"), Decimal("100000.00"))],
)
@pytest.mark.parametrize("compact_history", [False, True])
def test_native_perpetual_funding_rate_settles_once_in_account(
    quantity: str, funding_change: Decimal, equity: Decimal, compact_history: bool,
) -> None:
    from dataclasses import replace
    from nautilus_trader.model import BarType
    from rles_world_model.routeb.imagined_account import ImaginedAccount
    from rles_world_model.routeb.market import Candle, MINUTE_NS

    instrument = _perpetual_instrument()
    submitter = NativePlanSubmitter(
        instrument_id=str(instrument.id), quantity_unit="ETH",
        quantity_increment=Decimal("0.001"), execution_spec_hash="e" * 64,
        latency_model_id="latency-1",
    )
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    decoder = NativeMarketEventDecoder(instrument=instrument, bar_type=bar_type)
    session = EngineSession(
        build_native=_perpetual_builder(submitter), submit_plan=submitter,
        decode_event=decoder,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT", compact_history=compact_history,
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    imagined = ImaginedAccount(
        initial_equity=100_000, fee_maker=0.001, fee_taker=0.001)

    def imagined_bar(number: int, price: float = 1000.0) -> Candle:
        timestamp = number * MINUTE_NS
        return Candle(timestamp, timestamp, str(instrument.id), price, price,
                      price, price, 100.0)

    try:
        initial = session.bootstrap((_perpetual_batch(1),), decision_ts_ns=60_000_000_000)
        imagined.on_bar(imagined_bar(1))
        entered = session.advance(
            _perpetual_plan("funding-entry", quantity),
            batches=(_perpetual_batch(2),), next_decision_ts_ns=120_000_000_000,
        )
        imagined.submit(target_qty=float(Decimal(quantity)), decision_ns=MINUTE_NS)
        imagined_entry = imagined.on_bar(imagined_bar(2))
        assert imagined_entry is not None
        assert Decimal(str(imagined.equity)) == entered.equity
        assert len(entered.positions) == len(entered.fills) == 1
        funding_event_ns = 180_000_000_000
        settlement_ns = 240_000_000_000
        mark = ("mark", {"instrument": str(instrument.id), "price": "1000.00"})
        rate = ("funding_update", {"instrument": str(instrument.id), "rate": "0.001",
                                    "interval_minutes": 480,
                                    "next_funding_ns": settlement_ns})
        wait = replace(_perpetual_plan("funding-observe", "1.000"),
                       action="NOOP", delta_qty=Decimal(0))
        session.advance(
            wait, batches=(_perpetual_batch(3, mark, rate),),
            next_decision_ts_ns=funding_event_ns,
        )
        imagined.on_bar(imagined_bar(3))
        with pytest.raises(ValueError, match="not-yet-available"):
            imagined.apply_funding_settlement(
                event_id="funding-too-early", rate=0.001, mark_price=1000.0,
                settlement_ns=settlement_ns,
            )
        settlement_event = ("funding_settlement", {
            "instrument": str(instrument.id), "rate": "0.001",
            "interval_minutes": 480, "settlement_ns": settlement_ns,
        })
        funding_boundary = session.advance(
            replace(wait, decision_id="funding-boundary", plan_id="funding-boundary",
                    idempotency_key="funding-boundary"),
            batches=(_perpetual_batch(4, settlement_event),),
            next_decision_ts_ns=settlement_ns,
        )
        imagined.on_bar(imagined_bar(4))
        imagined_funding = imagined.apply_funding_settlement(
            event_id="funding-240s", rate=0.001, mark_price=1000.0,
            settlement_ns=settlement_ns,
        )
        assert Decimal(str(imagined_funding)) == funding_change
        with pytest.raises(ValueError, match="duplicate"):
            imagined.apply_funding_settlement(
                event_id="funding-240s", rate=0.001, mark_price=1000.0,
                settlement_ns=settlement_ns,
            )
        settled = session.advance(
            replace(wait, decision_id="funding-settled", plan_id="funding-settled",
                    idempotency_key="funding-settled"),
            batches=(_perpetual_batch(5),), next_decision_ts_ns=300_000_000_000,
        )
        assert settled.mark == Decimal("1000.00")
        assert len(funding_boundary.funding) == 1
        adjustment = funding_boundary.funding[0]
        assert Money.from_str(adjustment["pnl_change"]).as_decimal() == funding_change
        assert settled.equity == initial.equity - Decimal("1.00") + funding_change
        assert settled.equity == equity
        assert Decimal(str(imagined.equity)) == settled.equity
        assert adjustment["ts_event"] == settlement_ns
        assert settled.margins == entered.margins
        if compact_history:
            assert settled.funding == ()
            assert session.snapshot().funding == ()
        else:
            assert len(settled.funding) == 1

        imagined.on_bar(imagined_bar(5))
        second_settlement_ns = 420_000_000_000
        second_mark = ("mark", {"instrument": str(instrument.id), "price": "1100.00"})
        second_rate = ("funding_update", {
            "instrument": str(instrument.id), "rate": "0.002",
            "interval_minutes": 480, "next_funding_ns": second_settlement_ns,
        })
        second_observation = replace(
            wait, decision_id="funding-observe-2", plan_id="funding-observe-2",
            idempotency_key="funding-observe-2")
        session.advance(
            second_observation,
            batches=(_perpetual_batch(6, second_mark, second_rate),),
            next_decision_ts_ns=360_000_000_000,
        )
        imagined.on_bar(imagined_bar(6, price=1100.0))
        with pytest.raises(ValueError, match="not-yet-available"):
            imagined.apply_funding_settlement(
                event_id="funding-420s-too-early", rate=0.002, mark_price=1100.0,
                settlement_ns=second_settlement_ns,
            )
        second_event = ("funding_settlement", {
            "instrument": str(instrument.id), "rate": "0.002",
            "interval_minutes": 480, "settlement_ns": second_settlement_ns,
        })
        funding_boundary_2 = session.advance(
            replace(wait, decision_id="funding-boundary-2", plan_id="funding-boundary-2",
                    idempotency_key="funding-boundary-2"),
            batches=(_perpetual_batch(7, second_event),),
            next_decision_ts_ns=second_settlement_ns,
        )
        imagined.on_bar(imagined_bar(7, price=1100.0))
        second_funding_change = imagined.apply_funding_settlement(
            event_id="funding-420s", rate=0.002, mark_price=1100.0,
            settlement_ns=second_settlement_ns,
        )
        expected_second_funding = -Decimal(quantity) * Decimal("1100.00") * Decimal("0.002")
        assert Decimal(str(second_funding_change)) == expected_second_funding
        with pytest.raises(ValueError, match="duplicate"):
            imagined.apply_funding_settlement(
                event_id="funding-420s", rate=0.002, mark_price=1100.0,
                settlement_ns=second_settlement_ns,
            )
        settled_2 = session.advance(
            replace(wait, decision_id="funding-settled-2", plan_id="funding-settled-2",
                    idempotency_key="funding-settled-2"),
            batches=(_perpetual_batch(8),), next_decision_ts_ns=480_000_000_000,
        )
        expected_equity_2 = equity + Decimal(quantity) * Decimal("100.00") + expected_second_funding
        assert settled_2.mark == Decimal("1100.00")
        assert len(funding_boundary_2.funding) == (1 if compact_history else 2)
        adjustment_2 = funding_boundary_2.funding[-1]
        assert Money.from_str(adjustment_2["pnl_change"]).as_decimal() == expected_second_funding
        assert settled_2.equity == expected_equity_2
        assert Decimal(str(imagined.equity)) == settled_2.equity
        assert settled_2.margins == entered.margins
        if compact_history:
            assert settled_2.funding == ()
        else:
            assert len(settled_2.funding) >= 1
    finally:
        session.close()


def test_native_l08_pending_limit_remains_unfunded_across_perpetual_settlement() -> None:
    from dataclasses import replace
    from nautilus_trader.model import BarType

    instrument = _perpetual_instrument()
    rules = LimitInstrumentRules(
        instrument_id=str(instrument.id), price_tick=Decimal("0.01"),
        qty_step=Decimal("0.001"), min_qty=Decimal("0.001"),
        min_notional=Decimal("10"),
    )
    submitter = NativeLimitSubmitter(rules=rules, quantity_unit="ETH")
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    decoder = NativeMarketEventDecoder(instrument=instrument, bar_type=bar_type)
    session = EngineSession(
        build_native=_perpetual_builder(submitter), submit_plan=lambda _engine, _plan: None,
        decode_event=decoder,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT",
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
        submit_limit_intent=lambda engine, intent, ts: submitter(
            engine, intent, decision_ts_ns=ts,
        ),
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    try:
        initial = session.bootstrap((_perpetual_batch(1),), decision_ts_ns=60_000_000_000)
        pending = session.advance_limit(
            LimitOrderIntent(
                intent_id="native-l08-order", instrument_id=str(instrument.id),
                kind=OrderIntentKind.LIMIT, signed_qty=Decimal("1.000"),
                decision_ts_ns=60_000_000_000, effective_ts_ns=60_000_000_001,
                limit_price=Decimal("500.00"), pending_id="native-l08-pending",
            ),
            batches=(_perpetual_batch(2),), next_decision_ts_ns=120_000_000_000,
        )
        wait = replace(_perpetual_plan("native-l08-funding-observe", "1.000"),
                       action="NOOP", delta_qty=Decimal(0))
        mark = ("mark", {"instrument": str(instrument.id), "price": "1000.00"})
        rate = ("funding_update", {"instrument": str(instrument.id), "rate": "0.001",
                                   "interval_minutes": 480,
                                   "next_funding_ns": 240_000_000_000})
        session.advance(
            wait, batches=(_perpetual_batch(3, mark, rate),),
            next_decision_ts_ns=180_000_000_000,
        )
        settled = session.advance(
            replace(wait, decision_id="native-l08-settlement", plan_id="native-l08-settlement",
                    idempotency_key="native-l08-settlement"),
            batches=(_perpetual_batch(4, ("funding_settlement", {
                "instrument": str(instrument.id), "rate": "0.001",
                "interval_minutes": 480, "settlement_ns": 240_000_000_000,
            })),),
            next_decision_ts_ns=240_000_000_000,
        )
        assert len(pending.orders) == len(settled.orders) == 1
        assert pending.orders[0]["status"] == settled.orders[0]["status"] == "ACCEPTED"
        assert settled.fills == settled.positions == settled.funding == ()
        assert settled.equity == initial.equity
    finally:
        session.close()


def test_compact_funding_reader_skips_unchanged_positions_and_emits_only_new_facts() -> None:
    reader = NativeFactsReader(account_id="SIM-001", instrument_id="ETHUSDT-PERP.SIM",
                               currency="USDT", compact_history=True)
    first_record = {"event_id": "funding-1", "ts_init": 10, "ts_event": 10}
    second_record = {"event_id": "funding-2", "ts_init": 20, "ts_event": 20}

    class Adjustment:
        adjustment_type = "FUNDING"

        def __init__(self, record):
            self.record = record

        def to_dict(self):
            return self.record

    class Position:
        id = "position-1"
        instrument_id = "ETHUSDT-PERP.SIM"
        ts_opened = 1

        def __init__(self, timestamp, adjustments):
            self.ts_last = timestamp
            self.records = list(adjustments)
            self.adjustment_reads = 0

        def adjustments(self):
            self.adjustment_reads += 1
            return tuple(Adjustment(record) for record in self.records)

    archived = Position(10, (first_record,))
    live = Position(10, (first_record,))
    cache = SimpleNamespace(position_snapshots=lambda **_kwargs: (archived,))

    first = reader._compact_funding_updates((live,), cache, "FUNDING")
    unchanged = reader._compact_funding_updates((live,), cache, "FUNDING")
    assert first == (first_record,)
    assert unchanged == ()
    assert live.adjustment_reads == 1
    assert archived.adjustment_reads == 0

    live.ts_last = 20
    live.records.append(second_record)
    next_update = reader._compact_funding_updates((live,), cache, "FUNDING")
    assert next_update == (second_record,)
    assert live.adjustment_reads == 2


def test_native_quote_execution_uses_displayed_book_and_records_submission_reference() -> None:
    from rles_execution import FillProfile, LatencyProfile, ResolvedExecutionConfig
    from rles_nautilus_adapter.perpetual import make_perpetual_session

    latency = LatencyProfile("quote-base", 0, 0, 0)
    stress_latency = LatencyProfile("quote-stress", 0, 0, 1_000_000)
    execution = ResolvedExecutionConfig(
        "0.4.0", Decimal("0.01"), "wait", "MARKET", "GTC", "force_flatten_with_cost",
        latency, stress_latency, FillProfile("native-book", Decimal(0), "quote"),
        FillProfile("native-book-stress", Decimal(0), "quote"),
    )
    rule = {
        "instrument": "ETHUSDT-PERP.SIM", "tick_size": "0.01", "step_size": "0.001",
        "min_qty": "0.001", "min_notional": "10", "maker_fee": "0.001",
        "taker_fee": "0.001", "initial_margin_rate": "0.01",
        "maintenance_margin_rate": "0.005", "effective_ns": 1,
    }
    references: dict[str, Decimal] = {}
    session = make_perpetual_session(
        rule=rule, execution=execution, mode="base", starting_balance=Decimal("100000"),
        record_fill_reference=lambda client_id, reference: references.__setitem__(client_id, reference),
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash=execution.fingerprint()))
    try:
        from dataclasses import replace

        session.bootstrap((_perpetual_batch(
            1, ("best_quote", {"instrument": rule["instrument"], "bid_price": "1000.00",
                               "bid_qty": "1.000", "ask_price": "1000.01", "ask_qty": "0.100",
                               "update_id": 1, "transaction_time_ns": 60_000_000_000}),
        ),), decision_ts_ns=60_000_000_000)
        entered = session.advance(
            replace(_perpetual_plan("quote-entry", "0.500"),
                    latency_model_id=latency.model_id,
                    execution_spec_hash=execution.fingerprint()),
            batches=(_perpetual_batch(
                2, ("best_quote", {"instrument": rule["instrument"], "bid_price": "1000.04",
                                   "bid_qty": "1.000", "ask_price": "1000.05", "ask_qty": "0.400",
                                   "update_id": 2, "transaction_time_ns": 120_000_000_000}),
            ),), next_decision_ts_ns=120_000_000_000,
        )
        assert sum(Decimal(fill["last_qty"]) for fill in entered.fills) == Decimal("0.500")
        at_best_ask = sum(Decimal(fill["last_qty"]) for fill in entered.fills
                          if Decimal(fill["last_px"]) == Decimal("1000.01"))
        assert at_best_ask <= Decimal("0.100")
        assert max(Decimal(fill["last_px"]) for fill in entered.fills) <= Decimal("1000.05")
        assert all(references[fill["client_order_id"]] == Decimal("1000.01") for fill in entered.fills)
        assert entered.orders == ()
    finally:
        session.close()


def test_native_bar_depth_stress_limits_fill_size_to_frozen_synthetic_fraction() -> None:
    from dataclasses import replace
    from rles_execution import FillProfile, LatencyProfile, ResolvedExecutionConfig
    from rles_nautilus_adapter.perpetual import make_perpetual_session

    latency = LatencyProfile("depth-base", 0, 0, 0)
    stress_latency = LatencyProfile("depth-stress", 0, 0, 0)
    execution = ResolvedExecutionConfig(
        "0.4.0", Decimal("0.01"), "wait", "MARKET", "GTC", "force_flatten_with_cost",
        latency, stress_latency,
        FillProfile("bar-base", Decimal("1")),
        FillProfile("bar-limited-depth", Decimal("2"), simulated_depth_fraction=Decimal("0.4")),
    )
    rule = {
        "instrument": "ETHUSDT-PERP.SIM", "tick_size": "0.01", "step_size": "0.001",
        "min_qty": "0.001", "min_notional": "10", "maker_fee": "0.001",
        "taker_fee": "0.001", "initial_margin_rate": "0.01",
        "maintenance_margin_rate": "0.005", "effective_ns": 1,
    }
    session = make_perpetual_session(
        rule=rule, execution=execution, mode="stress", starting_balance=Decimal("100000"),
    )
    session.open(SimpleNamespace(dataset_hash="c" * 64, fold_hash="d" * 64,
                                 config_hash=execution.fingerprint()))
    try:
        session.bootstrap((_perpetual_batch(1),), decision_ts_ns=60_000_000_000)
        plan = replace(_perpetual_plan("partial-depth", "0.500"),
                       latency_model_id=stress_latency.model_id,
                       execution_spec_hash=execution.fingerprint())
        snapshot = session.advance(plan, batches=(_perpetual_batch(2),),
                                   next_decision_ts_ns=120_000_000_000)
        filled = sum(Decimal(fill["last_qty"]) for fill in snapshot.fills)
        assert filled == Decimal("0.200")
        assert filled / Decimal("0.500") == Decimal("0.4")
        assert len(snapshot.orders) == 1
    finally:
        session.close()


def test_native_market_event_decoder_checks_index_quality_scope_and_funding_conflicts() -> None:
    from nautilus_trader.model import BarType, IndexPriceUpdate, QuoteTick

    instrument = _perpetual_instrument()
    decoder = NativeMarketEventDecoder(
        instrument=instrument,
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
    )

    def market_event(event_id: str, kind: str, timestamp: int, payload: dict,
                     quality: DataQualityGrade = DataQualityGrade.B,
                     available_at: int | None = None):
        initialized = timestamp if available_at is None else available_at
        return MarketEvent(
            event_id=event_id, ts_event_ns=timestamp, ts_init_ns=initialized,
            available_at_ns=initialized, source_hash="c" * 64, quality_grade=quality,
            event_type=kind, payload=FrozenFields.from_mapping(payload),
        )

    index = decoder(market_event(
        "index-1", "index", 100,
        {"instrument": str(instrument.id), "price": "1000.25"},
    ))
    assert isinstance(index, IndexPriceUpdate)
    assert index.value == instrument.make_price(1000.25)
    quote = decoder(market_event("quote-1", "best_quote", 100, {
        "instrument": str(instrument.id), "bid_price": "1000.20", "bid_qty": "0.100",
        "ask_price": "1000.30", "ask_qty": "0.200", "update_id": 1,
        "transaction_time_ns": 99,
    }))
    assert isinstance(quote, QuoteTick)
    assert quote.bid_price == instrument.make_price(1000.20)
    assert quote.ask_size == instrument.make_qty(0.200)
    with pytest.raises(EngineBoundaryError, match="unavailable or has unusable quality"):
        decoder(market_event(
            "low-quality-mark", "mark", 101,
            {"instrument": str(instrument.id), "price": "1000"},
            DataQualityGrade.D,
        ))
    with pytest.raises(EngineBoundaryError, match="differs from native adapter scope"):
        decoder(market_event(
            "wrong-instrument", "index", 102,
            {"instrument": "BTCUSDT-PERP.SIM", "price": "1000"},
        ))

    scheduled = market_event("rate-1", "funding_update", 110, {
        "instrument": str(instrument.id), "rate": "0.001", "next_funding_ns": 200,
    })
    assert decoder(scheduled).next_funding_ns == 200
    late_settlement = market_event("published-settlement", "funding_settlement", 130, {
        "instrument": str(instrument.id), "rate": "0.001", "settlement_ns": 130,
        "interval_minutes": 480,
    }, available_at=140)
    decoded_settlement = decoder(late_settlement)
    assert decoded_settlement.ts_event == 130
    assert decoded_settlement.ts_init == 140
    assert decoded_settlement.next_funding_ns == 141
    with pytest.raises(EngineBoundaryError, match="conflicting funding rates"):
        decoder(market_event("settlement-conflict", "funding_settlement", 200, {
            "instrument": str(instrument.id), "rate": "0.002", "settlement_ns": 200,
        }))


def test_native_funding_event_deduplication_rejects_conflicting_records() -> None:
    from nautilus_trader.core import UUID4
    from nautilus_trader.model import PositionAdjusted, PositionAdjustmentType, PositionId, StrategyId, TraderId

    # This test covers extraction deduplication; the neighboring replay test
    # verifies the account-side funding payment through the native engine.
    event = PositionAdjusted(
        trader_id=TraderId("TRADER-001"), strategy_id=StrategyId("test-001"),
        instrument_id=_instrument().id, position_id=PositionId("test-position"),
        account_id=AccountId.from_str("SIM-001"), adjustment_type=PositionAdjustmentType.FUNDING,
        quantity_change=None, pnl_change=Money.from_str("-0.01 USD"), reason="test extraction",
        event_id=UUID4(), ts_event=60_000_000_000, ts_init=60_000_000_000,
    )
    record = event.to_dict()
    identical = PositionAdjusted.from_dict(record)
    assert NativeFactsReader._unique_events((event, identical)) == (record,)
    conflict = PositionAdjusted.from_dict({**record, "pnl_change": "-0.02 USD"})
    with pytest.raises(EngineBoundaryError, match="conflicting facts"):
        NativeFactsReader._unique_events((event, conflict))
