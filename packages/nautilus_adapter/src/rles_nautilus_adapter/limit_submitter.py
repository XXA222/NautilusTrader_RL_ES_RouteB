"""Opt-in native 0.5.0 LIMIT submit/cancel/expiry adapter.

This adapter only translates frozen intents to Nautilus public order APIs. The
engine remains authoritative for order events, fills, fees, positions and equity.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any

from rles_contracts.limit_execution import (
    LimitInstrumentRules,
    LimitOrderIntent,
    OrderIntentKind,
)

from .session import EngineBoundaryError


class NativeLimitSubmitter:
    """Submit versioned LIMIT intents through a bound native Strategy.

    An intent remains GTC unless its explicit expiry timestamp maps it to
    Nautilus GTD. Expiry is then processed by the native engine clock.
    """

    def __init__(self, *, rules: LimitInstrumentRules, quantity_unit: str) -> None:
        if not quantity_unit:
            raise ValueError("frozen LIMIT quantity unit required")
        self._rules = rules
        self._quantity_unit = quantity_unit
        self._engine: Any = None
        self._strategy: Any = None
        self._used_intents: set[str] = set()
        self._orders_by_pending_id: dict[str, str] = {}
        self._expiry_by_pending_id: dict[str, int | None] = {}
        self._cancel_alerts: set[str] = set()

    def bind(self, engine: Any, strategy: Any) -> None:
        from nautilus_trader.backtest import BacktestEngine
        from nautilus_trader.model import InstrumentId
        from nautilus_trader.trading import Strategy

        if self._engine is not None:
            raise EngineBoundaryError("native LIMIT submitter cannot be rebound across episodes")
        if not isinstance(engine, BacktestEngine) or not isinstance(strategy, Strategy):
            raise EngineBoundaryError("official native engine and strategy required")
        instrument = engine.cache.instrument(InstrumentId.from_str(self._rules.instrument_id))
        if instrument is None:
            raise EngineBoundaryError("frozen LIMIT instrument absent from native cache")
        if (instrument.price_increment.as_decimal() != self._rules.price_tick
                or instrument.size_increment.as_decimal() != self._rules.qty_step
                or str(instrument.base_currency) != self._quantity_unit):
            raise EngineBoundaryError("native instrument differs from frozen LIMIT rules")
        if strategy.trader_id != engine.trader_id:
            raise EngineBoundaryError("strategy is not registered with native trader")
        self._engine, self._strategy = engine, strategy

    def __call__(self, engine: Any, intent: LimitOrderIntent, *, decision_ts_ns: int) -> None:
        from nautilus_trader.model import (
            ClientOrderId,
            InstrumentId,
            OrderSide,
            Price,
            Quantity,
        )
        from nautilus_trader.model import TimeInForce as NativeTimeInForce

        self._require_bound(engine, intent, decision_ts_ns)
        if intent.intent_id in self._used_intents:
            raise EngineBoundaryError("native LIMIT intent id already used")
        if intent.kind is OrderIntentKind.CANCEL:
            self._cancel(engine, intent)
            self._used_intents.add(intent.intent_id)
            return
        if intent.kind is not OrderIntentKind.LIMIT:
            raise EngineBoundaryError("native LIMIT adapter accepts LIMIT and CANCEL intents only")
        if intent.limit_price is None or intent.pending_id is None:
            raise EngineBoundaryError("native LIMIT intent is missing price or pending identity")
        try:
            self._rules.validate_limit(intent)
            instrument_id = InstrumentId.from_str(intent.instrument_id)
            instrument = engine.cache.instrument(instrument_id)
            quantity = Quantity.from_decimal_dp(abs(intent.signed_qty), instrument.size_precision)
            price = Price.from_decimal_dp(intent.limit_price, instrument.price_precision)
        except (TypeError, ValueError, OverflowError, AttributeError) as exc:
            raise EngineBoundaryError("LIMIT intent violates native instrument representation") from exc
        if quantity.as_decimal() != abs(intent.signed_qty) or price.as_decimal() != intent.limit_price:
            raise EngineBoundaryError("native LIMIT conversion would round frozen intent")
        client_order_id = ClientOrderId(
            "RLESL-" + sha256(intent.intent_id.encode("utf-8")).hexdigest()[:26],
        )
        if engine.cache.order(client_order_id) is not None:
            raise EngineBoundaryError("native LIMIT client order identity collision")
        order = self._strategy.order_factory.limit(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY if intent.signed_qty > 0 else OrderSide.SELL,
            quantity=quantity,
            price=price,
            time_in_force=(NativeTimeInForce.GTD if intent.expiry_ts_ns is not None
                           else NativeTimeInForce.GTC),
            expire_time=intent.expiry_ts_ns,
            client_order_id=client_order_id,
        )
        self._used_intents.add(intent.intent_id)
        self._orders_by_pending_id[intent.pending_id] = str(client_order_id)
        self._expiry_by_pending_id[intent.pending_id] = intent.expiry_ts_ns
        self._strategy.submit_order(order)

    def _require_bound(self, engine: Any, intent: LimitOrderIntent, decision_ts_ns: int) -> None:
        if self._engine is None or engine is not self._engine or not self._strategy.is_running():
            raise EngineBoundaryError("native LIMIT submit requires its bound running episode")
        if type(intent) is not LimitOrderIntent:
            raise EngineBoundaryError("native LIMIT submit requires frozen 0.5.0 intent")
        if decision_ts_ns != intent.decision_ts_ns:
            raise EngineBoundaryError("LIMIT intent decision timestamp differs from native barrier")
        if intent.effective_ts_ns <= decision_ts_ns:
            raise EngineBoundaryError("LIMIT intent must become effective after its decision barrier")

    def _cancel(self, engine: Any, intent: LimitOrderIntent) -> None:
        from nautilus_trader.model import ClientOrderId, InstrumentId

        if intent.pending_id is None:
            raise EngineBoundaryError("native LIMIT cancel is missing pending identity")
        if intent.cancel_id is None:
            raise EngineBoundaryError("native LIMIT cancel is missing cancel identity")
        client_id = self._orders_by_pending_id.get(intent.pending_id)
        if client_id is None:
            raise EngineBoundaryError("native LIMIT cancel references unknown pending id")
        order = engine.cache.order(ClientOrderId(client_id))
        if (order is None or order.instrument_id != InstrumentId.from_str(intent.instrument_id)
                or order.strategy_id != self._strategy.strategy_id or order.is_closed):
            raise EngineBoundaryError("LIMIT cancel target is not this strategy's live native order")
        expiry_ts_ns = self._expiry_by_pending_id.get(intent.pending_id)
        # Expiry wins an exact timestamp tie, matching the frozen 0.5.0 rule.
        if expiry_ts_ns is not None and expiry_ts_ns <= intent.effective_ts_ns:
            return
        alert_name = "RLESL-CANCEL-" + sha256(intent.cancel_id.encode("utf-8")).hexdigest()[:24]
        if alert_name in self._cancel_alerts:
            raise EngineBoundaryError("native LIMIT cancel alert identity already used")

        def cancel_if_open(_event: Any) -> None:
            # The Rust engine invokes callbacks while holding its mutable
            # engine borrow. Querying cache here would re-enter that borrow;
            # route the command by the already-validated immutable client ID.
            self._strategy.cancel_order(ClientOrderId(client_id))

        self._strategy.clock.set_time_alert_ns(
            alert_name, intent.effective_ts_ns, callback=cancel_if_open, allow_past=False,
        )
        self._cancel_alerts.add(alert_name)

    def cancel_open_orders(self, engine: Any, client_order_ids: tuple[str, ...]) -> bool:
        """Immediately cancel a terminal set of this adapter's live LIMIT orders.

        Return ``False`` without side effects when the complete set belongs to
        another submitter, allowing EngineSession to use its normal cancel path.
        """
        from nautilus_trader.model import ClientOrderId

        if self._engine is None or engine is not self._engine or not self._strategy.is_running():
            raise EngineBoundaryError("terminal LIMIT cancel requires its bound running episode")
        if (type(client_order_ids) is not tuple or not client_order_ids
                or len(set(client_order_ids)) != len(client_order_ids)
                or any(type(value) is not str or not value for value in client_order_ids)):
            raise EngineBoundaryError("terminal LIMIT cancel requires distinct native order IDs")
        owned_ids = set(self._orders_by_pending_id.values())
        if not set(client_order_ids) <= owned_ids:
            return False
        orders = []
        for value in client_order_ids:
            order = engine.cache.order(ClientOrderId(value))
            if (order is None or order.strategy_id != self._strategy.strategy_id
                    or str(order.instrument_id) != self._rules.instrument_id or order.is_closed):
                raise EngineBoundaryError("terminal LIMIT cancel target is not a live owned order")
            orders.append(ClientOrderId(value))
        for client_order_id in orders:
            self._strategy.cancel_order(client_order_id)
        return True
