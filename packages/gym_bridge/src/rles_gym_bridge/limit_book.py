"""Deterministic OHLC LIMIT lifecycle prototype; not native parity evidence."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from rles_contracts.limit_execution import (
    LimitInstrumentRules,
    LimitOrderIntent,
    OrderIntentKind,
)


class LimitOrderState(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    AMBIGUOUS = "AMBIGUOUS"


class LimitEventKind(StrEnum):
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    AMBIGUOUS_PATH = "AMBIGUOUS_PATH"
    CANCEL_IGNORED_CLOSED = "CANCEL_IGNORED_CLOSED"
    TERMINAL_CANCELLED = "TERMINAL_CANCELLED"


@dataclass(frozen=True, slots=True)
class PriceBar:
    instrument_id: str
    start_ts_ns: int
    end_ts_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        if not self.instrument_id or type(self.instrument_id) is not str:
            raise ValueError("bar instrument_id is required")
        if any(type(ts) is not int or ts < 0 for ts in (self.start_ts_ns, self.end_ts_ns)):
            raise ValueError("bar timestamps must be nonnegative integers")
        if self.end_ts_ns <= self.start_ts_ns:
            raise ValueError("bar end must follow bar start")
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if type(value) is not Decimal or not value.is_finite() or value <= 0:
                raise ValueError(f"bar {name} must be a positive finite Decimal")
        if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close, self.high):
            raise ValueError("invalid OHLC ordering")


@dataclass(frozen=True, slots=True)
class LimitEvent:
    sequence: int
    event_kind: LimitEventKind
    pending_id: str
    ts_ns: int
    signed_qty: Decimal = Decimal(0)
    price: Decimal | None = None
    fee_class: str | None = None
    reason: str = ""


@dataclass(slots=True)
class _PendingLimit:
    intent: LimitOrderIntent
    state: LimitOrderState = LimitOrderState.PENDING
    cancel_effective_ts_ns: int | None = None


class LimitOrderBook:
    """Whole-order OHLC trigger rules with fail-closed ambiguous intrabar races.

    This module deliberately does not model account balances, fees, funding,
    queue position, partial fills, or Nautilus events. A fill is only a matching
    event, not proof of executable economics.
    """

    def __init__(self, rules: LimitInstrumentRules) -> None:
        if type(rules) is not LimitInstrumentRules:
            raise TypeError("frozen LimitInstrumentRules required")
        self.rules = rules
        self._orders: dict[str, _PendingLimit] = {}
        self._intent_ids: set[str] = set()
        self._cancel_ids: set[str] = set()
        self._events: list[LimitEvent] = []
        self._last_bar_end_ns: int | None = None
        self._closed = False

    @property
    def events(self) -> tuple[LimitEvent, ...]:
        return tuple(self._events)

    @property
    def open_orders(self) -> tuple[str, ...]:
        return tuple(sorted(
            key for key, order in self._orders.items()
            if order.state is LimitOrderState.PENDING
        ))

    def state(self, pending_id: str) -> LimitOrderState:
        return self._orders[pending_id].state

    def submit(self, intent: LimitOrderIntent) -> None:
        self._ensure_open()
        if type(intent) is not LimitOrderIntent or intent.kind is not OrderIntentKind.LIMIT:
            raise TypeError("LIMIT intent required")
        self.rules.validate_limit(intent)
        pending_id = intent.pending_id
        if pending_id is None:
            raise ValueError("LIMIT pending identity is missing")
        if intent.intent_id in self._intent_ids or pending_id in self._orders:
            raise ValueError("duplicate LIMIT intent or pending identity")
        self._intent_ids.add(intent.intent_id)
        self._orders[pending_id] = _PendingLimit(intent)
        self._emit(LimitEventKind.ACCEPTED, pending_id, intent.decision_ts_ns,
                   reason="accepted; not active before effective_ts_ns")

    def cancel(self, intent: LimitOrderIntent) -> None:
        self._ensure_open()
        if type(intent) is not LimitOrderIntent or intent.kind is not OrderIntentKind.CANCEL:
            raise TypeError("CANCEL intent required")
        if intent.instrument_id != self.rules.instrument_id:
            raise ValueError("cancel instrument/rules mismatch")
        pending_id, cancel_id = intent.pending_id, intent.cancel_id
        if pending_id is None or cancel_id is None:
            raise ValueError("cancel identities are missing")
        if intent.intent_id in self._intent_ids or cancel_id in self._cancel_ids:
            raise ValueError("duplicate cancel identity")
        order = self._orders.get(pending_id)
        if order is None:
            raise ValueError("cancel target is unknown")
        self._intent_ids.add(intent.intent_id)
        self._cancel_ids.add(cancel_id)
        if order.state is not LimitOrderState.PENDING:
            self._emit(LimitEventKind.CANCEL_IGNORED_CLOSED, pending_id,
                       intent.effective_ts_ns, reason=f"order already {order.state.value}")
            return
        if order.cancel_effective_ts_ns is not None:
            raise ValueError("pending order already has a scheduled cancellation")
        order.cancel_effective_ts_ns = intent.effective_ts_ns

    def advance(self, bar: PriceBar) -> tuple[LimitEvent, ...]:
        self._ensure_open()
        if type(bar) is not PriceBar or bar.instrument_id != self.rules.instrument_id:
            raise ValueError("bar instrument mismatch")
        if self._last_bar_end_ns is not None and bar.start_ts_ns < self._last_bar_end_ns:
            raise ValueError("bars must be ordered and non-overlapping")
        first_new = len(self._events)
        for pending_id in sorted(self._orders):
            order = self._orders[pending_id]
            if order.state is not LimitOrderState.PENDING:
                continue
            intent = order.intent
            if intent.expiry_ts_ns is not None and intent.expiry_ts_ns <= bar.start_ts_ns:
                order.state = LimitOrderState.EXPIRED
                self._emit(LimitEventKind.EXPIRED, pending_id, intent.expiry_ts_ns,
                           reason="GTC order reached explicit expiry")
                continue
            cancel_ts = order.cancel_effective_ts_ns
            if cancel_ts is not None and cancel_ts <= bar.start_ts_ns:
                order.state = LimitOrderState.CANCELLED
                self._emit(LimitEventKind.CANCELLED, pending_id, cancel_ts,
                           reason="cancel effective before bar")
                continue

            touched = self._touches(intent, bar)
            if bar.start_ts_ns < intent.effective_ts_ns:
                # A cancellation that is effective strictly before activation
                # removes the order before it can participate in this bar.
                # Resolve that known ordering even when OHLC contains a touch;
                # only a cancel at/after activation can race with an eligible
                # trigger here.
                if (cancel_ts is not None and cancel_ts < intent.effective_ts_ns
                        and cancel_ts < bar.end_ts_ns):
                    order.state = LimitOrderState.CANCELLED
                    self._emit(LimitEventKind.CANCELLED, pending_id, cancel_ts,
                               reason="cancel effective before order activation")
                    continue
                # If an OHLC bar overlaps activation, it cannot prove whether a
                # pre-activation or post-activation touch occurred.
                if bar.end_ts_ns > intent.effective_ts_ns and touched:
                    order.state = LimitOrderState.AMBIGUOUS
                    self._emit(LimitEventKind.AMBIGUOUS_PATH, pending_id,
                               intent.effective_ts_ns,
                               reason="bar straddles activation and contains a trigger")
                elif cancel_ts is not None and cancel_ts < bar.end_ts_ns:
                    order.state = LimitOrderState.CANCELLED
                    self._emit(LimitEventKind.CANCELLED, pending_id, cancel_ts,
                               reason="cancel completed before any eligible trigger")
                elif intent.expiry_ts_ns is not None and intent.expiry_ts_ns < bar.end_ts_ns:
                    order.state = LimitOrderState.EXPIRED
                    self._emit(LimitEventKind.EXPIRED, pending_id, intent.expiry_ts_ns,
                               reason="expired before any eligible trigger")
                continue

            expiry_ts = intent.expiry_ts_ns
            terminal = self._first_terminal_in_bar(
                cancel_ts, expiry_ts, bar.start_ts_ns, bar.end_ts_ns,
            )
            limit_price = intent.limit_price
            if limit_price is None:
                raise ValueError("pending LIMIT price is missing")
            marketable_at_open = (
                bar.open <= limit_price if intent.signed_qty > 0
                else bar.open >= limit_price
            )
            # An order already marketable at the opening price triggers at the
            # bar boundary, before any later intrabar cancel or expiry.
            if touched and marketable_at_open:
                order.state = LimitOrderState.FILLED
                self._emit(
                    LimitEventKind.FILLED, pending_id, bar.start_ts_ns,
                    signed_qty=intent.signed_qty, price=intent.limit_price,
                    fee_class="TAKER",
                    reason="marketable at bar open; no queue or price improvement modeled",
                )
            elif touched and terminal is not None:
                terminal_ts, terminal_kind = terminal
                order.state = LimitOrderState.AMBIGUOUS
                self._emit(
                    LimitEventKind.AMBIGUOUS_PATH, pending_id, terminal_ts,
                    reason=f"trigger and {terminal_kind} race within one path-unknown bar",
                )
                continue
            elif touched:
                order.state = LimitOrderState.FILLED
                self._emit(
                    LimitEventKind.FILLED, pending_id, bar.start_ts_ns,
                    signed_qty=intent.signed_qty, price=intent.limit_price,
                    fee_class="SCENARIO_REQUIRED",
                    reason="whole quantity at limit; no queue or price improvement modeled",
                )
            elif terminal is not None:
                terminal_ts, terminal_kind = terminal
                if terminal_kind == "expiry":
                    order.state = LimitOrderState.EXPIRED
                    self._emit(LimitEventKind.EXPIRED, pending_id, terminal_ts,
                               reason="GTC order reached explicit expiry")
                else:
                    order.state = LimitOrderState.CANCELLED
                    self._emit(LimitEventKind.CANCELLED, pending_id, terminal_ts,
                               reason="no trigger before intrabar cancellation")
        self._last_bar_end_ns = bar.end_ts_ns
        return tuple(self._events[first_new:])

    def terminate(self, timestamp_ns: int) -> tuple[LimitEvent, ...]:
        self._ensure_open()
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("terminal timestamp must be nonnegative")
        self._closed = True
        first_new = len(self._events)
        for pending_id in self.open_orders:
            order = self._orders[pending_id]
            order.state = LimitOrderState.CANCELLED
            self._emit(LimitEventKind.TERMINAL_CANCELLED, pending_id, timestamp_ns,
                       reason="cancel unfilled order before episode-end account handling")
        return tuple(self._events[first_new:])

    @staticmethod
    def _touches(intent: LimitOrderIntent, bar: PriceBar) -> bool:
        limit = intent.limit_price
        if limit is None:
            return False
        return bar.low <= limit if intent.signed_qty > 0 else bar.high >= limit

    @staticmethod
    def _first_terminal_in_bar(
        cancel_ts: int | None, expiry_ts: int | None, start_ts: int, end_ts: int,
    ) -> tuple[int, str] | None:
        candidates: list[tuple[int, str]] = []
        if expiry_ts is not None and start_ts < expiry_ts < end_ts:
            candidates.append((expiry_ts, "expiry"))
        if cancel_ts is not None and start_ts < cancel_ts < end_ts:
            candidates.append((cancel_ts, "cancel"))
        # Expiry wins a timestamp tie: a cancel cannot revive an order whose
        # frozen lifetime has already ended.
        return min(candidates, key=lambda item: (item[0], item[1] != "expiry"),
                   default=None)

    def _emit(
        self, kind: LimitEventKind, pending_id: str, ts_ns: int, *,
        signed_qty: Decimal = Decimal(0), price: Decimal | None = None,
        fee_class: str | None = None, reason: str = "",
    ) -> None:
        self._events.append(LimitEvent(
            len(self._events), kind, pending_id, ts_ns,
            signed_qty, price, fee_class, reason,
        ))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("episode limit book is terminal")
