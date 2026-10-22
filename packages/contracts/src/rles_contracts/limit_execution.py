"""Opt-in 0.5.0 LIMIT intent contracts; legacy 0.4.0 MARKET plans stay unchanged."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


def _decimal(value: Decimal, name: str, *, positive: bool = False) -> None:
    if type(value) is not Decimal or not value.is_finite():
        raise TypeError(f"{name} must be a finite Decimal")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _timestamp(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative nanosecond timestamp")


class OrderIntentKind(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    CANCEL = "CANCEL"


@dataclass(frozen=True, slots=True)
class LimitOrderIntent:
    """Versioned execution intent, separate from the legacy ExecutionPlan DTO."""

    intent_id: str
    instrument_id: str
    kind: OrderIntentKind
    signed_qty: Decimal
    decision_ts_ns: int
    effective_ts_ns: int
    limit_price: Decimal | None = None
    pending_id: str | None = None
    cancel_id: str | None = None
    expiry_ts_ns: int | None = None
    tif: str = "GTC"
    execution_schema_version: str = "0.5.0"

    def __post_init__(self) -> None:
        if self.execution_schema_version != "0.5.0":
            raise ValueError("unsupported LIMIT execution schema")
        if type(self.intent_id) is not str or not self.intent_id:
            raise ValueError("intent_id is required")
        if type(self.instrument_id) is not str or not self.instrument_id:
            raise ValueError("instrument_id is required")
        if type(self.kind) is not OrderIntentKind:
            raise TypeError("kind must be OrderIntentKind")
        if self.tif != "GTC":
            raise ValueError("0.5.0 LIMIT intents support GTC only")
        _decimal(self.signed_qty, "signed_qty")
        _timestamp(self.decision_ts_ns, "decision_ts_ns")
        _timestamp(self.effective_ts_ns, "effective_ts_ns")
        if self.effective_ts_ns <= self.decision_ts_ns:
            raise ValueError("order cannot become effective before the decision completes")
        if self.expiry_ts_ns is not None:
            _timestamp(self.expiry_ts_ns, "expiry_ts_ns")
            if self.expiry_ts_ns <= self.effective_ts_ns:
                raise ValueError("expiry must follow order effectiveness")

        if self.kind is OrderIntentKind.LIMIT:
            if self.signed_qty == 0 or self.limit_price is None:
                raise ValueError("LIMIT intent requires nonzero quantity and limit price")
            _decimal(self.limit_price, "limit_price", positive=True)
            if not self.pending_id or self.cancel_id is not None:
                raise ValueError("LIMIT intent requires pending_id and forbids cancel_id")
        elif self.kind is OrderIntentKind.CANCEL:
            if self.signed_qty != 0 or self.limit_price is not None or self.expiry_ts_ns is not None:
                raise ValueError("CANCEL intent cannot carry order economics or expiry")
            if not self.pending_id or not self.cancel_id:
                raise ValueError("CANCEL intent requires pending_id and unique cancel_id")
        else:
            if self.signed_qty == 0 or any(
                value is not None for value in (self.limit_price, self.pending_id, self.cancel_id, self.expiry_ts_ns)
            ):
                raise ValueError("MARKET intent requires quantity and no LIMIT-only fields")


@dataclass(frozen=True, slots=True)
class LimitInstrumentRules:
    instrument_id: str
    price_tick: Decimal
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal

    def __post_init__(self) -> None:
        if type(self.instrument_id) is not str or not self.instrument_id:
            raise ValueError("instrument_id is required")
        for name in ("price_tick", "qty_step", "min_qty", "min_notional"):
            _decimal(getattr(self, name), name, positive=True)

    def validate_limit(self, intent: LimitOrderIntent) -> None:
        if intent.kind is not OrderIntentKind.LIMIT or intent.instrument_id != self.instrument_id:
            raise ValueError("LIMIT intent instrument/rules mismatch")
        price = intent.limit_price
        if price is None:  # __post_init__ guarantees this; retain a local proof for type checkers.
            raise ValueError("limit price missing")
        if price % self.price_tick:
            raise ValueError("limit price violates exact price tick")
        if abs(intent.signed_qty) % self.qty_step:
            raise ValueError("quantity violates exact quantity step")
        if abs(intent.signed_qty) < self.min_qty:
            raise ValueError("quantity is below minimum quantity")
        if abs(intent.signed_qty * price) < self.min_notional:
            raise ValueError("limit notional is below instrument minimum")
