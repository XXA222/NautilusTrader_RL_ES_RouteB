from decimal import Decimal, Inexact, Rounded, localcontext
from types import SimpleNamespace

import pytest

from rles_research.native_metrics import (
    MetricsError,
    _add_decimal_exact,
    _multiply_decimal_exact,
    iter_account_rows,
    reconcile_linear_equity,
)


def _snapshot(timestamp: int, equity: str, *, fills=(), positions=(), mark=None):
    return SimpleNamespace(
        decision_ts_ns=timestamp,
        currency="USDT",
        equity=Decimal(equity),
        positions=positions,
        orders=(),
        fills=fills,
        funding=(),
        mark=mark,
    )


def _fill(event_id: str, order_id: str, side: str, quantity: str, price: str):
    return {
        "event_id": event_id,
        "client_order_id": order_id,
        "order_side": side,
        "last_qty": quantity,
        "last_px": price,
        "commission": "0 USDT",
    }


def test_exact_decimal_helpers_preserve_subnormals_and_clear_inherited_flags():
    tiny = Decimal("1e-1000026")
    with localcontext() as context:
        context.prec = 3
        context.traps[Inexact] = False
        context.traps[Rounded] = False
        _ = Decimal(1) / Decimal(3)
        assert context.flags[Inexact] and context.flags[Rounded]
        assert _add_decimal_exact(tiny, tiny) == Decimal("2e-1000026")
        assert _multiply_decimal_exact(tiny, Decimal(2)) == Decimal("2e-1000026")


def test_exact_decimal_helpers_reject_context_precision_loss():
    with pytest.raises(MetricsError, match="would round"):
        _add_decimal_exact(Decimal("1e100"), Decimal(1))


def test_fill_summary_arithmetic_stays_exact_under_low_precision_context():
    snapshot = _snapshot(
        1,
        "1000",
        fills=(_fill("fill-1", "order-1", "BUY", "0.123456789", "1.000000001"),),
    )
    with localcontext() as context:
        context.prec = 3
        rows = list(iter_account_rows(
            (snapshot,),
            fill_references={"order-1": Decimal("1")},
        ))

    assert rows[0]["turnover_notional"] == pytest.approx(0.123456789123456789)
    assert rows[0]["slippage_cost"] == pytest.approx(0.000000000123456789)


def test_linear_equity_reconciliation_is_exact_under_low_precision_context():
    quantity = "0.123456789"
    opened = _snapshot(
        1,
        "1000.123456789",
        fills=(_fill("buy-1", "buy-order", "BUY", quantity, "100"),),
        positions=({
            "is_inverse": False,
            "multiplier": Decimal(1),
            "settlement_currency": "USDT",
            "side": "LONG",
            "quantity": Decimal(quantity),
        },),
        mark=Decimal("101"),
    )
    snapshots = (
        _snapshot(0, "1000"),
        opened,
        _snapshot(
            2,
            "1000.123456789",
            fills=(_fill("sell-1", "sell-order", "SELL", quantity, "101"),),
        ),
    )
    with localcontext() as context:
        context.prec = 3
        result = reconcile_linear_equity(snapshots, rounding_per_event=Decimal(0))

    assert result["checked_open_barriers"] == 1
    assert result["checked_flat_barriers"] == 2
    assert result["rows"][1]["expected_equity"] == "1000.123456789"
    assert result["rows"][2]["expected_equity"] == "1000.123456789"
