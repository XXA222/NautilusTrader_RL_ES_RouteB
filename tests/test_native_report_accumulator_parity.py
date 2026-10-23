from decimal import Decimal
import gc
from types import SimpleNamespace
import weakref

import pytest

from rles_research.native_metrics import (
    MetricsError,
    NativeReportAccumulator,
    native_report,
)


def _snapshot(timestamp: int, equity: str, *, capability: bool = True):
    return SimpleNamespace(
        decision_ts_ns=timestamp,
        currency="USDT",
        equity=Decimal(equity),
        positions=(),
        orders=(),
        fills=(),
        funding=(),
        liquidation_capability=capability,
        liquidation_orders=() if capability else None,
        all_orders=(),
    )


def _flow(event_id: str, timestamp: int, kind: str, amount: str):
    return {
        "event_id": event_id,
        "decision_ts_ns": timestamp,
        "currency": "USDT",
        "kind": kind,
        "amount": amount,
        "source_id": "synthetic-bank-statement",
        "source_sha256": "a" * 64,
    }


def test_online_native_summary_matches_batch_report_with_duplicate_barrier_and_cashflows():
    snapshots = (
        _snapshot(1, "1000"),
        _snapshot(1, "1000"),
        _snapshot(11, "1100"),
        _snapshot(21, "1050"),
    )
    cashflows = (
        _flow("deposit", 11, "DEPOSIT", "100"),
        _flow("withdrawal", 21, "WITHDRAWAL", "50"),
    )
    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=len(snapshots) - 1,
        contract_multiplier=Decimal(1),
        external_cashflows=cashflows,
        linear_equity_rounding=Decimal(0),
    )
    for snapshot in snapshots:
        streamed.add_snapshot(snapshot, {})

    actual = streamed.finish()
    expected = native_report(
        snapshots,
        fill_references={},
        sample_interval_ns=10,
        include_account_rows=False,
        external_cashflows=cashflows,
        linear_equity_rounding=Decimal(0),
        contract_multiplier=Decimal(1),
    )

    assert actual == expected
    assert actual["metrics"]["net_return"] == pytest.approx(0.0)
    assert actual["equity_reconciliation"]["net_external_cashflow"] == "50"
    assert actual["liquidations_observed"] is True


def test_online_native_summary_matches_batch_when_liquidation_history_is_unavailable():
    snapshots = (
        _snapshot(1, "1000"),
        _snapshot(11, "1000", capability=False),
        _snapshot(21, "1000"),
    )
    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=len(snapshots) - 1,
        contract_multiplier=Decimal(1),
        linear_equity_rounding=Decimal(0),
    )
    for snapshot in snapshots:
        streamed.add_snapshot(snapshot, {})

    actual = streamed.finish()
    expected = native_report(
        snapshots,
        fill_references={},
        sample_interval_ns=10,
        include_account_rows=False,
        contract_multiplier=Decimal(1),
        linear_equity_rounding=Decimal(0),
    )

    assert actual == expected
    assert actual["liquidations"] is None
    assert actual["liquidations_observed"] is False
    assert actual["liquidation_orders"] is None


def test_online_native_summary_rejects_cashflow_at_ambiguous_duplicate_barrier():
    snapshots = (
        _snapshot(1, "1000"),
        _snapshot(11, "1100"),
        _snapshot(11, "1100"),
        _snapshot(21, "1100"),
    )
    cashflows = (_flow("deposit", 11, "DEPOSIT", "100"),)

    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=len(snapshots) - 1,
        contract_multiplier=Decimal(1),
        external_cashflows=cashflows,
        linear_equity_rounding=Decimal(0),
    )
    with pytest.raises(MetricsError, match="duplicate native barriers"):
        for snapshot in snapshots:
            streamed.add_snapshot(snapshot, {})
        streamed.finish()

    with pytest.raises(MetricsError, match="duplicate native barriers"):
        native_report(
            snapshots,
            fill_references={},
            sample_interval_ns=10,
            include_account_rows=False,
            external_cashflows=cashflows,
            linear_equity_rounding=Decimal(0),
            contract_multiplier=Decimal(1),
        )


def test_online_native_summary_matches_batch_exact_tail_metrics():
    snapshots = tuple(
        _snapshot(index * 10, str(Decimal(1000) + Decimal((index * 17) % 9 * 5 - index)))
        for index in range(41)
    )
    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=len(snapshots) - 1,
        contract_multiplier=Decimal(1),
    )
    for snapshot in snapshots:
        streamed.add_snapshot(snapshot, {})

    actual = streamed.finish()
    expected = native_report(
        snapshots,
        fill_references={},
        sample_interval_ns=10,
        include_account_rows=False,
        contract_multiplier=Decimal(1),
    )

    assert actual["metrics"] == expected["metrics"]
    assert actual["metrics"]["observations"] == 40
    assert actual["metrics"]["conditional_value_at_risk"] == expected["metrics"][
        "conditional_value_at_risk"
    ]


def test_online_summary_merges_same_timestamp_fill_and_funding_facts():
    snapshots = (
        _snapshot(0, "1000"),
        SimpleNamespace(
            **{**_snapshot(10, "1008").__dict__,
               "fills": ({
                   "event_id": "buy-1", "client_order_id": "buy-order",
                   "order_side": "BUY", "last_qty": "1", "last_px": "10",
                   "commission": "1 USDT",
               },)},
        ),
        SimpleNamespace(
            **{**_snapshot(10, "1008").__dict__,
               "funding": ({"event_id": "funding-1", "pnl_change": "-0.5 USDT"},)},
        ),
        _snapshot(20, "1007.5"),
    )
    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=2,
        contract_multiplier=Decimal(1),
    )
    for snapshot in snapshots:
        streamed.add_snapshot(snapshot, {"buy-order": Decimal(10)})

    expected = native_report(
        snapshots,
        fill_references={"buy-order": Decimal(10)},
        sample_interval_ns=10,
        include_account_rows=False,
        contract_multiplier=Decimal(1),
    )

    assert streamed.finish() == expected


def test_online_summary_releases_completed_snapshot_groups():
    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=2,
        contract_multiplier=Decimal(1),
    )
    first = type("WeakrefableSnapshot", (), {})()
    vars(first).update(vars(_snapshot(0, "1000")))
    first_ref = weakref.ref(first)
    streamed.add_snapshot(first, {})
    del first
    second = _snapshot(10, "1001")
    streamed.add_snapshot(second, {})
    gc.collect()

    assert first_ref() is None

    streamed.add_snapshot(_snapshot(20, "1002"), {})
    streamed.finish()


def test_online_summary_requires_reference_for_each_current_fill():
    snapshot = _snapshot(0, "1000")
    snapshot.fills = ({
        "event_id": "fill-1", "client_order_id": "order-1", "order_side": "BUY",
        "last_qty": "1", "last_px": "1", "commission": "0 USDT",
    },)
    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=1,
        contract_multiplier=Decimal(1),
    )

    streamed.add_snapshot(snapshot, {"unrelated-order": Decimal(1)})
    with pytest.raises(MetricsError, match="reference"):
        streamed.add_snapshot(_snapshot(10, "1000"), {})


def test_online_summary_reconciles_open_and_closed_linear_fill_barriers():
    quantity = Decimal("0.1")
    initial = _snapshot(0, "1000")
    opened = _snapshot(10, "1000.9")
    opened.mark = Decimal(110)
    opened.positions = ({
        "is_inverse": False,
        "multiplier": Decimal(1),
        "settlement_currency": "USDT",
        "side": "LONG",
        "quantity": quantity,
    },)
    opened.fills = ({
        "event_id": "buy-1", "client_order_id": "buy-order", "order_side": "BUY",
        "last_qty": str(quantity), "last_px": "100", "commission": "0.1 USDT",
    },)
    closed = _snapshot(20, "1000.8")
    closed.fills = ({
        "event_id": "sell-1", "client_order_id": "sell-order", "order_side": "SELL",
        "last_qty": str(quantity), "last_px": "110", "commission": "0.1 USDT",
    },)
    snapshots = (initial, opened, closed)
    references = {"buy-order": Decimal(100), "sell-order": Decimal(110)}
    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=2,
        contract_multiplier=Decimal(1),
        linear_equity_rounding=Decimal(0),
    )
    for snapshot in snapshots:
        streamed.add_snapshot(snapshot, references)

    expected = native_report(
        snapshots,
        fill_references=references,
        sample_interval_ns=10,
        include_account_rows=False,
        linear_equity_rounding=Decimal(0),
        contract_multiplier=Decimal(1),
    )

    assert streamed.finish() == expected
