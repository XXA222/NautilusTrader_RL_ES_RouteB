from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest

from rles_research.native_metrics import (
    MetricsError,
    NativeReportAccumulator,
    account_rows,
    _external_cashflows_by_time,
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


@pytest.mark.parametrize("amount", ["1_000", "١٢٣", "+100", " 100 ", "1e2"])
def test_batch_and_online_cashflow_paths_reject_noncanonical_amount_strings(amount: str):
    flow = _flow("malformed-transfer", 11, "DEPOSIT", amount)
    snapshots = (_snapshot(1, "1000"), _snapshot(11, "1100"), _snapshot(21, "1200"))

    with pytest.raises(MetricsError, match="exact decimal"):
        native_report(
            snapshots, fill_references={}, sample_interval_ns=10,
            external_cashflows=(flow,),
        )
    with pytest.raises(MetricsError, match="exact decimal"):
        NativeReportAccumulator(
            sample_interval_ns=10, max_return_count=4,
            contract_multiplier=Decimal(1), external_cashflows=(flow,),
        )


def test_batch_and_online_cashflow_paths_normalize_decimal_sum_overflow():
    amount = Decimal("9.9E+999999")
    cashflows = (
        _flow("overflowing-transfer-a", 11, "DEPOSIT", amount),
        _flow("overflowing-transfer-b", 11, "DEPOSIT", amount),
    )
    snapshots = (_snapshot(1, "1000"), _snapshot(11, "1100"), _snapshot(21, "1200"))

    with pytest.raises(MetricsError, match="cashflow sum exceeds finite decimal range"):
        native_report(
            snapshots, fill_references={}, sample_interval_ns=10,
            external_cashflows=cashflows,
        )
    with pytest.raises(MetricsError, match="cashflow sum exceeds finite decimal range"):
        NativeReportAccumulator(
            sample_interval_ns=10, max_return_count=4,
            contract_multiplier=Decimal(1), external_cashflows=cashflows,
        )


def test_batch_and_online_fill_accounting_ignore_low_decimal_context_precision():
    fill = {
        "event_id": "precision-fill",
        "client_order_id": "precision-order",
        "order_side": "BUY",
        "last_qty": "12345",
        "last_px": "12.34",
        "commission": "0.01 USDT",
    }
    snapshots = [_snapshot(1, "100000"), _snapshot(11, "100000"), _snapshot(21, "100000")]
    snapshots[1].fills = (fill,)
    references = {"precision-order": Decimal("12.33")}

    with localcontext() as context:
        context.prec = 3
        rows = account_rows(
            snapshots, fill_references=references,
            contract_multiplier=Decimal(1),
        )
        online = NativeReportAccumulator(
            sample_interval_ns=10, max_return_count=4,
            contract_multiplier=Decimal(1),
        )
        for snapshot in snapshots:
            online.add_snapshot(snapshot, references)
        report = online.finish()

    assert rows[1]["turnover_notional"] == pytest.approx(152337.3)
    assert rows[1]["slippage_cost"] == pytest.approx(123.45)
    assert report["metrics"]["turnover_ratio"] == pytest.approx(1.523373)
    assert report["metrics"]["total_slippage_cost"] == pytest.approx(123.45)


def test_cashflow_aggregation_preserves_exact_sum_in_low_precision_context():
    cashflows = (
        _flow("large-transfer", 11, "DEPOSIT", Decimal("1E+28")),
        _flow("small-transfer", 11, "DEPOSIT", Decimal("1")),
    )
    snapshots = (_snapshot(1, "1000"), _snapshot(11, "1100"), _snapshot(21, "1200"))
    expected = Decimal("10000000000000000000000000001")

    with localcontext() as context:
        context.prec = 3
        batch, _ = _external_cashflows_by_time(cashflows, snapshots)
        online = NativeReportAccumulator(
            sample_interval_ns=10, max_return_count=4,
            contract_multiplier=Decimal(1), external_cashflows=cashflows,
        )

    assert batch[11] == expected
    assert online._cashflows_by_time[11] == expected


def test_cashflow_withdrawal_sign_preserves_digits_in_low_precision_context():
    cashflows = (_flow("large-withdrawal", 11, "WITHDRAWAL", Decimal("12345")),)
    snapshots = (_snapshot(1, "1000"), _snapshot(11, "1100"), _snapshot(21, "1200"))
    expected = Decimal("-12345")

    with localcontext() as context:
        context.prec = 3
        batch, _ = _external_cashflows_by_time(cashflows, snapshots)
        online = NativeReportAccumulator(
            sample_interval_ns=10, max_return_count=4,
            contract_multiplier=Decimal(1), external_cashflows=cashflows,
        )

    assert batch[11] == expected
    assert online._cashflows_by_time[11] == expected


def test_batch_and_online_cashflow_sum_rejects_excessive_exact_precision():
    cashflows = (
        _flow("wide-transfer", 11, "DEPOSIT", Decimal("1E+4095")),
        _flow("unit-transfer", 11, "DEPOSIT", Decimal("1")),
    )
    snapshots = (_snapshot(1, "1000"), _snapshot(11, "1100"), _snapshot(21, "1200"))

    with pytest.raises(MetricsError, match="exact decimal precision limit"):
        native_report(
            snapshots, fill_references={}, sample_interval_ns=10,
            external_cashflows=cashflows,
        )
    with pytest.raises(MetricsError, match="exact decimal precision limit"):
        NativeReportAccumulator(
            sample_interval_ns=10, max_return_count=4,
            contract_multiplier=Decimal(1), external_cashflows=cashflows,
        )


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



def test_batch_and_online_reports_reject_conflicting_same_timestamp_funding_ids():
    first_fact = {"event_id": "funding-duplicate", "pnl_change": "-0.1 USDT"}
    changed_fact = {"event_id": "funding-duplicate", "pnl_change": "-0.2 USDT"}
    first_barrier = _snapshot(11, "999.9")
    second_barrier = _snapshot(11, "999.7")
    first_barrier = SimpleNamespace(**{**vars(first_barrier), "funding": (first_fact,)})
    second_barrier = SimpleNamespace(**{**vars(second_barrier), "funding": (changed_fact,)})
    snapshots = (_snapshot(1, "1000"), first_barrier, second_barrier)

    streamed = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=2,
        contract_multiplier=Decimal(1),
    )
    streamed.add_snapshot(snapshots[0], {})
    streamed.add_snapshot(snapshots[1], {})
    with pytest.raises(MetricsError, match="funding identity has conflicting facts"):
        streamed.add_snapshot(snapshots[2], {})

    with pytest.raises(MetricsError, match="funding identity has conflicting facts"):
        native_report(
            snapshots,
            fill_references={},
            sample_interval_ns=10,
            include_account_rows=False,
            contract_multiplier=Decimal(1),
        )


@pytest.mark.parametrize("operation", ["add", "finish"])
def test_online_accumulator_cannot_continue_after_snapshot_failure(operation):
    first_fact = {"event_id": "funding-duplicate", "pnl_change": "-0.1 USDT"}
    changed_fact = {"event_id": "funding-duplicate", "pnl_change": "-0.2 USDT"}
    first_barrier = SimpleNamespace(**{
        **vars(_snapshot(11, "999.9")), "funding": (first_fact,),
    })
    conflicting_barrier = SimpleNamespace(**{
        **vars(_snapshot(11, "999.7")), "funding": (changed_fact,),
    })
    accumulator = NativeReportAccumulator(
        sample_interval_ns=10,
        max_return_count=2,
        contract_multiplier=Decimal(1),
    )
    accumulator.add_snapshot(_snapshot(1, "1000"), {})
    accumulator.add_snapshot(first_barrier, {})
    with pytest.raises(MetricsError, match="funding identity has conflicting facts"):
        accumulator.add_snapshot(conflicting_barrier, {})

    with pytest.raises(MetricsError, match="cannot continue after a failed snapshot"):
        if operation == "add":
            accumulator.add_snapshot(_snapshot(21, "999.9"), {})
        else:
            accumulator.finish()
