Warning: truncated output (original token count: 13274)
Total output lines: 1090

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace

import pytest

import rles_research.metrics as research_metrics
from rles_research.metrics import (
    AccountFactsAccumulator,
    MetricsError,
    from_account_facts,
    series_metrics,
)
from rles_research.native_metrics import (
    NativeReportAccumulator,
    account_rows,
    native_report,
    reconcile_linear_equity,
)
from rles_research.statistics import (
    deflated_sharpe, deflated_sharpe_trial_count_sensitivity,
)

pytestmark = pytest.mark.unit


def test_dsr_trial_count_sensitivity_brackets_dependence_without_claiming_correlation() -> None:
    returns = [0.01, -0.004, 0.012, 0.0, -0.003, 0.009, 0.002, -0.006]
    trial_sharpes = [0.2, 0.5, 0.1]
    sensitivity = deflated_sharpe_trial_count_sensitivity(
        returns, trial_sharpes=trial_sharpes,
    )
    perfect_dependence, independence = sensitivity["scenarios"]
    assert sensitivity["status"] == "scenario_sensitivity_only"
    assert sensitivity["effective_trial_count_range"] == [1, 3]
    assert sensitivity["correlation_estimated"] is False
    assert sensitivity["project_historical_trials"] == "not_estimated"
    assert perfect_dependence["effective_trials"] == 1
    assert independence["effective_trials"] == 3
    assert perfect_dependence["dsr_probability"] >= independence["dsr_probability"]
    declared = deflated_sharpe(
        returns, effective_trials=3, trial_sharpes=trial_sharpes,
    )
    assert independence["dsr_probability"] == declared["dsr_probability"]


def test_account_metrics_use_period_flows_and_native_signed_funding_once() -> None:
    facts = (
        {"ts_ns": 1, "equity": 1_000.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0},
        {"ts_ns": 11, "equity": 1_100.0, "external_cashflow": 100.0,
         "turnover_notional": 200.0, "fee": 5.0, "funding": -2.0, "slippage_cost": 3.0},
        {"ts_ns": 21, "equity": 990.0, "external_cashflow": 0.0,
         "turnover_notional": 110.0, "fee": 1.0, "funding": 1.0, "slippage_cost": 1.0},
    )

    result = from_account_facts(facts, periods_per_year=2, sample_interval_ns=10)

    # The first sample's boundary deposit does not appear as trading return.
    assert result["net_return"] == pytest.approx(-0.1)
    assert result["turnover_ratio"] == pytest.approx(0.3)
    assert result["total_fee"] == pytest.approx(6.0)
    assert result["total_funding"] == pytest.approx(-1.0)
    assert result["total_slippage_cost"] == pytest.approx(4.0)
    # Net equity already includes these flows. The gross metric strips each
    # period cost once and removes the signed funding cashflow once.
    assert result["gross_return_before_reported_costs"] == pytest.approx(-0.099)


def test_account_metrics_only_compute_profit_factor_from_closed_trade_pnls() -> None:
    facts = tuple(
        {"ts_ns": index * 10, "equity": equity, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0}
        for index, equity in enumerate((1_000.0, 1_010.0, 1_005.0), start=1)
    )

    without_closed_trades = from_account_facts(
        facts, periods_per_year=1, sample_interval_ns=10
    )
    with_closed_trades = from_account_facts(
        facts, periods_per_year=1, sample_interval_ns=10,
        realized_trade_pnls=(4.0, -2.0, -2.0),
    )

    assert "trade_profit_factor" not in without_closed_trades
    assert with_closed_trades["trade_profit_factor"] == pytest.approx(1.0)
    assert with_closed_trades["closed_trade_count"] == 3


def test_account_metrics_reject_irregular_sampling() -> None:
    facts = tuple(
        {"ts_ns": timestamp, "equity": 1_000.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0}
        for timestamp in (1, 11, 22)
    )
    with pytest.raises(MetricsError, match="irregular account sampling"):
        from_account_facts(facts, periods_per_year=1, sample_interval_ns=10)


def test_disk_backed_return_metrics_match_in_memory_series_exactly() -> None:
    equities = [1_000.0]
    for index in range(4_096):
        period_return = (0.001 if index % 7 in (0, 1, 3) else -0.0007)
        equities.append(equities[-1] * (1 + period_return))
    facts = (
        {"ts_ns": index, "equity": equity, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0}
        for index, equity in enumerate(equities)
    )

    accumulator = AccountFactsAccumulator(sample_interval_ns=1)
    for fact in facts:
        accumulator.add(fact)

    replayed_returns = list(accumulator.returns)
    assert len(accumulator.returns) == len(equities) - 1
    assert not isinstance(accumulator.returns, list)
    assert accumulator.returns._file.seek(0, 2) == len(replayed_returns) * 8
    expected = series_metrics(replayed_returns, periods_per_year=365)
    result = accumulator.result(periods_per_year=365)
    assert {key: result[key] for key in expected} == expected


def test_disk_backed_tail_selection_avoids_fraction_sized_heap(monkeypatch) -> None:
    values = [-0.4, -0.3, -0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    expected = series_metrics(values, periods_per_year=1, tail_quantile=0.3)
    returns = research_metrics._DiskBackedFloatSequence()
    try:
        for value in values:
            returns.append(value)

        monkeypatch.setattr(research_metrics, "_TAIL_SORT_CHUNK", 2)
        monkeypatch.setattr(research_metrics, "_TAIL_MERGE_FAN_IN", 2)

        def reject_fraction_sized_heap(*_args, **_kwargs):
            pytest.fail("disk-backed tail selection must not call heapq.nsmallest")

        monkeypatch.setattr(research_metrics.heapq, "nsmallest", reject_fraction_sized_heap)
        actual = series_metrics(returns, periods_per_year=1, tail_quantile=0.3)
        assert actual == expected
    finally:
        returns.close()


def test_in_memory_tail_heap_uses_fixed_cap_and_spills_larger_exact_tails(monkeypatch) -> None:
    values = [((index * 37) % 101 - 50) / 100 for index in range(257)]
    expected = series_metrics(values, periods_per_year=1, tail_quantile=0.3)
    monkeypatch.setattr(research_metrics, "_TAIL_HEAP_MAX_COUNT", 2)
    monkeypatch.setattr(research_metrics, "_TAIL_SORT_CHUNK", 3)
    monkeypatch.setattr(research_metrics, "_TAIL_MERGE_FAN_IN", 2)

    def reject_uncapped_heap(*_args, **_kwargs):
        pytest.fail("tail larger than the fixed heap cap must use external selection")

    monkeypatch.setattr(research_metrics.heapq, "nsmallest", reject_uncapped_heap)

    actual = series_metrics(values, periods_per_year=1, tail_quantile=0.3)

    assert actual == expected


def test_in_memory_tail_heap_keeps_small_selection_within_fixed_cap(monkeypatch) -> None:
    values = [-0.2, 0.1, -0.1, 0.3, 0.0, 0.2]
    expected = series_metrics(values, periods_per_year=1, tail_quantile=0.1)
    original = research_metrics.heapq.nsmallest
    requested_counts = []

    def bounded_heap(count, iterable):
        requested_counts.append(count)
        assert count <= 2
        return original(count, iterable)

    monkeypatch.setattr(research_metrics, "_TAIL_HEAP_MAX_COUNT", 2)
    monkeypatch.setattr(research_metrics.heapq, "nsmallest", bounded_heap)

    actual = series_metrics(values, periods_per_year=1, tail_quantile=0.1)

    assert actual == expected
    assert requested_counts == [1]


def test_account_return_limit_rejects_before_advancing_accumulator_state() -> None:
    accumulator = AccountFactsAccumulator(sample_interval_ns=1, max_return_count=1)
    facts = (
        {"ts_ns": index, "equity": equity, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0}
        for index, equity in enumerate((100.0, 101.0, 102.0))
    )
    first, second, third = facts
    accumulator.add(first)
    accumulator.add(second)

    with pytest.raises(MetricsError, match="configured maximum return count"):
        accumulator.add(third)

    assert accumulator.count == 2
    assert accumulator.last_ts == second["ts_ns"]
    assert len(accumulator.returns) == 1


@pytest.mark.parametrize("field", ["external_cashflow", "fee"])
def test_account_aggregate_overflow_rejects_before_advancing_accumulator_state(field) -> None:
    accumulator = AccountFactsAccumulator(sample_interval_ns=1)
    first = {"ts_ns": 0, "equity": 1e308, "external_cashflow": 0.0,
             "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0,
             "slippage_cost": 0.0}
    second = {"ts_ns": 1, "equity": 1e308, "external_cashflow": 0.0,
              "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0,
              "slippage_cost": 0.0}
    third = {"ts_ns": 2, "equity": 1e308, "external_cashflow": 0.0,
             "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0,
             "slippage_cost": 0.0}
    second[field] = third[field] = 1e308
    accumulator.add(first)
    accumulator.add(second)
    spool_bytes = len(accumulator.returns) * accumulator.returns._FLOAT_BYTES

    label = field.replace("_", " ")
    with pytest.raises(MetricsError, match=f"{label} aggregate"):
        accumulator.add(third)

    assert accumulator.count == 2
    assert accumulator.last_ts == second["ts_ns"]
    assert accumulator.external_cashflow == (1e308 if field == "external_cashflow" else 0.0)
    assert accumulator.totals["fee"] == (1e308 if field == "fee" else 0.0)
    assert len(accumulator.returns) * accumulator.returns._FLOAT_BYTES == spool_bytes


def test_account_turnover_ratio_overflow_is_rejected() -> None:
    facts = (
        {"ts_ns": 0, "equity": 1e-308, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0,
         "slippage_cost": 0.0},
        {"ts_ns": 1, "equity": 1e-308, "external_cashflow": 0.0,
         "turnover_notional": 1e308, "fee": 0.0, "funding": 0.0,
         "slippage_cost": 0.0},
    )

    with pytest.raises(MetricsError, match="turnover increment"):
        from_account_facts(facts, periods_per_year=1, sample_interval_ns=1)


def test_gross_return_overflow_is_rejected_at_report_boundary() -> None:
    facts = (
        {"ts_ns": 0, "equity": 1e-308, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0,
         "slippage_cost": 0.0},
        {"ts_ns": 1, "equity": 1e-308, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 1e308, "funding": 0.0,
         "slippage_cost": 0.0},
        {"ts_ns": 2, "equity": 1e-308, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0,
         "slippage_cost": 0.0},
    )

    with pytest.raises(MetricsError, match="gross return before reported costs"):
        from_account_facts(facts, periods_per_year=1, sample_interval_ns=1)


def test_series_metrics_rejects_nonfinite_risk_adjusted_ratio() -> None:
    with pytest.raises(MetricsError, match="series metrics produced a nonfinite result"):
        series_metrics([0.1, 0.2], periods_per_year=1, risk_free_per_period=1e308)


def test_trade_profit_factor_rejects_overflowing_pnl_aggregates() -> None:
    facts = tuple(
        {"ts_ns": index, "equity": 1_000.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0,
         "slippage_cost": 0.0}
        for index in range(3)
    )

    with pytest.raises(MetricsError, match="gross profit aggregate"):
        from_account_facts(
            facts, periods_per_year=1, sample_interval_ns=1,
            realized_trade_pnls=(1e308, 1e308, -1.0),
        )


def test_quote_execution_can_report_realized_price_improvement_as_signed_slippage() -> None:
    facts = (
        {"ts_ns": 1, "equity": 1_000.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0},
        {"ts_ns": 11, "equity": 1_001.0, "external_cashflow": 0.0,
         "turnover_notional": 100.0, "fee": 0.1, "funding": 0.0, "slippage_cost": -0.2},
        {"ts_ns": 21, "equity": 1_001.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0},
    )
    with pytest.raises(MetricsError, match="invalid account values"):
        from_account_facts(facts, periods_per_year=1, sample_interval_ns=10)

    result = from_account_facts(facts, periods_per_year=1, sample_interval_ns=10,
                                allow_price_improvement=True)
    assert result["total_slippage_cost"] == pytest.approx(-0.2)


def test_native_report_reassembles_incremental_fill_and_funding_facts() -> None:
    fill = {"event_id": "fill-1", "client_order_id": "order-1", "order_side": "BUY",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    funding = {"event_id": "funding-1", "pnl_change": "-0.2 USDT"}

    def snapshot(timestamp, equity, fills=(), adjustments=()):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=equity, currency="USDT", fills=fills,
            funding=adjustments, positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=(),
        )

    report = native_report(
        (snapshot(1, 1_000.0), snapshot(11, 999.9, (fill,)),
         snapshot(21, 999.7, (), (funding,))),
        fill_references={"order-1": Decimal("100")}, sample_interval_ns=10,
    )
    compact_report = native_report(
        (snapshot(1, 1_000.0), snapshot(11, 999.9, (fill,)),
         snapshot(21, 999.7, (), (funding,))),
        fill_references={"order-1": Decimal("100")}, sample_interval_ns=10,
        include_account_rows=False,
    )

    assert report["fills"] == [fill]
    assert report["funding"] == [funding]
    assert report["metrics"]["total_fee"] == pytest.approx(0.1)
    assert report["metrics"]["total_funding"] == pytest.approx(-0.2)
    assert compact_report["metrics"] == report["metrics"]
    assert "account_…6274 tokens truncated…FILLED", "filled_qty": "1", "leaves_qty": "0"}

    def snapshot(timestamp, liquidation_orders):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal("1000"), currency="USDT", fills=(),
            funding=(), positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=liquidation_orders,
        )

    report = native_report(
        (snapshot(1, ()), snapshot(2, (initial,)), snapshot(3, (updated,))),
        fill_references={}, sample_interval_ns=1,
    )

    assert report["liquidations"] == 1
    assert report["liquidation_orders"] == [updated]


@pytest.mark.parametrize(
    "first_filled, next_filled, message",
    (("0.5", "0.4", "cumulative fill quantity regressed"),
     ("0.5", "1.1", "exceeds order quantity")),
)
def test_native_report_rejects_invalid_cumulative_liquidation_fills(
        first_filled: str, next_filled: str, message: str) -> None:
    first = {
        "client_order_id": "LIQUIDATION-BINANCE-1", "instrument_id": "BTCUSDT.BINANCE",
        "order_side": "SELL", "quantity": "1", "tags": ["LIQUIDATION_BINANCE_CLOSE"],
        "status": "PARTIALLY_FILLED", "filled_qty": first_filled,
        "leaves_qty": str(Decimal("1") - Decimal(first_filled)),
    }
    next_order = {
        **first, "status": "PARTIALLY_FILLED", "filled_qty": next_filled,
        "leaves_qty": str(Decimal("1") - Decimal(next_filled)),
    }

    def snapshot(timestamp: int, liquidation_orders: tuple[dict, ...]):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal("1000"), currency="USDT", fills=(),
            funding=(), positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=liquidation_orders,
        )

    with pytest.raises(MetricsError, match=message):
        native_report(
            (snapshot(0, ()), snapshot(1, (first,)), snapshot(2, (next_order,))),
            fill_references={}, sample_interval_ns=1,
        )


def test_native_report_orders_facts_and_liquidation_state_by_snapshot_time() -> None:
    buy = {"event_id": "buy-1", "client_order_id": "buy-order", "order_side": "BUY",
           "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    sell = {"event_id": "sell-1", "client_order_id": "sell-order", "order_side": "SELL",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    funding_early = {"event_id": "funding-early", "pnl_change": "-0.2 USDT"}
    funding_late = {"event_id": "funding-late", "pnl_change": "0.1 USDT"}
    initial_order = {
        "client_order_id": "LIQUIDATION-BINANCE-1", "instrument_id": "BTCUSDT.BINANCE",
        "order_side": "SELL", "quantity": "1", "tags": ["LIQUIDATION_BINANCE_CLOSE"],
        "status": "ACCEPTED", "filled_qty": "0", "leaves_qty": "1",
    }
    updated_order = {**initial_order, "status": "FILLED", "filled_qty": "1", "leaves_qty": "0"}

    def snapshot(timestamp, equity, *, fills=(), funding=(), liquidation_orders=()):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal(equity), currency="USDT", fills=fills,
            funding=funding, positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=liquidation_orders,
        )

    # The t=21 barrier arrives before t=11. Report facts and cumulative order
    # lifecycle state must still follow the authoritative timestamp order.
    snapshots = (
        snapshot(1, "1000"),
        snapshot(21, "999.8", fills=(sell,), funding=(funding_late,),
                 liquidation_orders=(updated_order,)),
        snapshot(11, "999.9", fills=(buy,), funding=(funding_early,),
                 liquidation_orders=(initial_order,)),
    )
    report = native_report(
        snapshots, fill_references={"buy-order": Decimal(100), "sell-order": Decimal(100)},
        sample_interval_ns=10,
    )

    assert [row["ts_ns"] for row in report["account_rows"]] == [1, 11, 21]
    assert report["fills"] == [buy, sell]
    assert report["funding"] == [funding_early, funding_late]
    assert report["liquidation_orders"] == [updated_order]
    assert report["metrics"]["total_fee"] == pytest.approx(0.2)
    assert report["metrics"]["total_funding"] == pytest.approx(-0.1)


@pytest.mark.parametrize("field, value", (("quantity", "2"),
                                          ("tags", ["LIQUIDATION_OTHER_CLOSE"])))
def test_native_report_rejects_conflicting_liquidation_order_identity(field, value) -> None:
    order = {"client_order_id": "LIQUIDATION-BINANCE-1", "instrument_id": "BTCUSDT.BINANCE",
             "order_side": "SELL", "quantity": "1", "tags": ["LIQUIDATION_BINANCE_CLOSE"],
             "status": "ACCEPTED"}
    changed = {**order, field: value}

    def snapshot(timestamp, liquidation_orders):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal("1000"), currency="USDT", fills=(),
            funding=(), positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=liquidation_orders,
        )

    with pytest.raises(MetricsError, match="liquidation order identity has conflicting facts"):
        native_report(
            (snapshot(1, (order,)), snapshot(2, (changed,)), snapshot(3, ())),
            fill_references={}, sample_interval_ns=1,
        )


def test_native_report_rejects_noncanonical_liquidation_order_identity() -> None:
    order = {"client_order_id": " LIQUIDATION-BINANCE-1 ",
             "instrument_id": "BTCUSDT.BINANCE", "order_side": "SELL", "quantity": "1",
             "tags": ["LIQUIDATION_BINANCE_CLOSE"], "status": "ACCEPTED"}
    snapshots = tuple(SimpleNamespace(
        decision_ts_ns=timestamp, equity=Decimal("1000"), currency="USDT", fills=(),
        funding=(), positions=(), orders=(), liquidation_capability=True,
        liquidation_orders=orders,
    ) for timestamp, orders in ((1, ()), (2, (order,)), (3, ())))

    with pytest.raises(MetricsError, match="native liquidation order lacks identity"):
        native_report(snapshots, fill_references={}, sample_interval_ns=1)


def test_linear_reconciliation_detects_missing_cost_or_extra_cashflow() -> None:
    snapshots, _ = _linear_ledger()
    changed = SimpleNamespace(**{**vars(snapshots[-1]), "equity": Decimal("1008.089")})
    with pytest.raises(MetricsError, match="equity mismatch"):
        reconcile_linear_equity((*snapshots[:-1], changed))
    with pytest.raises(MetricsError, match="flat event-free initial"):
        reconcile_linear_equity(snapshots[1:])


def test_reported_account_state_does_not_infer_external_cashflow() -> None:
    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), reported=False, **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("1100"), reported=True, **base),
    )

    # AccountState.reported has no transfer semantics. Without a separately
    # sourced cashflow fact, the unexplained balance change must fail closed.
    with pytest.raises(MetricsError, match="linear native equity mismatch"):
        reconcile_linear_equity(snapshots, rounding_per_event=Decimal(0))


@pytest.mark.parametrize("label", ("fill", "funding"))
@pytest.mark.parametrize("identity", (None, "", " ", " fill-1 ", 0))
def test_account_rows_reject_missing_event_identity_before_summarizing(label, identity) -> None:
    snapshots, references = _linear_ledger()
    changed = dict(snapshots[1].fills[0] if label == "fill" else snapshots[2].funding[0])
    changed["event_id"] = identity
    bad = SimpleNamespace(**{**vars(snapshots[1]),
                             "fills": (changed,) if label == "fill" else (),
                             "funding": (changed,) if label == "funding" else ()})
    with pytest.raises(MetricsError, match="lacks identity"):
        account_rows((snapshots[0], bad), fill_references=references)


@pytest.mark.parametrize("field, value", (("order_side", "UNKNOWN"), ("order_side", []),
                                         ("order_side", {}), ("last_qty", "0"),
                                         ("last_qty", "NaN"), ("last_px", "-1"),
                                         ("last_px", "broken")))
def test_account_rows_reject_invalid_native_fill_economics(field, value) -> None:
    snapshots, references = _linear_ledger()
    changed = {**snapshots[1].fills[0], field: value}
    bad = SimpleNamespace(**{**vars(snapshots[1]), "fills": (changed,)})
    with pytest.raises(MetricsError, match="native fill"):
        account_rows((snapshots[0], bad), fill_references=references)


@pytest.mark.parametrize("side", [[], {}, None, "UNKNOWN"])
def test_linear_reconciliation_rejects_invalid_native_position_side(side: object) -> None:
    fill = {"event_id": "open", "client_order_id": "open-order", "order_side": "BUY",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    position = {"side": side, "quantity": "1", "multiplier": "1", "is_inverse": False,
                "settlement_currency": "USDT"}
    base = dict(currency="USDT", funding=(), orders=(), liquidation_capability=True,
                liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), fills=(), positions=(), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("999.9"), fills=(fill,),
                         positions=(position,), mark=Decimal("100"), **base),
    )

    with pytest.raises(MetricsError, match="native linear position side and quantity are invalid"):
        reconcile_linear_equity(snapshots, rounding_per_event=Decimal(0))


def test_funding_identity_conflict_across_timestamps_is_rejected() -> None:
    snapshots, references = _linear_ledger()
    changed = {**snapshots[2].funding[0], "pnl_change": "1 USDT"}
    bad = SimpleNamespace(**{**vars(snapshots[-1]), "funding": (changed,)})
    with pytest.raises(MetricsError, match="funding identity has conflicting"):
        account_rows((*snapshots[:-1], bad), fill_references=references)


def test_snapshot_currency_change_and_invalid_timestamp_are_rejected() -> None:
    snapshots, references = _linear_ledger()
    for change, message in (({"currency": "USD"}, "conflicting currencies"),
                            ({"decision_ts_ns": True}, "timestamp")):
        bad = SimpleNamespace(**{**vars(snapshots[-1]), **change})
        with pytest.raises(MetricsError, match=message):
            account_rows((*snapshots[:-1], bad), fill_references=references)


@pytest.mark.parametrize(
    "snapshot, message",
    (
        (SimpleNamespace(decision_ts_ns=1), "lacks currency"),
        (SimpleNamespace(decision_ts_ns=1, currency="USDT"), "lacks timestamp, currency, or equity"),
        (SimpleNamespace(decision_ts_ns=1, currency="USDT", equity=float("nan")), "nonfinite native equity"),
    ),
)
def test_native_snapshot_shape_and_equity_fail_closed(snapshot, message) -> None:
    with pytest.raises(MetricsError, match=message):
        account_rows((snapshot,), fill_references={})


@pytest.mark.parametrize("snapshots", (None, 1, "not-snapshots"))
def test_native_snapshot_collection_requires_a_sequence(snapshots) -> None:
    with pytest.raises(MetricsError, match="native account snapshots required"):
        account_rows(snapshots, fill_references={})


@pytest.mark.parametrize("field, value, label", [
    ("fills", None, "fill"), ("funding", 7, "funding"),
])
def test_native_snapshot_economic_history_requires_sequences(field, value, label) -> None:
    snapshots, references = _linear_ledger()
    malformed = SimpleNamespace(**{**vars(snapshots[1]), field: value})
    with pytest.raises(MetricsError, match=f"native {label} history must be a sequence"):
        account_rows((snapshots[0], malformed), fill_references=references)


@pytest.mark.parametrize("field, value, label", [
    ("positions", None, "position"), ("orders", "not-orders", "order"),
])
def test_native_report_terminal_fact_collections_require_sequences(field, value, label) -> None:
    snapshots, references = _linear_ledger()
    malformed = SimpleNamespace(**{**vars(snapshots[-1]), field: value})
    with pytest.raises(MetricsError, match=f"native {label} history must be a sequence"):
        native_report((*snapshots[:-1], malformed), fill_references=references,
                      sample_interval_ns=10)


def test_native_report_liquidation_order_history_requires_sequence() -> None:
    snapshots, references = _linear_ledger()
    malformed = SimpleNamespace(**{**vars(snapshots[-1]), "liquidation_orders": 1})
    with pytest.raises(MetricsError, match="native liquidation order history must be a sequence"):
        native_report((*snapshots[:-1], malformed), fill_references=references,
                      sample_interval_ns=10)


@pytest.mark.parametrize("client_order_id", [None, 1, [], {}, " order-1 "])
def test_native_fill_requires_string_client_order_identity(client_order_id) -> None:
    snapshots, references = _linear_ledger()
    fill = {**snapshots[1].fills[0], "client_order_id": client_order_id}
    malformed = SimpleNamespace(**{**vars(snapshots[1]), "fills": (fill,)})
    with pytest.raises(MetricsError, match="client order identity is invalid"):
        account_rows((snapshots[0], malformed), fill_references=references)


@pytest.mark.parametrize("cashflows", [None, 1, "not-cashflows"])
def test_external_cashflow_collection_requires_sequence(cashflows) -> None:
    snapshots, _ = _linear_ledger()
    with pytest.raises(MetricsError, match="external cashflows must be a sequence"):
        reconcile_linear_equity(snapshots, external_cashflows=cashflows)


@pytest.mark.parametrize("references", [None, [], "not-a-mapping"])
def test_native_fill_references_require_mapping(references) -> None:
    snapshots, _ = _linear_ledger()
    with pytest.raises(MetricsError, match="fill references must be a mapping"):
        account_rows(snapshots, fill_references=references)

