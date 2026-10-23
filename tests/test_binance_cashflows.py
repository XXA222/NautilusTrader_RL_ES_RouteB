from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from rles_research.binance_cashflows import (
    BinanceIncomePage,
    import_binance_usdm_transfer_income,
    import_binance_usdm_transfer_income_pages,
)
from rles_research.metrics import MetricsError
from rles_research.native_metrics import native_report

pytestmark = pytest.mark.unit


def test_binance_transfer_export_maps_to_native_cashflow_facts_and_reconciles() -> None:
    payload = json.dumps([
        {"incomeType": "TRANSFER", "income": "125.50", "asset": "USDT",
         "time": 15, "tranId": 9001},
        {"incomeType": "TRANSFER", "income": "-25.25", "asset": "USDT",
         "time": 25, "tranId": 9002},
    ], separators=(",", ":"))
    imported = import_binance_usdm_transfer_income(
        payload, account_scope_id="subaccount-7", currency="USDT",
        sample_barriers_ns=(10_000_000, 20_000_000, 30_000_000),
    )

    assert imported.source_sha256 == hashlib.sha256(payload.encode()).hexdigest()
    assert imported.source_id.endswith(imported.source_sha256)
    assert [fact["kind"] for fact in imported.facts] == ["DEPOSIT", "WITHDRAWAL"]
    assert [fact["amount"] for fact in imported.facts] == ["125.50", "25.25"]
    assert [fact["decision_ts_ns"] for fact in imported.facts] == [20_000_000, 30_000_000]
    assert imported.event_time_range_ns == (15_000_000, 25_000_000)

    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = tuple(
        SimpleNamespace(decision_ts_ns=timestamp, equity=Decimal(equity), **base)
        for timestamp, equity in ((10_000_000, "1000"), (20_000_000, "1125.50"),
                                  (30_000_000, "1100.25"))
    )
    report = native_report(
        snapshots, fill_references={}, sample_interval_ns=10_000_000,
        external_cashflows=imported.facts, linear_equity_rounding=Decimal(0),
    )
    assert report["equity_reconciliation"]["net_external_cashflow"] == "100.25"
    assert report["equity_reconciliation"]["max_abs_residual"] == "0"


@pytest.mark.parametrize("payload", [
    b"{bad json",
    b"{}",
    b'[{"incomeType":"COMMISSION","income":"-1","asset":"USDT","time":15,"tranId":1}]',
    b'[{"incomeType":"TRANSFER","income":"1","asset":"BTC","time":15,"tranId":1}]',
    b'[{"incomeType":"TRANSFER","income":"0","asset":"USDT","time":15,"tranId":1}]',
    b'[{"incomeType":"TRANSFER","income":0.10000000000000001,"asset":"USDT","time":15,"tranId":1}]',
    b'[{"incomeType":"TRANSFER","income":" 1.00","asset":"USDT","time":15,"tranId":1}]',
    b'[{"incomeType":"TRANSFER","income":"1_0","asset":"USDT","time":15,"tranId":1}]',
    '[{"incomeType":"TRANSFER","income":"١","asset":"USDT","time":15,"tranId":1}]'.encode(),
    b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT","time":5,"tranId":1}]',
])
def test_binance_transfer_import_fails_closed_on_bad_source_or_alignment(payload: bytes) -> None:
    with pytest.raises(MetricsError):
        import_binance_usdm_transfer_income(
            payload, account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10_000_000, 20_000_000, 30_000_000),
        )


def test_binance_transfer_import_normalizes_unencodable_text_payload() -> None:
    with pytest.raises(MetricsError, match="not valid UTF-8 text"):
        import_binance_usdm_transfer_income(
            "[\ud800]", account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10, 20),
        )


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize(
    ("identity_field", "value"),
    [
        ("account_scope_id", " account-1"),
        ("account_scope_id", "account-1 "),
        ("account_scope_id", "   "),
        ("currency", " USDT"),
        ("currency", "USDT "),
        ("currency", "   "),
    ],
)
def test_binance_transfer_import_rejects_noncanonical_scope_and_currency(
    paged: bool, identity_field: str, value: str,
) -> None:
    scope = value if identity_field == "account_scope_id" else "account-1"
    currency = value if identity_field == "currency" else "USDT"
    row = {"incomeType": "TRANSFER", "income": "1", "asset": currency,
           "time": 15, "tranId": 44}
    payload = json.dumps([row], separators=(",", ":")).encode("utf-8")
    message = ("account scope ID must be nonempty and trimmed"
               if identity_field == "account_scope_id"
               else "account currency must be nonempty and trimmed")

    with pytest.raises(MetricsError, match=message):
        if paged:
            import_binance_usdm_transfer_income_pages(
                (BinanceIncomePage(10, 20, 2, payload),),
                query_start_ms=10, query_end_ms=20,
                account_scope_id=scope, currency=currency,
                sample_barriers_ns=(10_000_000, 20_000_000),
            )
        else:
            import_binance_usdm_transfer_income(
                payload, account_scope_id=scope, currency=currency,
                sample_barriers_ns=(10_000_000, 20_000_000),
            )


def test_binance_transfer_import_rejects_conflicting_ids_and_invalid_barriers() -> None:
    conflict = json.dumps([
        {"incomeType": "TRANSFER", "income": "1", "asset": "USDT", "time": 15, "tranId": 1},
        {"incomeType": "TRANSFER", "income": "2", "asset": "USDT", "time": 15, "tranId": 1},
    ])
    with pytest.raises(MetricsError, match="conflicting facts"):
        import_binance_usdm_transfer_income(
            conflict, account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10_000_000, 20_000_000),
        )


@pytest.mark.parametrize("barriers", [None, 1, "not-barriers", b"timestamps"])
def test_binance_transfer_import_normalizes_non_sequence_barriers(barriers: object) -> None:
    with pytest.raises(MetricsError, match="sample barriers must be a sequence"):
        import_binance_usdm_transfer_income(
            b"[]", account_scope_id="account", currency="USDT",
            sample_barriers_ns=barriers,
        )


@pytest.mark.parametrize("pages", [None, 1, "not-pages", b"[]"])
def test_binance_paged_import_normalizes_non_sequence_pages(pages: object) -> None:
    with pytest.raises(MetricsError, match="income pages must be a sequence"):
        import_binance_usdm_transfer_income_pages(
            pages, query_start_ms=10, query_end_ms=19,
            account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10, 20),
        )
    with pytest.raises(MetricsError, match="strictly increasing"):
        import_binance_usdm_transfer_income(
            b"[]", account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10, 10),
        )


def test_binance_transfer_import_rejects_duplicate_json_object_keys() -> None:
    payload = (
        b'[{"incomeType":"TRANSFER","incomeType":"TRANSFER","income":"1",'
        b'"asset":"USDT","time":15,"tranId":1}]'
    )
    with pytest.raises(MetricsError, match="duplicate object key"):
        import_binance_usdm_transfer_income(
            payload, account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10_000_000, 20_000_000),
        )


@pytest.mark.parametrize("paged", [False, True])
def test_binance_transfer_import_normalizes_excessive_json_nesting(paged: bool) -> None:
    payload = (b"[" * 20_000) + b"0" + (b"]" * 20_000)
    with pytest.raises(MetricsError, match="invalid Binance income-history"):
        if paged:
            import_binance_usdm_transfer_income_pages(
                (BinanceIncomePage(10, 20, 2, payload),), query_start_ms=10,
                query_end_ms=20, account_scope_id="account", currency="USDT",
                sample_barriers_ns=(10_000_000, 20_000_000),
            )
        else:
            import_binance_usdm_transfer_income(
                payload, account_scope_id="account", currency="USDT",
                sample_barriers_ns=(10_000_000, 20_000_000),
            )


def test_binance_paged_import_binds_exact_pages_and_contiguous_query_scope() -> None:
    page1 = b'[{"incomeType":"TRANSFER","income":"10","asset":"USDT","time":15,"tranId":1}]'
    page2 = b'[{"incomeType":"TRANSFER","income":"-2","asset":"USDT","time":25,"tranId":2}]'
    imported = import_binance_usdm_transfer_income_pages(
        (
            BinanceIncomePage(10, 19, 2, page1),
            BinanceIncomePage(20, 30, 2, page2),
        ),
        query_start_ms=10,
        query_end_ms=30,
        account_scope_id="account-3",
        currency="USDT",
        sample_barriers_ns=(10_000_000, 20_000_000, 30_000_000),
    )

    assert imported.query_time_range_ms == (10, 30)
    assert imported.page_sha256 == tuple(hashlib.sha256(page).hexdigest() for page in (page1, page2))
    assert imported.source_sha256 not in {
        hashlib.sha256(page1).hexdigest(), hashlib.sha256(page2).hexdigest(),
    }
    assert all(fact["source_sha256"] == imported.source_sha256 for fact in imported.facts)
    assert [fact["kind"] for fact in imported.facts] == ["DEPOSIT", "WITHDRAWAL"]
    assert imported.source_id.endswith(imported.source_sha256)
    assert any("do not authenticate" in item for item in imported.limitations)


def test_binance_paged_import_escapes_unpaired_surrogate_in_ignored_metadata() -> None:
    payload = (
        b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT",'
        b'"time":15,"tranId":1,"info":"\\ud800"}]'
    )
    imported = import_binance_usdm_transfer_income_pages(
        (BinanceIncomePage(10, 20, 100, payload),),
        query_start_ms=10,
        query_end_ms=20,
        account_scope_id="account",
        currency="USDT",
        sample_barriers_ns=(10_000_000, 20_000_000),
    )

    assert len(imported.facts) == 1
    assert imported.facts[0]["amount"] == "1"
    assert imported.page_sha256 == (hashlib.sha256(payload).hexdigest(),)


@pytest.mark.parametrize(
    ("pages", "query_start_ms", "query_end_ms", "message"),
    [
        ((BinanceIncomePage(11, 19, 2, b"[]"),), 10, 19, "contiguously"),
        ((BinanceIncomePage(10, 19, 2, b"[]"),), 10, 20, "do not reach"),
        ((BinanceIncomePage(10, 19, 2,
          b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT","time":15,"tranId":1},'
          b'{"incomeType":"TRANSFER","income":"2","asset":"USDT","time":16,"tranId":2}]'),),
         10, 19, "may be truncated"),
        ((BinanceIncomePage(10, 19, 5,
          b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT","time":20,"tranId":1}]'),),
         10, 19, "outside its declared page range"),
    ],
)
def test_binance_paged_import_rejects_incomplete_or_truncated_coverage(
    pages: tuple[BinanceIncomePage, ...],
    query_start_ms: int,
    query_end_ms: int,
    message: str,
) -> None:
    with pytest.raises(MetricsError, match=message):
        import_binance_usdm_transfer_income_pages(
            pages,
            query_start_ms=query_start_ms,
            query_end_ms=query_end_ms,
            account_scope_id="account",
            currency="USDT",
            sample_barriers_ns=(10, 20_000_000, 30_000_000),
        )


def test_binance_paged_import_rejects_duplicate_json_object_keys() -> None:
    page = BinanceIncomePage(
        10, 19, 2,
        b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT",'
        b'"time":15,"time":16,"tranId":1}]',
    )
    with pytest.raises(MetricsError, match="duplicate object key"):
        import_binance_usdm_transfer_income_pages(
            (page,), query_start_ms=10, query_end_ms=19,
            account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10, 20_000_000, 30_000_000),
        )


def test_binance_transfer_import_rejects_nonstandard_json_constant() -> None:
    payload = (b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT",'
               b'"time":15,"tranId":1,"metadata":{"value":NaN}}]')
    with pytest.raises(MetricsError, match="invalid Binance income-history JSON"):
        import_binance_usdm_transfer_income(
            payload, account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10_000_000, 20_000_000),
        )


def test_binance_paged_import_rejects_nonstandard_json_constant() -> None:
    page = BinanceIncomePage(
        10, 19, 2,
        b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT",'
        b'"time":15,"tranId":1,"metadata":{"value":Infinity}}]',
    )
    with pytest.raises(MetricsError, match="invalid Binance income-history page JSON"):
        import_binance_usdm_transfer_income_pages(
            (page,), query_start_ms=10, query_end_ms=19,
            account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10, 20_000_000, 30_000_000),
        )
