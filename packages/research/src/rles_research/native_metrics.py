"""Map detached native snapshots into regular economic metric increments."""
from __future__ import annotations

from decimal import DefaultContext, Decimal, DecimalException, Inexact, InvalidOperation, Rounded
from decimal import MIN_EMIN, localcontext
from collections.abc import Iterator, Sequence as SequenceABC
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .metrics import AccountFactsAccumulator, MetricsError

NATIVE_LIQUIDATION_COUNTS_AVAILABLE = True


class _JsonlReportRows:
    """Write optional report detail rows to a private temporary file, then publish once."""

    def __init__(self, output_path: str | os.PathLike[str]) -> None:
        requested = Path(output_path)
        if not requested.name:
            raise MetricsError("account rows output must name a file")
        cursor = requested.parent.absolute()
        while True:
            if cursor.is_symlink():
                raise MetricsError("account rows output directory must not contain symlinks")
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        parent = requested.parent.resolve()
        if not parent.is_dir():
            raise MetricsError("account rows output parent directory must exist")
        self.destination = parent / requested.name
        if self.destination.exists() or self.destination.is_symlink():
            raise MetricsError("account rows output already exists")
        try:
            handle = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", prefix=f".{requested.name}.",
                suffix=".tmp", dir=parent, delete=False,
            )
        except OSError as exc:
            raise MetricsError("could not create temporary account rows output") from exc
        self._handle = handle
        self._temporary = Path(handle.name)
        self.account_rows = 0
        self.reconciliation_rows = 0
        self._published = False

    def write(self, record_type: str, row: dict) -> None:
        record = {"record_type": record_type, **row}
        try:
            self._handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                          separators=(",", ":")) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            raise MetricsError("could not serialize account report row") from exc
        if record_type == "account_row":
            self.account_rows += 1
        else:
            self.reconciliation_rows += 1

    def publish(self) -> dict[str, Any]:
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            checksum = hashlib.sha256()
            with self._temporary.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    checksum.update(chunk)
            digest = checksum.hexdigest()
            # A hard link publishes the complete same-directory file atomically
            # and fails if another process created the destination meanwhile.
            # Unlike creating a placeholder followed by replace(), this cannot
            # overwrite a path swapped in by a concurrent writer.
            os.link(self._temporary, self.destination)
            self._temporary.unlink()
        except OSError as exc:
            raise MetricsError("could not publish account rows output") from exc
        self._published = True
        return {
            "file": self.destination.name,
            "format": "rles.native-report-rows.jsonl.v1",
            "sha256": digest,
            "account_row_count": self.account_rows,
            "equity_reconciliation_row_count": self.reconciliation_rows,
        }

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        if not self._published:
            self._temporary.unlink(missing_ok=True)


def _decimal_exact(operation: str, left: Decimal, right: Decimal) -> Decimal:
    """Perform exact Decimal arithmetic or reject values the active context cannot represent."""
    try:
        with localcontext(DefaultContext) as context:
            context.Emin = MIN_EMIN
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            context.clear_flags()
            if operation == "add":
                return left + right
            if operation == "multiply":
                return left * right
    except DecimalException as exc:
        raise MetricsError("native Decimal arithmetic would round or overflow") from exc
    raise AssertionError("unsupported exact Decimal operation")


def _add_decimal_exact(left: Decimal, right: Decimal) -> Decimal:
    return _decimal_exact("add", left, right)


def _multiply_decimal_exact(left: Decimal, right: Decimal) -> Decimal:
    return _decimal_exact("multiply", left, right)


def _subtract_decimal_exact(left: Decimal, right: Decimal) -> Decimal:
    return _add_decimal_exact(left, right.copy_negate())

_LIQUIDATION_ORDER_STATE_FIELDS = frozenset({
    "avg_px", "filled_qty", "is_canceled", "is_closed", "is_emulated", "is_expired",
    "is_filled", "is_inflight", "is_open", "is_pending_cancel", "is_pending_update",
    "is_rejected", "is_triggered", "is_working", "leaves_qty", "slippage", "status",
    "ts_last",
})


def _money(value: Any, currency: str) -> Decimal:
    parts = str(value).split()
    if len(parts) != 2 or parts[1] != currency:
        raise MetricsError("native amount has unexpected currency")
    return _decimal(parts[0], "ledger amount")


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise MetricsError(f"invalid native {label}") from exc
    if not result.is_finite():
        raise MetricsError(f"nonfinite native {label}")
    return result


def _external_cashflow_amount(value: Any) -> Decimal:
    """Parse sourced cash amounts without accepting binary floats or loose syntax."""
    if isinstance(value, Decimal):
        amount = value
    elif type(value) is int:
        amount = Decimal(value)
    elif (isinstance(value, str)
          and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value)):
        amount = Decimal(value)
    else:
        raise MetricsError("external cashflow amount must be canonical exact decimal text")
    if not amount.is_finite():
        raise MetricsError("external cashflow amount must be finite")
    return amount


def _ordered_snapshots(snapshots: Sequence[Any]) -> list[Any]:
    if (not isinstance(snapshots, SequenceABC)
            or isinstance(snapshots, (str, bytes, bytearray)) or not snapshots):
        raise MetricsError("native account snapshots required")
    try:
        currency = snapshots[0].currency
    except (AttributeError, TypeError) as exc:
        raise MetricsError("native account snapshot lacks currency") from exc
    if not isinstance(currency, str) or not currency:
        raise MetricsError("native account currency required")
    for snapshot in snapshots:
        try:
            timestamp = snapshot.decision_ts_ns
            snapshot_currency = snapshot.currency
            equity = snapshot.equity
        except (AttributeError, TypeError) as exc:
            raise MetricsError("native account snapshot lacks timestamp, currency, or equity") from exc
        if type(timestamp) is not int or timestamp < 0:
            raise MetricsError("native snapshot timestamp must be a nonnegative integer")
        if snapshot_currency != currency:
            raise MetricsError("native snapshots have conflicting currencies")
        _decimal(equity, "equity")
    # Input order is the barrier sequence within a timestamp (e.g. decision,
    # then terminal flatten). Stable sorting must not reorder those barriers.
    return sorted(snapshots, key=lambda item: item.decision_ts_ns)


def _snapshot_records(snapshot: Any, field: str, label: str) -> Sequence[Any]:
    try:
        records = getattr(snapshot, field)
    except (AttributeError, TypeError) as exc:
        raise MetricsError(f"native snapshot lacks {label} history") from exc
    if (not isinstance(records, SequenceABC)
            or isinstance(records, (str, bytes, bytearray))):
        raise MetricsError(f"native {label} history must be a sequence")
    return records


def _new_facts(records: Any, seen: dict[str, dict], label: str) -> Iterator[dict]:
    for record in records:
        if not isinstance(record, Mapping):
            raise MetricsError(f"native {label} history fact must be a mapping")
        identity = record.get("event_id")
        if not isinstance(identity, str) or not identity or identity != identity.strip():
            raise MetricsError(f"native {label} history fact lacks identity")
        required = ({"client_order_id", "order_side", "last_qty", "last_px", "commission"}
                    if label == "fill" else {"pnl_change"} if label == "funding" else set())
        if not required.issubset(record):
            raise MetricsError(f"native {label} history fact lacks required fields")
        if label == "fill" and (
                not isinstance(record["client_order_id"], str)
                or not record["client_order_id"]
                or record["client_order_id"] != record["client_order_id"].strip()):
            raise MetricsError("native fill client order identity is invalid")
        normalized = dict(record)
        previous = seen.get(identity)
        if previous is not None:
            if previous != normalized:
                raise MetricsError(f"native {label} identity has conflicting facts")
            continue
        seen[identity] = normalized
        yield normalized


def _liquidation_order_facts(snapshots: Sequence[Any]) -> dict[str, dict]:
    """Collect each liquidation order once, allowing only native state updates."""
    facts: dict[str, dict] = {}
    identities: dict[str, dict] = {}
    filled_quantities: dict[str, Decimal] = {}
    for snapshot in snapshots:
        for raw_order in _snapshot_records(snapshot, "liquidation_orders", "liquidation order"):
            if not isinstance(raw_order, Mapping):
                raise MetricsError("native liquidation order must be a mapping")
            identity = raw_order.get("client_order_id")
            if not isinstance(identity, str) or not identity or identity != identity.strip():
                raise MetricsError("native liquidation order lacks identity")
            order = dict(raw_order)
            immutable = {key: value for key, value in order.items()
                         if key not in _LIQUIDATION_ORDER_STATE_FIELDS}
            previous = identities.get(identity)
            if previous is not None and previous != immutable:
                raise MetricsError("native liquidation order identity has conflicting facts")
            identities[identity] = immutable
            if "filled_qty" in order:
                filled_qty = _decimal(order["filled_qty"], "liquidation order filled quantity")
                if filled_qty < 0:
                    raise MetricsError("native liquidation order filled quantity is negative")
                if "quantity" in order:
                    quantity = _decimal(order["quantity"], "liquidation order quantity")
                    if quantity <= 0 or filled_qty > quantity:
                        raise MetricsError("native liquidation order filled quantity exceeds order quantity")
                previous_filled = filled_quantities.get(identity)
                if previous_filled is not None and filled_qty < previous_filled:
                    raise MetricsError("native liquidation order cumulative fill quantity regressed")
                filled_quantities[identity] = filled_qty
            # Native order rows are cumulative snapshots. Keep the latest lifecycle state
            # after confirming that its stable order facts still match.
            facts[identity] = order
    return facts


def _external_cashflows_by_time(cashflows: Sequence[Mapping[str, Any]],
                                snapshots: Sequence[Any]) -> tuple[dict[int, Decimal], list[dict]]:
    """Validate explicitly sourced transfers and align them to unique sample barriers.

    The native AccountState API does not identify deposits or withdrawals. This
    interface therefore accepts only caller-supplied, source-identified facts;
    it never infers transfers from balance changes or ``reported`` snapshots.
    """
    if (not isinstance(cashflows, SequenceABC)
            or isinstance(cashflows, (str, bytes, bytearray))):
        raise MetricsError("external cashflows must be a sequence of sourced facts")
    ordered = _ordered_snapshots(snapshots)
    times: dict[int, int] = {}
    for snapshot in ordered:
        times[snapshot.decision_ts_ns] = times.get(snapshot.decision_ts_ns, 0) + 1
    allowed_times = set(times)
    first_time = ordered[0].decision_ts_ns
    currency = ordered[0].currency
    normalized: list[dict] = []
    for raw in cashflows:
        if not isinstance(raw, Mapping):
            raise MetricsError("external cashflow must be an explicitly sourced mapping")
        normalized.append(dict(raw))

    seen: dict[str, dict] = {}
    by_time: dict[int, Decimal] = {}
    unique: list[dict] = []
    for fact in _new_facts(normalized, seen, "external cashflow"):
        timestamp = fact.get("decision_ts_ns")
        source = fact.get("source_id")
        source_hash = fact.get("source_sha256")
        kind = fact.get("kind")
        if (type(timestamp) is not int or timestamp <= first_time or timestamp not in allowed_times):
            raise MetricsError("external cashflow must align to a sample after the initial barrier")
        if times[timestamp] != 1:
            raise MetricsError("external cashflow cannot align to duplicate native barriers")
        if fact.get("currency") != currency:
            raise MetricsError("external cashflow currency differs from native account")
        if not isinstance(kind, str) or kind not in {"DEPOSIT", "WITHDRAWAL"}:
            raise MetricsError("external cashflow kind must be DEPOSIT or WITHDRAWAL")
        amount = _external_cashflow_amount(fact.get("amount"))
        if amount <= 0:
            raise MetricsError("external cashflow amount must be positive")
        if not isinstance(source, str) or not source or source != source.strip():
            raise MetricsError("external cashflow source identity must be canonical")
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(character not in "0123456789abcdef" for character in source_hash)):
            raise MetricsError("external cashflow source SHA-256 required")
        # Identity is checked by _new_facts before semantic normalization. Store
        # the original typed fact and apply direction only in the ledger sum.
        signed = amount if kind == "DEPOSIT" else -amount
        by_time[timestamp] = _add_decimal_exact(by_time.get(timestamp, Decimal(0)), signed)
        unique.append(fact)
    return by_time, unique


def _fill_values(fill: dict) -> tuple[Decimal, Decimal]:
    side = fill.get("order_side")
    if not isinstance(side, str) or side not in {"BUY", "SELL"}:
        raise MetricsError("native fill side must be BUY or SELL")
    quantity = _decimal(fill.get("last_qty"), "fill quantity")
    price = _decimal(fill.get("last_px"), "fill price")
    if quantity <= 0 or price <= 0:
        raise MetricsError("native fill quantity and price must be positive")
    return quantity, price


def iter_account_rows(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                      external_cashflows: Sequence[Mapping[str, Any]] = (),
                      allow_price_improvement: bool = False) -> Iterator[dict]:
    """Yield regular native account rows with explicit sample-boundary flows.

    Callers must split history at a transfer barrier. A source hash records the
    supplied provenance identity; callers remain responsible for source trust.
    """
    if type(allow_price_improvement) is not bool:
        raise MetricsError("allow_price_improvement must be Boolean")
    if not isinstance(fill_references, Mapping):
        raise MetricsError("native fill references must be a mapping")
    fill_facts: dict[str, dict] = {}
    funding_facts: dict[str, dict] = {}
    # Multiple native barriers can share a timestamp (decision, then terminal
    # flatten). Stable sorting preserves their event order. Merge their facts
    # and use the last barrier's account state for the one sampled equity row.
    ordered = _ordered_snapshots(snapshots)
    for item in ordered:
        _snapshot_records(item, "fills", "fill")
        _snapshot_records(item, "funding", "funding")
    cashflows_by_time, _ = _external_cashflows_by_time(external_cashflows, ordered)
    for timestamp, group in groupby(ordered, key=lambda item: item.decision_ts_ns):
        same_time = tuple(group)
        snapshot = same_time[-1]
        fee = funding = turnover = slippage = Decimal(0)
        for fill in _new_facts((fill for item in same_time
                                for fill in _snapshot_records(item, "fills", "fill")),
                               fill_facts, "fill"):
            quantity, price = _fill_values(fill)
            reference = fill_references.get(fill["client_order_id"])
            if reference is None:
                raise MetricsError("fill has no native simulation reference for slippage accounting")
            reference = _decimal(reference, "fill reference")
            if reference <= 0:
                raise MetricsError("native fill reference must be positive")
            sign = 1 if fill["order_side"] == "BUY" else -1
            cost = _multiply_decimal_exact(
                _multiply_decimal_exact(_subtract_decimal_exact(price, reference), quantity),
                Decimal(sign),
            )
            if cost < 0 and not allow_price_improvement:
                raise MetricsError("conservative model produced favorable slippage")
            slippage = _add_decimal_exact(slippage, cost)
            turnover = _add_decimal_exact(turnover, _multiply_decimal_exact(quantity, price))
            fee = _add_decimal_exact(fee, _money(fill["commission"], snapshot.currency))
        for adjustment in _new_facts((adjustment for item in same_time
                                     for adjustment in _snapshot_records(item, "funding", "funding")),
                                     funding_facts, "funding"):
            funding = _add_decimal_exact(funding, _money(adjustment["pnl_change"], snapshot.currency))
        yield {"ts_ns": timestamp, "equity": float(snapshot.equity),
               "external_cashflow": float(cashflows_by_time.get(timestamp, Decimal(0))),
               "turnover_notional": float(turnover), "fee": float(fee), "funding": float(funding),
               "slippage_cost": float(slippage)}


def account_rows(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                 external_cashflows: Sequence[Mapping[str, Any]] = (),
                 allow_price_improvement: bool = False) -> list[dict]:
    """Materialize account rows, including only caller-supplied transfer facts."""
    return list(iter_account_rows(snapshots, fill_references=fill_references,
                                  external_cashflows=external_cashflows,
                                  allow_price_improvement=allow_price_improvement))


def reconcile_linear_equity(snapshots: Sequence[Any], *,
                            external_cashflows: Sequence[Mapping[str, Any]] = (),
                            rounding_per_event: Decimal = Decimal("0.00000001"),
                            include_rows: bool = True,
                            row_consumer: Callable[[dict], None] | None = None) -> dict:
    """Independently reconcile quote-settled, unit-multiplier linear accounts.

    Requires a flat, event-free starting barrier. External cashflows must be
    supplied as explicit source-identified facts aligned to unique sample barriers.
    The caller is responsible for verifying the source; the recorded SHA-256 only
    identifies the supplied source artifact.
    Check every flat barrier and, when the native snapshot contains a complete
    unit-multiplier linear position plus its authoritative mark, check open
    barriers using unrealized price PnL. Actual fill prices already include
    execution slippage. Inverse contracts, multi-asset accounts and external
    deposits/withdrawals without independent typed source facts are outside this contract.
    """
    if (not isinstance(rounding_per_event, Decimal) or not rounding_per_event.is_finite()
            or rounding_per_event < 0 or type(include_rows) is not bool
            or (row_consumer is not None and not callable(row_consumer))
            or (include_rows and row_consumer is not None)):
        raise MetricsError("finite nonnegative Decimal rounding and Boolean rows required")
    ordered = _ordered_snapshots(snapshots)
    for item in ordered:
        _snapshot_records(item, "positions", "position")
        _snapshot_records(item, "orders", "order")
        _snapshot_records(item, "fills", "fill")
        _snapshot_records(item, "funding", "funding")
    cashflows_by_time, cashflow_facts = _external_cashflows_by_time(external_cashflows, ordered)
    first = ordered[0]
    if (_snapshot_records(first, "positions", "position")
            or _snapshot_records(first, "orders", "order")
            or _snapshot_records(first, "fills", "fill")
            or _snapshot_records(first, "funding", "funding")):
        raise MetricsError("linear reconciliation requires a flat event-free initial barrier")
    initial = _decimal(first.equity, "initial equity")
    if initial <= 0:
        raise MetricsError("linear reconciliation requires positive initial equity")
    quantity = cash = fee = funding = external_cashflow = max_residual = Decimal(0)
    fills: dict[str, dict] = {}
    adjustments: dict[str, dict] = {}
    checked = open_barriers = checked_open = unchecked_open = 0
    rows = []

    def record_row(row: dict) -> None:
        if row_consumer is not None:
            row_consumer(row)
        elif include_rows:
            rows.append(row)

    for index, snapshot in enumerate(ordered):
        external_cashflow = _add_decimal_exact(
            external_cashflow,
            cashflows_by_time.get(snapshot.decision_ts_ns, Decimal(0)),
        )
        for fill in _new_facts(_snapshot_records(snapshot, "fills", "fill"), fills, "fill"):
            size, price = _fill_values(fill)
            signed = size if fill["order_side"] == "BUY" else size.copy_negate()
            quantity = _add_decimal_exact(quantity, signed)
            cash = _subtract_decimal_exact(cash, _multiply_decimal_exact(signed, price))
            fee = _add_decimal_exact(fee, _money(fill["commission"], snapshot.currency))
        for adjustment in _new_facts(_snapshot_records(snapshot, "funding", "funding"),
                                     adjustments, "funding"):
            funding = _add_decimal_exact(funding, _money(adjustment["pnl_change"], snapshot.currency))
        if quantity != 0:
            open_barriers += 1
            # Compact intermediate snapshots may omit a full position row. Only
            # certify an open barrier when its mark and linear contract metadata
            # are available and agree with the native fill ledger.
            mark_value = getattr(snapshot, "mark", getattr(snapshot, "mark_price", None))
            positions = _snapshot_records(snapshot, "positions", "position")
            if mark_value is not None and positions:
                if len(positions) != 1:
                    raise MetricsError("linear reconciliation requires one open native position")
                position = positions[0]
                if not isinstance(position, Mapping):
                    raise MetricsError("linear reconciliation requires native position mapping")
                if position.get("is_inverse") is True:
                    raise MetricsError("inverse position is outside linear equity reconciliation")
                if position.get("is_inverse") is not False:
                    raise MetricsError("native position inverse capability is unknown")
                multiplier = _decimal(position.get("multiplier"), "position multiplier")
                if multiplier != 1:
                    raise MetricsError("non-unit position multiplier is outside linear equity reconciliation")
                if position.get("settlement_currency") != snapshot.currency:
                    raise MetricsError("position settlement currency differs from account currency")
                side = position.get("side")
                native_size = _decimal(position.get("quantity"), "position quantity")
                if (native_size <= 0 or not isinstance(side, str)
                        or side not in {"LONG", "SHORT"}):
                    raise MetricsError("native linear position side and quantity are invalid")
                native_quantity = native_size if side == "LONG" else -native_size
                if native_quantity != quantity:
                    raise MetricsError("native position disagrees with cumulative fill quantity")
                mark = _decimal(mark_value, "mark price")
                if mark <= 0:
                    raise MetricsError("native mark price must be positive")
                marked_notional = _multiply_decimal_exact(quantity, mark)
                expected = _add_decimal_exact(
                    _add_decimal_exact(
                        _subtract_decimal_exact(
                            _add_decimal_exact(initial, _add_decimal_exact(marked_notional, cash)),
                            fee,
                        ),
                        funding,
                    ),
                    external_cashflow,
                )
                actual = _decimal(snapshot.equity, "equity")
                residual = _subtract_decimal_exact(actual, expected)
                event_count = Decimal(len(fills) + len(adjustments) + 1)
                tolerance = _multiply_decimal_exact(rounding_per_event, event_count)
                max_residual = max(max_residual, residual.copy_abs())
                if residual.copy_abs() > tolerance:
                    raise MetricsError(f"linear native equity mismatch at open barrier {index}: "
                                       f"residual={residual}, tolerance={tolerance}")
                checked_open += 1
                record_row({"barrier_index": index, "ts_ns": snapshot.decision_ts_ns,
                            "native_equity": str(actual), "expected_equity": str(expected),
                            "price_pnl_at_mark": str(
                                _add_decimal_exact(cash, marked_notional)
                            ),
                            "mark_price": str(mark), "signed_quantity": str(quantity),
                            "fee": str(fee), "funding": str(funding),
                            "external_cashflow": str(external_cashflow),
                            "residual": str(residual), "tolerance": str(tolerance)})
            else:
                unchecked_open += 1
            # The final barrier must retain authoritative terminal state.
            if index == len(ordered) - 1 and not snapshot.positions:
                raise MetricsError("terminal position disagrees with native fill quantity")
            continue
        if _snapshot_records(snapshot, "positions", "position"):
            raise MetricsError("flat fill quantity disagrees with native positions")
        expected = _add_decimal_exact(
            _add_decimal_exact(_subtract_decimal_exact(initial, fee), cash),
            _add_decimal_exact(funding, external_cashflow),
        )
        actual = _decimal(snapshot.equity, "equity")
        residual = _subtract_decimal_exact(actual, expected)
        event_count = Decimal(len(fills) + len(adjustments) + 1)
        tolerance = _multiply_decimal_exact(rounding_per_event, event_count)
        max_residual = max(max_residual, residual.copy_abs())
        if residual.copy_abs() > tolerance:
            raise MetricsError(f"linear native equity mismatch at barrier {index}: "
                               f"residual={residual}, tolerance={tolerance}")
        checked += 1
        record_row({"barrier_index": index, "ts_ns": snapshot.decision_ts_ns,
                    "native_equity": str(actual), "expected_equity": str(expected),
                    "realized_price_pnl": str(cash), "fee": str(fee),
                    "funding": str(funding), "external_cashflow": str(external_cashflow),
                    "residual": str(residual), "tolerance": str(tolerance)})
    result: dict = {"scope": "flat_and_marked_open_barriers_quote_settled_unit_linear_with_explicit_cashflow_sources",
              "checked_flat_barriers": checked, "open_barriers": open_barriers,
              "unchecked_open_barriers": unchecked_open,
              "checked_open_barriers": checked_open,
              "terminal_checked": quantity == 0, "max_abs_residual": str(max_residual),
              "rounding_per_event": str(rounding_per_event),
              "unique_fills": len(fills), "unique_funding": len(adjustments),
              "unique_external_cashflows": len(cashflow_facts),
              "net_external_cashflow": str(external_cashflow)}
    if include_rows:
        result["rows"] = rows
    return result


def native_report(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                  sample_interval_ns: int, allow_price_improvement: bool = False,
                  include_account_rows: bool = True,
                  external_cashflows: Sequence[Mapping[str, Any]] = (),
                  linear_equity_rounding: Decimal | None = None,
                  contract_multiplier: Decimal = Decimal(1),
                  account_rows_path: str | os.PathLike[str] | None = None) -> dict:
    """Build native economic metrics without inferring external account flows."""
    if (type(include_account_rows) is not bool or not isinstance(contract_multiplier, Decimal)
            or not contract_multiplier.is_finite() or contract_multiplier != 1
            or (account_rows_path is not None
                and not isinstance(account_rows_path, (str, os.PathLike)))
            or (account_rows_path is not None and not include_account_rows)):
        raise MetricsError("Boolean row option and unit finite contract multiplier required")
    ordered = _ordered_snapshots(snapshots)
    for item in ordered:
        _snapshot_records(item, "positions", "position")
        _snapshot_records(item, "orders", "order")
    row_output = _JsonlReportRows(account_rows_path) if account_rows_path is not None else None
    try:
        report = _build_native_report(
            ordered, fill_references=fill_references,
            sample_interval_ns=sample_interval_ns,
            allow_price_improvement=allow_price_improvement,
            include_account_rows=include_account_rows,
            external_cashflows=external_cashflows,
            linear_equity_rounding=linear_equity_rounding,
            row_output=row_output,
        )
        if row_output is not None:
            report["account_rows_artifact"] = row_output.publish()
        return report
    finally:
        if row_output is not None:
            row_output.close()


def _build_native_report(ordered: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                         sample_interval_ns: int, allow_price_improvement: bool,
                         include_account_rows: bool,
                         external_cashflows: Sequence[Mapping[str, Any]],
                         linear_equity_rounding: Decimal | None,
                         row_output: _JsonlReportRows | None) -> dict:
    accumulator = AccountFactsAccumulator(sample_interval_ns=sample_interval_ns,
                                          allow_price_improvement=allow_price_improvement)
    rows: list[dict] | None = [] if include_account_rows and row_output is None else None
    account_row_count = 0
    for row in iter_account_rows(ordered, fill_references=fill_references,
                                 external_cashflows=external_cashflows,
                                 allow_price_improvement=allow_price_improvement):
        accumulator.add(row)
        if rows is not None:
            rows.append(row)
        elif row_output is not None:
            row_output.write("account_row", row)
            account_row_count += 1
    metrics = accumulator.result(periods_per_year=365.25 * 86400 * 1e9 / sample_interval_ns)
    final = ordered[-1]
    fill_facts: dict[str, dict] = {}
    funding_facts: dict[str, dict] = {}
    for snapshot in ordered:
        for record, facts, label in ((_snapshot_records(snapshot, "fills", "fill"), fill_facts, "fill"),
                                     (_snapshot_records(snapshot, "funding", "funding"),
                                      funding_facts, "funding")):
            for item in record:
                identity = str(item.get("event_id", ""))
                if not identity:
                    raise MetricsError(f"native {label} history fact lacks identity")
                previous = facts.get(identity)
                if previous is not None and previous != item:
                    raise MetricsError(f"native {label} identity has conflicting facts")
                facts[identity] = dict(item)
    observed = all(
        getattr(snapshot, "liquidation_capability", False) is True
        and getattr(snapshot, "liquidation_orders", None) is not None
        for snapshot in ordered
    )
    liquidation_orders = _liquidation_order_facts(ordered) if observed else {}
    _, cashflow_facts = _external_cashflows_by_time(external_cashflows, ordered)
    final_positions = _snapshot_records(final, "positions", "position")
    final_orders = _snapshot_records(final, "orders", "order")
    report = {"metrics": metrics, "fills": list(fill_facts.values()),
            "funding": list(funding_facts.values()),
            "terminal_flat": not final_positions and not final_orders,
            "external_cashflows": cashflow_facts,
            "liquidations": len(liquidation_orders) if observed else None,
            "liquidations_observed": observed,
            "liquidation_orders": list(liquidation_orders.values()) if observed else None,
            "costs_already_in_native_equity": True}
    if rows is not None:
        report["account_rows"] = rows
    elif row_output is not None:
        report["account_rows_streamed"] = account_row_count
    if linear_equity_rounding is not None:
        report["equity_reconciliation"] = reconcile_linear_equity(
            ordered, external_cashflows=external_cashflows,
            rounding_per_event=linear_equity_rounding,
            include_rows=include_account_rows and row_output is None,
            row_consumer=(None if row_output is None else
                          lambda row: row_output.write("equity_reconciliation_row", row)),
        )
    return report


class NativeReportAccumulator:
    """Reduce native account barriers online without retaining the snapshot history.

    One same-timestamp group is buffered because the batch report merges its fills
    and funding while sampling the final equity at that timestamp. Return order
    statistics are retained by ``AccountFactsAccumulator`` as required for exact
    VaR/CVaR; native snapshots themselves are released after each group.
    """

    def __init__(self, *, sample_interval_ns: int, max_return_count: int,
                 contract_multiplier: Decimal,
                 external_cashflows: Sequence[Mapping[str, Any]] = (),
                 linear_equity_rounding: Decimal | None = None,
                 allow_price_improvement: bool = False) -> None:
        if (type(max_return_count) is not int or max_return_count < 1
                or not isinstance(contract_multiplier, Decimal)
                or not contract_multiplier.is_finite() or contract_multiplier != 1
                or (linear_equity_rounding is not None
                    and (not isinstance(linear_equity_rounding, Decimal)
                         or not linear_equity_rounding.is_finite()
                         or linear_equity_rounding < 0))
                or type(allow_price_improvement) is not bool):
            raise MetricsError("invalid native report accumulator configuration")
        self.sample_interval_ns = sample_interval_ns
        self.max_return_count = max_return_count
        self.contract_multiplier = contract_multiplier
        self.linear_equity_rounding = linear_equity_rounding
        self.allow_price_improvement = allow_price_improvement
        self._metrics = AccountFactsAccumulator(
            sample_interval_ns=sample_interval_ns,
            allow_price_improvement=allow_price_improvement,
            max_return_count=max_return_count,
        )
        self._currency: str | None = None
        self._first_ts: int | None = None
        self._last_ts = -1
        self._last_snapshot: Any = None
        self._group: list[Any] = []
        self._group_refs: dict[str, Decimal] = {}
        self._finished = False
        self._fill_facts: dict[str, dict] = {}
        self._funding_facts: dict[str, dict] = {}
        self._fills: list[dict] = []
        self._funding: list[dict] = []
        self._references: dict[str, Decimal] = {}
        self._cashflow_facts: list[dict] = []
        self._cashflows_by_time: dict[int, Decimal] = {}
        self._cashflow_seen: set[int] = set()
        self._liquidation_observed = True
        self._liquidation_orders: dict[str, dict] = {}
        self._liquidation_identities: dict[str, dict] = {}
        self._liquidation_filled: dict[str, Decimal] = {}
        self._reconciliation: dict[str, Any] | None = None
        self._prepare_cashflows(external_cashflows)

    def _prepare_cashflows(self, external_cashflows: Sequence[Mapping[str, Any]]) -> None:
        if (not isinstance(external_cashflows, SequenceABC)
                or isinstance(external_cashflows, (str, bytes, bytearray))):
            raise MetricsError("external cashflows must be a sequence")
        seen: dict[str, dict] = {}
        for fact in _new_facts(external_cashflows, seen, "external cashflow"):
            timestamp = fact.get("decision_ts_ns")
            kind = fact.get("kind")
            source = fact.get("source_id")
            source_hash = fact.get("source_sha256")
            if type(timestamp) is not int or timestamp < 0:
                raise MetricsError("external cashflow timestamp must be a nonnegative integer")
            if not isinstance(kind, str) or kind not in {"DEPOSIT", "WITHDRAWAL"}:
                raise MetricsError("external cashflow kind must be DEPOSIT or WITHDRAWAL")
            if not isinstance(source, str) or not source or source != source.strip():
                raise MetricsError("external cashflow source identity must be canonical")
            if (not isinstance(source_hash, str) or len(source_hash) != 64
                    or any(character not in "0123456789abcdef" for character in source_hash)):
                raise MetricsError("external cashflow source SHA-256 required")
            amount = _external_cashflow_amount(fact.get("amount"))
            if amount <= 0:
                raise MetricsError("external cashflow amount must be positive")
            if not isinstance(fact.get("currency"), str) or not fact["currency"]:
                raise MetricsError("external cashflow currency required")
            signed = amount if kind == "DEPOSIT" else amount.copy_negate()
            self._cashflows_by_time[timestamp] = _add_decimal_exact(
                self._cashflows_by_time.get(timestamp, Decimal(0)), signed,
            )
            self._cashflow_facts.append(fact)

    def add_snapshot(self, snapshot: Any,
                     fill_references: Mapping[str, Decimal]) -> None:
        if self._finished:
            raise MetricsError("native report accumulator is already finished")
        if not isinstance(fill_references, Mapping):
            raise MetricsError("native fill references must be a mapping")
        _ordered_snapshots((snapshot,))
        timestamp = snapshot.decision_ts_ns
        for field, label in (("positions", "position"), ("orders", "order"),
                             ("fills", "fill"), ("funding", "funding")):
            _snapshot_records(snapshot, field, label)
        if self._currency is None:
            self._currency = snapshot.currency
            self._first_ts = timestamp
            if any(fact["currency"] != self._currency for fact in self._cashflow_facts):
                raise MetricsError("external cashflow currency differs from native account")
        elif snapshot.currency != self._currency:
            raise MetricsError("native snapshots have conflicting currencies")
        if timestamp < self._last_ts:
            raise MetricsError("native snapshot time order invalid")
        if self._group and timestamp != self._last_ts:
            self._flush_group()
        self._group.append(snapshot)
        self._last_snapshot = snapshot
        self._last_ts = timestamp
        current_fill_ids = {
            fact.get("client_order_id")
            for fact in _snapshot_records(snapshot, "fills", "fill")
            if isinstance(fact, Mapping) and isinstance(fact.get("client_order_id"), str)
        }
        for order_id in current_fill_ids:
            if not isinstance(order_id, str) or not order_id or order_id != order_id.strip():
                raise MetricsError("native fill reference order identity is invalid")
            if order_id not in fill_references:
                continue
            raw_reference = fill_references[order_id]
            reference = _decimal(raw_reference, "fill reference")
            previous = self._references.get(order_id)
            if previous is not None and previous != reference:
                raise MetricsError("native fill reference changed for an order")
            self._references[order_id] = reference
            prior_group = self._group_refs.get(order_id)
            if prior_group is not None and prior_group != reference:
                raise MetricsError("native fill reference changed within a barrier group")
            self._group_refs[order_id] = reference

    def _update_liquidations(self, snapshot: Any) -> None:
        if (getattr(snapshot, "liquidation_capability", False) is not True
                or getattr(snapshot, "liquidation_orders", None) is None):
            self._liquidation_observed = False
            return
        for raw_order in snapshot.liquidation_orders:
            if not isinstance(raw_order, Mapping):
                raise MetricsError("native liquidation order must be a mapping")
            identity = raw_order.get("client_order_id")
            if not isinstance(identity, str) or not identity or identity != identity.strip():
                raise MetricsError("native liquidation order lacks identity")
            order = dict(raw_order)
            immutable = {key: value for key, value in order.items()
                         if key not in _LIQUIDATION_ORDER_STATE_FIELDS}
            previous = self._liquidation_identities.get(identity)
            if previous is not None and previous != immutable:
                raise MetricsError("native liquidation order identity has conflicting facts")
            self._liquidation_identities[identity] = immutable
            if "filled_qty" in order:
                filled = _decimal(order["filled_qty"], "liquidation order filled quantity")
                if filled < 0:
                    raise MetricsError("native liquidation order filled quantity is negative")
                if "quantity" in order:
                    quantity = _decimal(order["quantity"], "liquidation order quantity")
                    if quantity <= 0 or filled > quantity:
                        raise MetricsError("native liquidation order filled quantity exceeds order quantity")
                previous_filled = self._liquidation_filled.get(identity)
                if previous_filled is not None and filled < previous_filled:
                    raise MetricsError("native liquidation order cumulative fill quantity regressed")
                self._liquidation_filled[identity] = filled
            self._liquidation_orders[identity] = order

    def _update_reconciliation(self, snapshot: Any, fills: Sequence[dict],
                               funding: Sequence[dict], cashflow: Decimal) -> None:
        if self.linear_equity_rounding is None:
            return
        state = self._reconciliation
        if state is None:
            if (_snapshot_records(snapshot, "positions", "position")
                    or _snapshot_records(snapshot, "orders", "order") or fills or funding):
                raise MetricsError("linear reconciliation requires a flat event-free initial barrier")
            initial = _decimal(snapshot.equity, "initial equity")
            if initial <= 0:
                raise MetricsError("linear reconciliation requires positive initial equity")
            state = self._reconciliation = {
                "initial": initial, "quantity": Decimal(0), "cash": Decimal(0),
                "fee": Decimal(0), "funding": Decimal(0), "external": Decimal(0),
                "max_residual": Decimal(0), "checked": 0, "open": 0,
                "checked_open": 0, "unchecked_open": 0,
            }
        state["external"] = _add_decimal_exact(state["external"], cashflow)
        for fill in fills:
            size, price = _fill_values(fill)
            signed = size if fill["order_side"] == "BUY" else size.copy_negate()
            state["quantity"] = _add_decimal_exact(state["quantity"], signed)
            state["cash"] = _subtract_decimal_exact(
                state["cash"], _multiply_decimal_exact(signed, price),
            )
            state["fee"] = _add_decimal_exact(
                state["fee"], _money(fill["commission"], snapshot.currency),
            )
        for adjustment in funding:
            state["funding"] = _add_decimal_exact(
                state["funding"], _money(adjustment["pnl_change"], snapshot.currency),
            )
        quantity = state["quantity"]
        if quantity != 0:
            state["open"] += 1
            mark_value = getattr(snapshot, "mark", getattr(snapshot, "mark_price", None))
            positions = _snapshot_records(snapshot, "positions", "position")
            if mark_value is None or not positions:
                state["unchecked_open"] += 1
                return
            if len(positions) != 1 or not isinstance(positions[0], Mapping):
                raise MetricsError("linear reconciliation requires one open native position")
            position = positions[0]
            if position.get("is_inverse") is not False:
                raise MetricsError("inverse position capability is unknown or unsupported")
            if _decimal(position.get("multiplier"), "position multiplier") != 1:
                raise MetricsError("non-unit position multiplier is outside linear equity reconciliation")
            if position.get("settlement_currency") != snapshot.currency:
                raise MetricsError("position settlement currency differs from account currency")
            side = position.get("side")
            native_size = _decimal(position.get("quantity"), "position quantity")
            if native_size <= 0 or side not in {"LONG", "SHORT"}:
                raise MetricsError("native linear position side and quantity are invalid")
            native_quantity = native_size if side == "LONG" else native_size.copy_negate()
            if native_quantity != quantity:
                raise MetricsError("native position disagrees with cumulative fill quantity")
            mark = _decimal(mark_value, "mark price")
            if mark <= 0:
                raise MetricsError("native mark price must be positive")
            marked = _multiply_decimal_exact(quantity, mark)
            expected = _add_decimal_exact(
                _add_decimal_exact(
                    _subtract_decimal_exact(
                        _add_decimal_exact(
                            state["initial"], _add_decimal_exact(marked, state["cash"]),
                        ), state["fee"],
                    ), state["funding"],
                ), state["external"],
            )
            state["checked_open"] += 1
        else:
            positions = _snapshot_records(snapshot, "positions", "position")
            if positions:
                raise MetricsError("flat fill quantity disagrees with native positions")
            expected = _add_decimal_exact(
                _add_decimal_exact(
                    _subtract_decimal_exact(state["initial"], state["fee"]), state["cash"],
                ), _add_decimal_exact(state["funding"], state["external"]),
            )
            state["checked"] += 1
        actual = _decimal(snapshot.equity, "equity")
        residual = _subtract_decimal_exact(actual, expected)
        tolerance = _multiply_decimal_exact(
            self.linear_equity_rounding,
            Decimal(len(self._fill_facts) + len(self._funding_facts) + 1),
        )
        state["max_residual"] = max(state["max_residual"], residual.copy_abs())
        if residual.copy_abs() > tolerance:
            raise MetricsError(f"linear native equity mismatch at barrier {self._metrics.count}: "
                               f"residual={residual}, tolerance={tolerance}")

    def _flush_group(self) -> None:
        if not self._group:
            return
        timestamp = self._group[0].decision_ts_ns
        if timestamp in self._cashflows_by_time:
            if timestamp == self._first_ts:
                raise MetricsError("external cashflow must align after the initial barrier")
            if len(self._group) != 1:
                raise MetricsError("external cashflow cannot align to duplicate native barriers")
        currency = self._currency
        assert currency is not None
        fee = funding_total = turnover = slippage = Decimal(0)
        cashflow = self._cashflows_by_time.get(timestamp, Decimal(0))
        for snapshot in self._group:
            new_fills = tuple(_new_facts(
                _snapshot_records(snapshot, "fills", "fill"), self._fill_facts, "fill",
            ))
            new_funding = tuple(_new_facts(
                _snapshot_records(snapshot, "funding", "funding"),
                self._funding_facts, "funding",
            ))
            for fill in new_fills:
                quantity, price = _fill_values(fill)
                order_id = fill["client_order_id"]
                reference = self._references.get(order_id)
                if reference is None:
                    raise MetricsError("fill has no native simulation reference for slippage accounting")
                if reference <= 0:
                    raise MetricsError("native fill reference must be positive")
                direction = Decimal(1 if fill["order_side"] == "BUY" else -1)
                cost = _multiply_decimal_exact(
                    _multiply_decimal_exact(_subtract_decimal_exact(price, reference), quantity),
                    direction,
                )
                if cost < 0 and not self.allow_price_improvement:
                    raise MetricsError("conservative model produced favorable slippage")
                slippage = _add_decimal_exact(slippage, cost)
                turnover = _add_decimal_exact(
                    turnover, _multiply_decimal_exact(quantity, price),
                )
                fee = _add_decimal_exact(fee, _money(fill["commission"], currency))
                self._fills.append(fill)
            for adjustment in new_funding:
                amount = _money(adjustment["pnl_change"], currency)
                funding_total = _add_decimal_exact(funding_total, amount)
                self._funding.append(adjustment)
            self._update_reconciliation(snapshot, new_fills, new_funding, cashflow)
            self._update_liquidations(snapshot)
        if len(self._group) == 1 and timestamp in self._cashflows_by_time:
            self._cashflow_seen.add(timestamp)
        self._metrics.add({
            "ts_ns": timestamp,
            "equity": float(self._group[-1].equity),
            "external_cashflow": float(cashflow),
            "turnover_notional": float(turnover),
            "fee": float(fee),
            "funding": float(funding_total),
            "slippage_cost": float(slippage),
        })
        self._group.clear()
        self._group_refs.clear()

    def finish(self) -> dict:
        if self._finished:
            raise MetricsError("native report accumulator is already finished")
        if not self._group:
            raise MetricsError("native account snapshots required")
        self._flush_group()
        if self._cashflow_seen != set(self._cashflows_by_time):
            raise MetricsError("external cashflow does not align to a native sample barrier")
        final = self._last_snapshot
        final_positions = _snapshot_records(final, "positions", "position")
        final_orders = _snapshot_records(final, "orders", "order")
        if self._reconciliation is not None and self._reconciliation["quantity"] != 0 and not final_positions:
            raise MetricsError("terminal position disagrees with native fill quantity")
        metrics = self._metrics.result(
            periods_per_year=365.25 * 86400 * 1e9 / self.sample_interval_ns,
        )
        observed = self._liquidation_observed
        report = {
            "metrics": metrics,
            "fills": list(self._fills),
            "funding": list(self._funding),
            "terminal_flat": not final_positions and not final_orders,
            "external_cashflows": list(self._cashflow_facts),
            "liquidations": len(self._liquidation_orders) if observed else None,
            "liquidations_observed": observed,
            "liquidation_orders": list(self._liquidation_orders.values()) if observed else None,
            "costs_already_in_native_equity": True,
        }
        if self._reconciliation is not None:
            state = self._reconciliation
            report["equity_reconciliation"] = {
                "scope": "flat_and_marked_open_barriers_quote_settled_unit_linear_with_explicit_cashflow_sources",
                "checked_flat_barriers": state["checked"],
                "open_barriers": state["open"],
                "unchecked_open_barriers": state["unchecked_open"],
                "checked_open_barriers": state["checked_open"],
                "terminal_checked": state["quantity"] == 0,
                "max_abs_residual": str(state["max_residual"]),
                "rounding_per_event": str(self.linear_equity_rounding),
                "unique_fills": len(self._fill_facts),
                "unique_funding": len(self._funding_facts),
                "unique_external_cashflows": len(self._cashflow_facts),
                "net_external_cashflow": str(state["external"]),
            }
        self._finished = True
        self._group.clear()
        return report
