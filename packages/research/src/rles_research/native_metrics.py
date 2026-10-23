Warning: truncated output (original token count: 13971)
Total output lines: 1039

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
            # Create the destination exclusively so a concurrent file or symlink
            # cannot be overwritten by the final rename.
            descriptor = os.open(self.destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            try:
                os.replace(self._temporary, self.destination)
            except OSError:
                self.destination.unlink(missing_ok=True)
                raise
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
     …6971 tokens truncated…napshot, "fills", "fill")
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

