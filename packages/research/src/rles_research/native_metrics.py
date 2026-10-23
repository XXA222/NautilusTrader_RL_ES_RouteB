"""Map detached native snapshots into regular economic metric increments."""
from __future__ import annotations

from decimal import (
    Decimal, DecimalException, DefaultContext, Inexact, InvalidOperation, MIN_EMIN,
    Rounded, localcontext,
)
from collections.abc import Iterator, Sequence as SequenceABC
from itertools import groupby
import re
from typing import Any, Mapping, Sequence

from .metrics import AccountFactsAccumulator, MetricsError

NATIVE_LIQUIDATION_COUNTS_AVAILABLE = True

_MAX_EXACT_DECIMAL_OPERATION_DIGITS = 4096

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


def _cashflow_decimal(value: Any) -> Decimal:
    """Parse only exact, unambiguous external cashflow amount values."""
    if (isinstance(value, (bool, float))
            or not isinstance(value, (str, int, Decimal))
            or (isinstance(value, str)
                and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is None)):
        raise MetricsError("external cashflow amount must be an exact decimal value")
    return _decimal(value, "external cashflow amount")


def _add_decimal_exact(current: Decimal, increment: Decimal, *, label: str) -> Decimal:
    if not current.is_finite() or not increment.is_finite():
        raise MetricsError(f"{label} requires finite decimal values")
    if not current:
        return increment
    if not increment:
        return current
    left = current.as_tuple()
    right = increment.as_tuple()
    left_exponent = left.exponent
    right_exponent = right.exponent
    if not isinstance(left_exponent, int) or not isinstance(right_exponent, int):
        raise MetricsError(f"{label} requires finite decimal values")
    min_exponent = min(left_exponent, right_exponent)
    max_adjusted = max(
        left_exponent + len(left.digits) - 1,
        right_exponent + len(right.digits) - 1,
    )
    precision = max_adjusted - min_exponent + 2
    if precision > _MAX_EXACT_DECIMAL_OPERATION_DIGITS:
        raise MetricsError(f"{label} exceeds exact decimal precision limit")
    try:
        with localcontext() as context:
            context.prec = precision
            context.Emax = DefaultContext.Emax
            context.Emin = MIN_EMIN
            context.clear_flags()
            total = current + increment
            if context.flags[Inexact] or context.flags[Rounded]:
                raise MetricsError(f"{label} exceeds finite decimal range")
    except DecimalException as exc:
        raise MetricsError(f"{label} exceeds finite decimal range") from exc
    if not total.is_finite():
        raise MetricsError(f"{label} exceeds finite decimal range")
    return total


def _subtract_decimal_exact(left: Decimal, right: Decimal, *, label: str) -> Decimal:
    if not right.is_finite():
        raise MetricsError(f"{label} requires finite decimal values")
    return _add_decimal_exact(left, right.copy_negate(), label=label)


def _multiply_decimal_exact(left: Decimal, right: Decimal, *, label: str) -> Decimal:
    if not left.is_finite() or not right.is_finite():
        raise MetricsError(f"{label} requires finite decimal values")
    if not left or not right:
        return Decimal(0)
    precision = len(left.as_tuple().digits) + len(right.as_tuple().digits)
    if precision > _MAX_EXACT_DECIMAL_OPERATION_DIGITS:
        raise MetricsError(f"{label} exceeds exact decimal precision limit")
    try:
        with localcontext() as context:
            context.prec = precision
            context.Emax = DefaultContext.Emax
            context.Emin = MIN_EMIN
            context.clear_flags()
            product = left * right
            if context.flags[Inexact] or context.flags[Rounded]:
                raise MetricsError(f"{label} exceeds finite decimal range")
    except DecimalException as exc:
        raise MetricsError(f"{label} exceeds finite decimal range") from exc
    if not product.is_finite():
        raise MetricsError(f"{label} exceeds finite decimal range")
    return product


def _add_cashflow(current: Decimal, signed_amount: Decimal) -> Decimal:
    return _add_decimal_exact(current, signed_amount, label="external cashflow sum")


def _sum_decimal_exact(values: Sequence[Decimal], *, label: str) -> Decimal:
    total = Decimal(0)
    for value in values:
        total = _add_decimal_exact(total, value, label=label)
    return total


def _ordered_snapshots(snapshots: Sequence[Any]) -> Sequence[Any]:
    if (not isinstance(snapshots, SequenceABC)
            or isinstance(snapshots, (str, bytes, bytearray)) or not snapshots):
        raise MetricsError("native account snapshots required")
    try:
        currency = snapshots[0].currency
    except (AttributeError, TypeError) as exc:
        raise MetricsError("native account snapshot lacks currency") from exc
    if not isinstance(currency, str) or not currency:
        raise MetricsError("native account currency required")
    ordered = True
    previous_timestamp = -1
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
        if timestamp < previous_timestamp:
            ordered = False
        previous_timestamp = timestamp
    # Input order is the barrier sequence within a timestamp (e.g. decision,
    # then terminal flatten). Factory snapshots are already chronological, so
    # preserve that sequence without allocating a second list. Arbitrary
    # caller-provided order retains the stable-sort compatibility path.
    return snapshots if ordered else sorted(snapshots, key=lambda item: item.decision_ts_ns)


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


def _update_liquidation_order(raw_order: Any, facts: dict[str, dict],
                              filled_quantities: dict[str, Decimal]) -> None:
    if not isinstance(raw_order, Mapping):
        raise MetricsError("native liquidation order must be a mapping")
    identity = raw_order.get("client_order_id")
    if not isinstance(identity, str) or not identity or identity != identity.strip():
        raise MetricsError("native liquidation order lacks identity")
    order = dict(raw_order)
    immutable = {key: value for key, value in order.items()
                 if key not in _LIQUIDATION_ORDER_STATE_FIELDS}
    previous = facts.get(identity)
    if previous is not None:
        previous_immutable = {
            key: value for key, value in previous.items()
            if key not in _LIQUIDATION_ORDER_STATE_FIELDS
        }
        if previous_immutable != immutable:
            raise MetricsError("native liquidation order identity has conflicting facts")
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


def _liquidation_order_facts(snapshots: Sequence[Any]) -> dict[str, dict]:
    """Collect each liquidation order once, allowing only native state updates."""
    facts: dict[str, dict] = {}
    filled_quantities: dict[str, Decimal] = {}
    for snapshot in snapshots:
        for raw_order in _snapshot_records(snapshot, "liquidation_orders", "liquidation order"):
            _update_liquidation_order(raw_order, facts, filled_quantities)
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
    seen: dict[str, dict] = {}
    by_time: dict[int, Decimal] = {}
    unique: list[dict] = []
    for fact in _new_facts(cashflows, seen, "external cashflow"):
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
        raw_amount = fact.get("amount")
        amount = _cashflow_decimal(raw_amount)
        if amount <= 0:
            raise MetricsError("external cashflow amount must be positive")
        if (not isinstance(source, str) or not source.strip()
                or source != source.strip()):
            raise MetricsError("external cashflow source identity required without surrounding whitespace")
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(character not in "0123456789abcdef" for character in source_hash)):
            raise MetricsError("external cashflow source SHA-256 required")
        # Identity is checked by _new_facts before semantic normalization. Store
        # the original typed fact and apply direction only in the ledger sum.
        signed = amount if kind == "DEPOSIT" else amount.copy_negate()
        by_time[timestamp] = _add_cashflow(by_time.get(timestamp, Decimal(0)), signed)
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


def _observed_contract_multiplier(snapshots: Sequence[Any],
                                  explicit_multiplier: Decimal | None,
                                  *, has_fills: bool) -> Decimal:
    """Resolve one stable multiplier, requiring rule metadata for positionless fills."""
    observed: set[Decimal] = set()
    for snapshot in snapshots:
        for position in _snapshot_records(snapshot, "positions", "position"):
            if not isinstance(position, Mapping):
                raise MetricsError("native position history fact must be a mapping")
            if position.get("is_inverse") is True:
                raise MetricsError("inverse position is outside linear notional accounting")
            if position.get("is_inverse") is not False:
                raise MetricsError("native position linear capability is unknown")
            if "multiplier" not in position:
                raise MetricsError("native linear position multiplier is missing")
            multiplier = _decimal(position["multiplier"], "position multiplier")
            if multiplier <= 0:
                raise MetricsError("position multiplier must be positive")
            observed.add(multiplier)
    if len(observed) > 1:
        raise MetricsError("contract multiplier changes within one native report")
    if explicit_multiplier is not None:
        multiplier = _decimal(explicit_multiplier, "contract multiplier")
        if multiplier <= 0:
            raise MetricsError("contract multiplier must be positive")
        if observed and multiplier not in observed:
            raise MetricsError("explicit contract multiplier conflicts with native position facts")
        return multiplier
    if observed:
        return next(iter(observed))
    if has_fills:
        raise MetricsError("native fills without position snapshots require the frozen rule contract multiplier")
    return Decimal(1)


def iter_account_rows(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                      external_cashflows: Sequence[Mapping[str, Any]] = (),
                      allow_price_improvement: bool = False,
                      contract_multiplier: Decimal | None = None) -> Iterator[dict]:
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
    has_fills = False
    for item in ordered:
        has_fills = bool(_snapshot_records(item, "fills", "fill")) or has_fills
        _snapshot_records(item, "funding", "funding")
    cashflows_by_time, _ = _external_cashflows_by_time(external_cashflows, ordered)
    resolved_multiplier = _observed_contract_multiplier(ordered, contract_multiplier,
                                                        has_fills=has_fills)
    for timestamp, group in groupby(ordered, key=lambda item: item.decision_ts_ns):
        same_time = tuple(group)
        snapshot = same_time[-1]
        multiplier = resolved_multiplier
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
            cost = _multiply_decimal_exact(
                _subtract_decimal_exact(price, reference, label="native fill slippage"),
                quantity, label="native fill slippage",
            )
            if fill["order_side"] == "SELL":
                cost = cost.copy_negate()
            cost = _multiply_decimal_exact(cost, multiplier, label="native fill slippage")
            if cost < 0 and not allow_price_improvement:
                raise MetricsError("conservative model produced favorable slippage")
            slippage = _add_decimal_exact(slippage, cost, label="native slippage sum")
            notional = _multiply_decimal_exact(
                _multiply_decimal_exact(quantity, price, label="native fill notional"),
                multiplier, label="native fill notional",
            )
            turnover = _add_decimal_exact(turnover, notional, label="native turnover sum")
            fee = _add_decimal_exact(
                fee, _money(fill["commission"], snapshot.currency), label="native fee sum",
            )
        for adjustment in _new_facts((adjustment for item in same_time
                                     for adjustment in _snapshot_records(item, "funding", "funding")),
                                     funding_facts, "funding"):
            funding = _add_decimal_exact(
                funding, _money(adjustment["pnl_change"], snapshot.currency),
                label="native funding sum",
            )
        yield {"ts_ns": timestamp, "equity": float(snapshot.equity),
               "external_cashflow": float(cashflows_by_time.get(timestamp, Decimal(0))),
               "turnover_notional": float(turnover), "fee": float(fee), "funding": float(funding),
               "slippage_cost": float(slippage)}


def account_rows(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                 external_cashflows: Sequence[Mapping[str, Any]] = (),
                 allow_price_improvement: bool = False,
                 contract_multiplier: Decimal | None = None) -> list[dict]:
    """Materialize account rows, including only caller-supplied transfer facts."""
    return list(iter_account_rows(snapshots, fill_references=fill_references,
                                  external_cashflows=external_cashflows,
                                  allow_price_improvement=allow_price_improvement,
                                  contract_multiplier=contract_multiplier))


def reconcile_linear_equity(snapshots: Sequence[Any], *,
                            external_cashflows: Sequence[Mapping[str, Any]] = (),
                            rounding_per_event: Decimal = Decimal("0.00000001"),
                            include_rows: bool = True,
                            contract_multiplier: Decimal | None = None) -> dict:
    """Independently reconcile quote-settled linear accounts with stable multipliers.

    Requires a flat, event-free starting barrier. External cashflows must be
    supplied as explicit source-identified facts aligned to unique sample barriers.
    The caller is responsible for verifying the source; the recorded SHA-256 only
    identifies the supplied source artifact.
    Check every flat barrier and, when the native snapshot contains a complete
    linear position plus its authoritative mark, check open barriers using
    unrealized price PnL and the native contract multiplier. A report must retain
    one stable multiplier for its duration. Actual fill prices already include
    execution slippage. Inverse contracts, multi-asset accounts and external
    deposits/withdrawals without independent typed source facts are outside this contract.
    """
    if (not isinstance(rounding_per_event, Decimal) or not rounding_per_event.is_finite()
            or rounding_per_event < 0 or type(include_rows) is not bool):
        raise MetricsError("finite nonnegative Decimal rounding and Boolean rows required")
    ordered = _ordered_snapshots(snapshots)
    has_fills = False
    for item in ordered:
        _snapshot_records(item, "positions", "position")
        _snapshot_records(item, "orders", "order")
        has_fills = bool(_snapshot_records(item, "fills", "fill")) or has_fills
        _snapshot_records(item, "funding", "funding")
    cashflows_by_time, cashflow_facts = _external_cashflows_by_time(external_cashflows, ordered)
    first = ordered[0]
    initial_liquidations = getattr(first, "liquidation_orders", ())
    if initial_liquidations is not None and (
            not isinstance(initial_liquidations, SequenceABC)
            or isinstance(initial_liquidations, (str, bytes, bytearray))):
        raise MetricsError("native liquidation order history must be a sequence")
    if (_snapshot_records(first, "positions", "position")
            or _snapshot_records(first, "orders", "order")
            or _snapshot_records(first, "fills", "fill")
            or _snapshot_records(first, "funding", "funding")
            or initial_liquidations):
        raise MetricsError("linear reconciliation requires a flat event-free initial barrier")
    initial = _decimal(first.equity, "initial equity")
    if initial <= 0:
        raise MetricsError("linear reconciliation requires positive initial equity")
    quantity = cash = fee = funding = external_cashflow = max_residual = Decimal(0)
    resolved_multiplier = _observed_contract_multiplier(ordered, contract_multiplier,
                                                        has_fills=has_fills)
    fills: dict[str, dict] = {}
    adjustments: dict[str, dict] = {}
    checked = open_barriers = checked_open = unchecked_open = 0
    rows = []
    for index, snapshot in enumerate(ordered):
        external_cashflow = _add_decimal_exact(
            external_cashflow, cashflows_by_time.get(snapshot.decision_ts_ns, Decimal(0)),
            label="external cashflow sum",
        )
        multiplier = resolved_multiplier
        for fill in _new_facts(_snapshot_records(snapshot, "fills", "fill"), fills, "fill"):
            size, price = _fill_values(fill)
            signed = size if fill["order_side"] == "BUY" else size.copy_negate()
            quantity = _add_decimal_exact(
                quantity, signed, label="native position quantity sum",
            )
            cash_delta = _multiply_decimal_exact(
                _multiply_decimal_exact(signed, price, label="native cash flow"),
                multiplier, label="native cash flow",
            ).copy_negate()
            cash = _add_decimal_exact(cash, cash_delta, label="native cash flow sum")
            fee = _add_decimal_exact(
                fee, _money(fill["commission"], snapshot.currency), label="native fee sum",
            )
        for adjustment in _new_facts(_snapshot_records(snapshot, "funding", "funding"),
                                     adjustments, "funding"):
            funding = _add_decimal_exact(
                funding, _money(adjustment["pnl_change"], snapshot.currency),
                label="native funding sum",
            )
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
                position_multiplier = _decimal(position.get("multiplier"), "position multiplier")
                if position_multiplier <= 0 or position_multiplier != multiplier:
                    raise MetricsError("native position multiplier is invalid or inconsistent")
                if position.get("settlement_currency") != snapshot.currency:
                    raise MetricsError("position settlement currency differs from account currency")
                side = position.get("side")
                native_size = _decimal(position.get("quantity"), "position quantity")
                if (native_size <= 0 or not isinstance(side, str)
                        or side not in {"LONG", "SHORT"}):
                    raise MetricsError("native linear position side and quantity are invalid")
                native_quantity = native_size if side == "LONG" else native_size.copy_negate()
                if native_quantity != quantity:
                    raise MetricsError("native position disagrees with cumulative fill quantity")
                mark = _decimal(mark_value, "mark price")
                if mark <= 0:
                    raise MetricsError("native mark price must be positive")
                marked_notional = _multiply_decimal_exact(
                    _multiply_decimal_exact(quantity, mark, label="native marked notional"),
                    multiplier, label="native marked notional",
                )
                expected = _sum_decimal_exact(
                    (initial, marked_notional, cash, fee.copy_negate(), funding, external_cashflow),
                    label="native linear equity",
                )
                actual = _decimal(snapshot.equity, "equity")
                residual = _subtract_decimal_exact(actual, expected, label="native equity residual")
                tolerance = _multiply_decimal_exact(
                    rounding_per_event, Decimal(len(fills) + len(adjustments) + 1),
                    label="native equity tolerance",
                )
                max_residual = max(max_residual, abs(residual))
                if abs(residual) > tolerance:
                    raise MetricsError(f"linear native equity mismatch at open barrier {index}: "
                                       f"residual={residual}, tolerance={tolerance}")
                checked_open += 1
                if include_rows:
                    rows.append({"barrier_index": index, "ts_ns": snapshot.decision_ts_ns,
                                 "native_equity": str(actual), "expected_equity": str(expected),
                                     "price_pnl_at_mark": str(_add_decimal_exact(
                                      cash, marked_notional, label="native marked price pnl",
                                  )),
                                 "mark_price": str(mark), "signed_quantity": str(quantity),
                                 "contract_multiplier": str(multiplier),
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
        expected = _sum_decimal_exact(
            (initial, cash, fee.copy_negate(), funding, external_cashflow),
            label="native linear equity",
        )
        actual = _decimal(snapshot.equity, "equity")
        residual = _subtract_decimal_exact(actual, expected, label="native equity residual")
        tolerance = _multiply_decimal_exact(
            rounding_per_event, Decimal(len(fills) + len(adjustments) + 1),
            label="native equity tolerance",
        )
        max_residual = max(max_residual, abs(residual))
        if abs(residual) > tolerance:
            raise MetricsError(f"linear native equity mismatch at barrier {index}: "
                               f"residual={residual}, tolerance={tolerance}")
        checked += 1
        if include_rows:
            rows.append({"barrier_index": index, "ts_ns": snapshot.decision_ts_ns,
                         "native_equity": str(actual), "expected_equity": str(expected),
                         "realized_price_pnl": str(cash), "fee": str(fee),
                         "funding": str(funding), "external_cashflow": str(external_cashflow),
                         "residual": str(residual),
                         "tolerance": str(tolerance)})
    result: dict = {"scope": "flat_and_marked_open_barriers_quote_settled_stable_multiplier_linear_with_explicit_cashflow_sources",
              "contract_multiplier": str(resolved_multiplier),
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
                  contract_multiplier: Decimal | None = None) -> dict:
    """Build native economic metrics without inferring external account flows."""
    if type(include_account_rows) is not bool:
        raise MetricsError("include_account_rows must be Boolean")
    ordered = _ordered_snapshots(snapshots)
    for item in ordered:
        _snapshot_records(item, "positions", "position")
        _snapshot_records(item, "orders", "order")
    accumulator = AccountFactsAccumulator(
        sample_interval_ns=sample_interval_ns,
        allow_price_improvement=allow_price_improvement,
        max_return_count=max(0, len(ordered) - 1),
    )
    rows: list[dict] | None = [] if include_account_rows else None
    for row in iter_account_rows(ordered, fill_references=fill_references,
                                 external_cashflows=external_cashflows,
                                 allow_price_improvement=allow_price_improvement,
                                 contract_multiplier=contract_multiplier):
        accumulator.add(row)
        if rows is not None:
            rows.append(row)
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
    if linear_equity_rounding is not None:
        report["equity_reconciliation"] = reconcile_linear_equity(
            ordered, external_cashflows=external_cashflows,
            rounding_per_event=linear_equity_rounding, include_rows=include_account_rows,
            contract_multiplier=contract_multiplier,
        )
    return report


class NativeReportAccumulator:
    """Build a summary report as detached native snapshots arrive.

    The accumulator retains unique event facts, the current duplicate-timestamp
    group, final native state, and bounded metric accumulators. It does not keep
    the per-barrier snapshot sequence or account rows.
    """

    def __init__(self, *, sample_interval_ns: int,
                 max_return_count: int,
                 allow_price_improvement: bool = False,
                 contract_multiplier: Decimal,
                 external_cashflows: Sequence[Mapping[str, Any]] = (),
                 linear_equity_rounding: Decimal | None = None) -> None:
        if type(allow_price_improvement) is not bool:
            raise MetricsError("allow_price_improvement must be Boolean")
        if (not isinstance(contract_multiplier, Decimal)
                or not contract_multiplier.is_finite() or contract_multiplier <= 0):
            raise MetricsError("positive finite frozen contract multiplier required")
        if (linear_equity_rounding is not None
                and (not isinstance(linear_equity_rounding, Decimal)
                     or not linear_equity_rounding.is_finite()
                     or linear_equity_rounding < 0)):
            raise MetricsError("finite nonnegative Decimal rounding required")
        self.sample_interval_ns = sample_interval_ns
        self.allow_price_improvement = allow_price_improvement
        self.contract_multiplier = contract_multiplier
        self.linear_equity_rounding = linear_equity_rounding
        self._metrics = AccountFactsAccumulator(
            sample_interval_ns=sample_interval_ns,
            allow_price_improvement=allow_price_improvement,
            max_return_count=max_return_count,
        )
        self._currency: str | None = None
        self._first_timestamp: int | None = None
        self._last_timestamp = -1
        self._group_timestamp: int | None = None
        self._group_snapshot: Any = None
        self._group_fee = Decimal(0)
        self._group_funding = Decimal(0)
        self._group_turnover = Decimal(0)
        self._group_slippage = Decimal(0)
        self._fill_facts: dict[str, dict] = {}
        self._funding_facts: dict[str, dict] = {}
        self._cashflow_facts: list[dict] = []
        self._cashflows_by_time: dict[int, Decimal] = {}
        self._matched_cashflow_timestamps = 0
        self._last_snapshot: Any = None
        self._seen = 0
        self._finished = False
        self._failure: Exception | None = None
        self._failure_context: str | None = None
        self._liquidations_observed = True
        self._liquidation_facts: dict[str, dict] = {}
        self._liquidation_filled: dict[str, Decimal] = {}
        self._prepare_cashflows(external_cashflows)
        self._init_reconciliation()

    def _prepare_cashflows(self, cashflows: Sequence[Mapping[str, Any]]) -> None:
        if (not isinstance(cashflows, SequenceABC)
                or isinstance(cashflows, (str, bytes, bytearray))):
            raise MetricsError("external cashflows must be a sequence of sourced facts")
        seen: dict[str, dict] = {}
        for fact in _new_facts(cashflows, seen, "external cashflow"):
            timestamp = fact.get("decision_ts_ns")
            source = fact.get("source_id")
            source_hash = fact.get("source_sha256")
            kind = fact.get("kind")
            if type(timestamp) is not int or timestamp < 0:
                raise MetricsError("external cashflow timestamp must be a nonnegative integer")
            if not isinstance(kind, str) or kind not in {"DEPOSIT", "WITHDRAWAL"}:
                raise MetricsError("external cashflow kind must be DEPOSIT or WITHDRAWAL")
            raw_amount = fact.get("amount")
            amount = _cashflow_decimal(raw_amount)
            if amount <= 0:
                raise MetricsError("external cashflow amount must be positive")
            if (not isinstance(source, str) or not source or source != source.strip()):
                raise MetricsError("external cashflow source identity required without surrounding whitespace")
            if (not isinstance(source_hash, str) or len(source_hash) != 64
                    or any(character not in "0123456789abcdef" for character in source_hash)):
                raise MetricsError("external cashflow source SHA-256 required")
            if not isinstance(fact.get("currency"), str):
                raise MetricsError("external cashflow currency required")
            signed = amount if kind == "DEPOSIT" else amount.copy_negate()
            self._cashflows_by_time[timestamp] = _add_cashflow(
                self._cashflows_by_time.get(timestamp, Decimal(0)), signed,
            )
            # _new_facts already detached and normalized the input mapping.
            # Retain that object directly instead of duplicating every cashflow
            # fact while the temporary identity index is still alive.
            self._cashflow_facts.append(fact)

    def _init_reconciliation(self) -> None:
        self._initial_equity = Decimal(0)
        self._quantity = Decimal(0)
        self._cash = Decimal(0)
        self._fee = Decimal(0)
        self._funding = Decimal(0)
        self._external_cashflow = Decimal(0)
        self._max_residual = Decimal(0)
        self._checked_flat = 0
        self._open_barriers = 0
        self._checked_open = 0
        self._unchecked_open = 0

    @property
    def final_snapshot(self) -> Any:
        return self._last_snapshot

    @property
    def snapshot_count(self) -> int:
        """Number of native barriers consumed so far."""
        return self._seen

    def add_snapshot(self, snapshot: Any,
                     fill_references: Mapping[str, Decimal]) -> None:
        if self._failure is not None:
            context = self._failure_context or "report operation"
            raise MetricsError(f"cannot continue after a failed {context}") from self._failure
        try:
            self._add_snapshot(snapshot, fill_references)
        except Exception as exc:
            self._failure = exc
            self._failure_context = "snapshot"
            raise

    def _add_snapshot(self, snapshot: Any,
                      fill_references: Mapping[str, Decimal]) -> None:
        if self._finished:
            raise MetricsError("cannot add a snapshot after report finalization")
        if not isinstance(fill_references, Mapping):
            raise MetricsError("native fill references must be a mapping")
        try:
            timestamp = snapshot.decision_ts_ns
            currency = snapshot.currency
            equity = snapshot.equity
        except (AttributeError, TypeError) as exc:
            raise MetricsError("native account snapshot lacks timestamp, currency, or equity") from exc
        if type(timestamp) is not int or timestamp < 0:
            raise MetricsError("native snapshot timestamp must be a nonnegative integer")
        if not isinstance(currency, str) or not currency:
            raise MetricsError("native account currency required")
        _decimal(equity, "equity")
        if timestamp < self._last_timestamp:
            raise MetricsError("native snapshots are not chronological")
        positions = _snapshot_records(snapshot, "positions", "position")
        orders = _snapshot_records(snapshot, "orders", "order")
        fill_records = _snapshot_records(snapshot, "fills", "fill")
        funding_records = _snapshot_records(snapshot, "funding", "funding")
        liquidation_capability = getattr(snapshot, "liquidation_capability", False) is True
        raw_liquidation_records = getattr(snapshot, "liquidation_orders", None)
        if (not liquidation_capability or raw_liquidation_records is None):
            self._liquidations_observed = False
            liquidation_records: Sequence[Any] = ()
        else:
            liquidation_records = _snapshot_records(
                snapshot, "liquidation_orders", "liquidation order",
            )
        for position in positions:
            if not isinstance(position, Mapping):
                raise MetricsError("native position history fact must be a mapping")
            if position.get("is_inverse") is True:
                raise MetricsError("inverse position is outside linear notional accounting")
            if position.get("is_inverse") is not False:
                raise MetricsError("native position linear capability is unknown")
            observed_multiplier = _decimal(position.get("multiplier"), "position multiplier")
            if observed_multiplier <= 0 or observed_multiplier != self.contract_multiplier:
                raise MetricsError("native position multiplier is invalid or inconsistent")
        if self._seen == 0:
            self._currency = currency
            self._first_timestamp = timestamp
            for fact in self._cashflow_facts:
                if fact.get("currency") != currency:
                    raise MetricsError("external cashflow currency differs from native account")
            if (positions or orders or fill_records or funding_records
                    or liquidation_records):
                raise MetricsError("linear reconciliation requires a flat event-free initial barrier")
            self._initial_equity = _decimal(equity, "initial equity")
            if self._initial_equity <= 0:
                raise MetricsError("linear reconciliation requires positive initial equity")
        elif currency != self._currency:
            raise MetricsError("native snapshots have conflicting currencies")
        if timestamp in self._cashflows_by_time:
            if timestamp == self._last_timestamp:
                raise MetricsError("external cashflow cannot align to duplicate native barriers")
            self._matched_cashflow_timestamps += 1
        if (timestamp == self._first_timestamp and timestamp in self._cashflows_by_time):
            raise MetricsError("external cashflow must occur after the initial barrier")

        if self._group_timestamp is not None and timestamp != self._group_timestamp:
            self._flush_account_group()
        if self._group_timestamp is None or timestamp != self._group_timestamp:
            self._group_timestamp = timestamp
            self._group_fee = Decimal(0)
            self._group_funding = Decimal(0)
            self._group_turnover = Decimal(0)
            self._group_slippage = Decimal(0)

        new_fills = tuple(_new_facts(fill_records, self._fill_facts, "fill"))
        new_funding = tuple(_new_facts(funding_records, self._funding_facts, "funding"))
        for fill in new_fills:
            quantity, price = _fill_values(fill)
            reference = fill_references.get(fill["client_order_id"])
            if reference is None:
                raise MetricsError("fill has no native simulation reference for slippage accounting")
            reference = _decimal(reference, "fill reference")
            if reference <= 0:
                raise MetricsError("native fill reference must be positive")
            cost = _multiply_decimal_exact(
                _subtract_decimal_exact(price, reference, label="native fill slippage"),
                quantity, label="native fill slippage",
            )
            if fill["order_side"] == "SELL":
                cost = cost.copy_negate()
            cost = _multiply_decimal_exact(
                cost, self.contract_multiplier, label="native fill slippage",
            )
            if cost < 0 and not self.allow_price_improvement:
                raise MetricsError("conservative model produced favorable slippage")
            self._group_slippage = _add_decimal_exact(
                self._group_slippage, cost, label="native slippage sum",
            )
            notional = _multiply_decimal_exact(
                _multiply_decimal_exact(quantity, price, label="native fill notional"),
                self.contract_multiplier, label="native fill notional",
            )
            self._group_turnover = _add_decimal_exact(
                self._group_turnover, notional, label="native turnover sum",
            )
            self._group_fee = _add_decimal_exact(
                self._group_fee, _money(fill["commission"], currency), label="native fee sum",
            )
        for adjustment in new_funding:
            self._group_funding = _add_decimal_exact(
                self._group_funding, _money(adjustment["pnl_change"], currency),
                label="native funding sum",
            )

        if self._liquidations_observed:
            for order in liquidation_records:
                _update_liquidation_order(
                    order, self._liquidation_facts, self._liquidation_filled,
                )

        if self.linear_equity_rounding is not None:
            self._reconcile_snapshot(snapshot, positions, orders, new_fills, new_funding)
        self._group_snapshot = snapshot
        self._last_snapshot = snapshot
        self._last_timestamp = timestamp
        self._seen += 1

    def _flush_account_group(self) -> None:
        if self._group_timestamp is None or self._group_snapshot is None:
            return
        timestamp = self._group_timestamp
        flow = self._cashflows_by_time.get(timestamp, Decimal(0))
        if timestamp in self._cashflows_by_time:
            if timestamp <= (self._first_timestamp if self._first_timestamp is not None else timestamp):
                raise MetricsError("external cashflow must align to a sample after the initial barrier")
        snapshot = self._group_snapshot
        self._metrics.add({
            "ts_ns": timestamp, "equity": float(snapshot.equity),
            "external_cashflow": float(flow),
            "turnover_notional": float(self._group_turnover),
            "fee": float(self._group_fee), "funding": float(self._group_funding),
            "slippage_cost": float(self._group_slippage),
        })
        self._group_snapshot = None

    def _reconcile_snapshot(self, snapshot: Any, positions: Sequence[Any], orders: Sequence[Any],
                            new_fills: Sequence[dict],
                            new_funding: Sequence[dict]) -> None:
        rounding = self.linear_equity_rounding
        if rounding is None:
            raise MetricsError("linear equity reconciliation rounding is not configured")
        if self._seen == 0 and (positions or orders or new_fills or new_funding):
            raise MetricsError("linear reconciliation requires a flat event-free initial barrier")
        for position in positions:
            if not isinstance(position, Mapping):
                raise MetricsError("native position history fact must be a mapping")
            if position.get("is_inverse") is True:
                raise MetricsError("inverse position is outside linear equity reconciliation")
            if position.get("is_inverse") is not False:
                raise MetricsError("native position inverse capability is unknown")
            multiplier = _decimal(position.get("multiplier"), "position multiplier")
            if multiplier <= 0 or multiplier != self.contract_multiplier:
                raise MetricsError("native position multiplier is invalid or inconsistent")

        for fill in new_fills:
            size, price = _fill_values(fill)
            signed = size if fill["order_side"] == "BUY" else size.copy_negate()
            self._quantity = _add_decimal_exact(
                self._quantity, signed, label="native position quantity sum",
            )
            cash_delta = _multiply_decimal_exact(
                _multiply_decimal_exact(signed, price, label="native cash flow"),
                self.contract_multiplier, label="native cash flow",
            ).copy_negate()
            self._cash = _add_decimal_exact(self._cash, cash_delta, label="native cash flow sum")
            self._fee = _add_decimal_exact(
                self._fee, _money(fill["commission"], snapshot.currency), label="native fee sum",
            )
        for adjustment in new_funding:
            self._funding = _add_decimal_exact(
                self._funding, _money(adjustment["pnl_change"], snapshot.currency),
                label="native funding sum",
            )

        flow = self._cashflows_by_time.get(snapshot.decision_ts_ns, Decimal(0))
        self._external_cashflow = _add_decimal_exact(
            self._external_cashflow, flow, label="external cashflow sum",
        )
        if self._quantity != 0:
            self._open_barriers += 1
            mark_value = getattr(snapshot, "mark", getattr(snapshot, "mark_price", None))
            if mark_value is not None and positions:
                if len(positions) != 1:
                    raise MetricsError("linear reconciliation requires one open native position")
                position = positions[0]
                if position.get("settlement_currency") != snapshot.currency:
                    raise MetricsError("position settlement currency differs from account currency")
                side = position.get("side")
                native_size = _decimal(position.get("quantity"), "position quantity")
                if (native_size <= 0 or not isinstance(side, str)
                        or side not in {"LONG", "SHORT"}):
                    raise MetricsError("native linear position side and quantity are invalid")
                native_quantity = native_size if side == "LONG" else native_size.copy_negate()
                if native_quantity != self._quantity:
                    raise MetricsError("native position disagrees with cumulative fill quantity")
                mark = _decimal(mark_value, "mark price")
                if mark <= 0:
                    raise MetricsError("native mark price must be positive")
                marked_notional = _multiply_decimal_exact(
                    _multiply_decimal_exact(
                        self._quantity, mark, label="native marked notional",
                    ),
                    self.contract_multiplier, label="native marked notional",
                )
                expected = _sum_decimal_exact(
                    (self._initial_equity, marked_notional, self._cash,
                     self._fee.copy_negate(), self._funding, self._external_cashflow),
                    label="native linear equity",
                )
                actual = _decimal(snapshot.equity, "equity")
                residual = _subtract_decimal_exact(actual, expected, label="native equity residual")
                tolerance = _multiply_decimal_exact(
                    rounding, Decimal(len(self._fill_facts) + len(self._funding_facts) + 1),
                    label="native equity tolerance",
                )
                self._max_residual = max(self._max_residual, abs(residual))
                if abs(residual) > tolerance:
                    raise MetricsError(f"linear native equity mismatch at open barrier {self._seen}: "
                                       f"residual={residual}, tolerance={tolerance}")
                self._checked_open += 1
            else:
                self._unchecked_open += 1
            return
        if positions:
            raise MetricsError("flat fill quantity disagrees with native positions")
        expected = _sum_decimal_exact(
            (self._initial_equity, self._cash, self._fee.copy_negate(),
             self._funding, self._external_cashflow),
            label="native linear equity",
        )
        actual = _decimal(snapshot.equity, "equity")
        residual = _subtract_decimal_exact(actual, expected, label="native equity residual")
        tolerance = _multiply_decimal_exact(
            rounding, Decimal(len(self._fill_facts) + len(self._funding_facts) + 1),
            label="native equity tolerance",
        )
        self._max_residual = max(self._max_residual, abs(residual))
        if abs(residual) > tolerance:
            raise MetricsError(f"linear native equity mismatch at barrier {self._seen}: "
                               f"residual={residual}, tolerance={tolerance}")
        self._checked_flat += 1

    def finish(self) -> dict:
        if self._failure is not None:
            context = self._failure_context or "report operation"
            raise MetricsError(f"cannot continue after a failed {context}") from self._failure
        try:
            return self._finish_report()
        except Exception as exc:
            self._failure = exc
            self._failure_context = "report finalization"
            raise

    def _finish_report(self) -> dict:
        if self._finished:
            raise MetricsError("native summary report was already finalized")
        if self._seen == 0:
            raise MetricsError("native account snapshots required")
        self._flush_account_group()
        if self._matched_cashflow_timestamps != len(self._cashflows_by_time):
            raise MetricsError("external cashflow must align to exactly one native barrier")
        for timestamp in self._cashflows_by_time:
            if self._first_timestamp is not None and timestamp <= self._first_timestamp:
                raise MetricsError("external cashflow must align to a sample after the initial barrier")
        self._finished = True
        metrics = self._metrics.result(
            periods_per_year=365.25 * 86400 * 1e9 / self.sample_interval_ns,
        )
        final = self._last_snapshot
        final_positions = _snapshot_records(final, "positions", "position")
        final_orders = _snapshot_records(final, "orders", "order")
        report = {
            "metrics": metrics,
            "fills": list(self._fill_facts.values()),
            "funding": list(self._funding_facts.values()),
            "terminal_flat": not final_positions and not final_orders,
            "external_cashflows": self._cashflow_facts,
            "liquidations": len(self._liquidation_facts) if self._liquidations_observed else None,
            "liquidations_observed": self._liquidations_observed,
            "liquidation_orders": (list(self._liquidation_facts.values())
                                   if self._liquidations_observed else None),
            "costs_already_in_native_equity": True,
        }
        if self.linear_equity_rounding is not None:
            if self._quantity != 0 and not final_positions:
                raise MetricsError("terminal position disagrees with native fill quantity")
            report["equity_reconciliation"] = {
                "scope": "flat_and_marked_open_barriers_quote_settled_stable_multiplier_linear_with_explicit_cashflow_sources",
                "contract_multiplier": str(self.contract_multiplier),
                "checked_flat_barriers": self._checked_flat,
                "open_barriers": self._open_barriers,
                "unchecked_open_barriers": self._unchecked_open,
                "checked_open_barriers": self._checked_open,
                "terminal_checked": self._quantity == 0,
                "max_abs_residual": str(self._max_residual),
                "rounding_per_event": str(self.linear_equity_rounding),
                "unique_fills": len(self._fill_facts),
                "unique_funding": len(self._funding_facts),
                "unique_external_cashflows": len(self._cashflow_facts),
                "net_external_cashflow": str(self._external_cashflow),
            }
        return report
