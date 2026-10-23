"""Economic-only metrics rebuilt from native account/fill facts; never from shaped reward."""
from __future__ import annotations

import heapq
import math
import numpy as np
from array import array
from collections.abc import Iterable, Iterator, Sequence, Mapping
import struct
import tempfile
from .numeric import finite_real


class MetricsError(ValueError):
    pass


class _DiskBackedFloatSequence:
    """Replayable finite-double sequence stored in an automatically deleted temp file."""

    _FLOAT_BYTES = 8
    _CHUNK_BYTES = 8192
    _WRITE_BUFFER_BYTES = 64 * 1024

    def __init__(self) -> None:
        self._file = tempfile.TemporaryFile(mode="w+b")
        self._write_buffer = bytearray()
        self._count = 0

    def append(self, value: float) -> None:
        self._write_buffer.extend(struct.pack("=d", value))
        self._count += 1
        if len(self._write_buffer) >= self._WRITE_BUFFER_BYTES:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if self._write_buffer:
            self._file.seek(0, 2)
            self._file.write(self._write_buffer)
            self._write_buffer.clear()

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        self._file.close()

    def __iter__(self) -> Iterator[float]:
        self._flush_pending()
        self._file.seek(0)
        while block := self._file.read(self._CHUNK_BYTES):
            if len(block) % self._FLOAT_BYTES:
                raise MetricsError("temporary return spool has a truncated float")
            yield from (item[0] for item in struct.iter_unpack("=d", block))


def _number(v, name: str) -> float:
    if not finite_real(v):
        raise MetricsError(f'{name} must be finite numeric')
    try:
        result = float(v)
    except (OverflowError, TypeError, ValueError) as exc:
        raise MetricsError(f'{name} must be finite numeric') from exc
    if not math.isfinite(result):
        raise MetricsError(f'{name} must be finite numeric')
    return result


def _finite_sum(current: float, increment: float, label: str) -> float:
    if not math.isfinite(increment):
        raise MetricsError(f"{label} increment overflow")
    total = current + increment
    if not math.isfinite(total):
        raise MetricsError(f"{label} accumulation overflow")
    return total


_TAIL_HEAP_MAX = 8_192


_TAIL_RADIX_CHUNK_FLOATS = 65_536
_UINT64_SIGN_BIT = np.uint64(1 << 63)
_UINT64_MAX = np.uint64((1 << 64) - 1)


def _ordered_float_keys(values: np.ndarray) -> np.ndarray:
    """Map finite float64 values to unsigned keys in numeric sort order."""
    bits = values.view(np.uint64)
    negative = (bits & _UINT64_SIGN_BIT) != 0
    keys = np.where(negative, bits ^ _UINT64_MAX, bits ^ _UINT64_SIGN_BIT)
    # Python considers both signed zeros equal. Keep their common stable-input
    # tie group together so radix selection agrees with the in-memory heap.
    keys[values == 0.0] = _UINT64_SIGN_BIT
    return keys


def _disk_backed_tail(returns: _DiskBackedFloatSequence,
                      count: int) -> tuple[float, float]:
    """Select exact VaR/CVaR with fixed-size radix scans over the disk spool."""
    if count < 1 or count > len(returns):
        raise MetricsError("return tail count is outside the available series")
    returns._flush_pending()
    returns._file.flush()

    # Eight most-significant-byte passes identify the exact (count - 1)th
    # ordered float key. Each pass retains only one bounded NumPy chunk and a
    # 256-entry histogram, rather than mapping/touching the full history in RAM.
    target_rank = count - 1
    selected_before = 0
    prefix = 0
    prefix_mask = 0
    for shift in range(56, -1, -8):
        histogram = np.zeros(256, dtype=np.int64)
        returns._file.seek(0)
        remaining = len(returns)
        while remaining:
            size = min(_TAIL_RADIX_CHUNK_FLOATS, remaining)
            values = np.fromfile(returns._file, dtype=np.float64, count=size)
            if values.size != size:
                raise MetricsError("temporary return spool has a truncated float")
            keys = _ordered_float_keys(values)
            candidates = (keys & np.uint64(prefix_mask)) == np.uint64(prefix)
            byte = ((keys[candidates] >> np.uint64(shift)) & np.uint64(0xFF)).astype(
                np.uint8, copy=False,
            )
            histogram += np.bincount(byte, minlength=256)
            remaining -= size

        cumulative = np.cumsum(histogram, dtype=np.int64)
        bucket = int(np.searchsorted(cumulative, target_rank + 1, side="left"))
        if bucket >= 256 or histogram[bucket] == 0:
            raise MetricsError("radix tail selection could not locate its boundary")
        below = int(cumulative[bucket - 1]) if bucket else 0
        selected_before += below
        target_rank -= below
        prefix |= bucket << shift
        prefix_mask |= 0xFF << shift

    boundary_key = np.uint64(prefix)
    remaining_equal = count - selected_before
    boundary_rank = target_rank
    threshold: float | None = None

    def selected_values() -> Iterator[float]:
        nonlocal remaining_equal, boundary_rank, threshold
        returns._file.seek(0)
        remaining = len(returns)
        while remaining:
            size = min(_TAIL_RADIX_CHUNK_FLOATS, remaining)
            values = np.fromfile(returns._file, dtype=np.float64, count=size)
            if values.size != size:
                raise MetricsError("temporary return spool has a truncated float")
            keys = _ordered_float_keys(values)
            lower = keys < boundary_key
            equal_positions = np.flatnonzero(keys == boundary_key)
            if threshold is None:
                if boundary_rank < equal_positions.size:
                    threshold = float(values[equal_positions[boundary_rank]])
                else:
                    boundary_rank -= int(equal_positions.size)
            take_equal = min(remaining_equal, int(equal_positions.size))
            selected = lower.copy()
            if take_equal:
                selected[equal_positions[:take_equal]] = True
                remaining_equal -= take_equal
            yield from (float(value) for value in values[selected])
            remaining -= size

    tail_mean = math.fsum(selected_values()) / count
    if remaining_equal or threshold is None:
        raise MetricsError("radix tail selection produced an incomplete tail")
    return threshold, tail_mean


class _StreamingSeriesMetrics:
    """Incremental economic metrics with a fixed tail-heap cap."""

    def __init__(self, *, max_count: int, risk_free_per_period: float,
                 target_per_period: float, tail_quantile: float) -> None:
        if (type(max_count) is not int or max_count < 0
                or any(not finite_real(v) for v in (
                    risk_free_per_period, target_per_period, tail_quantile,
                ))
                or not 0 < tail_quantile < 0.5):
            raise MetricsError("invalid streaming series metric bounds")
        self.max_count = max_count
        self.risk_free_per_period = _number(risk_free_per_period, "risk-free return")
        self.target_per_period = _number(target_per_period, "target return")
        self.tail_quantile = _number(tail_quantile, "tail quantile")
        self.tail_capacity = math.ceil(max_count * self.tail_quantile)
        self.worst_heap: list[float] | None = (
            None if self.tail_capacity > _TAIL_HEAP_MAX else []
        )
        self.returns = (
            _DiskBackedFloatSequence() if self.worst_heap is None else None
        )
        self.path = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        self.mean = 0.0
        self.m2_scale, self.m2_ssq = 0.0, 1.0
        self.downside_scale, self.downside_ssq = 0.0, 1.0
        self.count = 0

    def add(self, value: float) -> None:
        if self.count >= self.max_count:
            raise MetricsError("return sequence exceeds declared maximum length")
        x = _number(value, "return")
        if x <= -1:
            raise MetricsError("returns <= -100% cannot be compounded safely")
        index = self.count + 1
        delta = x - self.mean
        self.mean += delta / index
        variance_term = math.sqrt(abs(delta)) * math.sqrt(abs(x - self.mean))
        if variance_term:
            if self.m2_scale < variance_term:
                self.m2_ssq = 1 + self.m2_ssq * (self.m2_scale / variance_term) ** 2
                self.m2_scale = variance_term
            else:
                self.m2_ssq += (variance_term / self.m2_scale) ** 2
        shortfall = self.target_per_period - x
        if not math.isfinite(shortfall):
            raise MetricsError("downside deviation overflow")
        downside = max(0.0, shortfall)
        if downside:
            if self.downside_scale < downside:
                self.downside_ssq = 1 + self.downside_ssq * (self.downside_scale / downside) ** 2
                self.downside_scale = downside
            else:
                self.downside_ssq += (downside / self.downside_scale) ** 2
        if self.returns is not None:
            self.returns.append(x)
        else:
            assert self.worst_heap is not None
            if len(self.worst_heap) < self.tail_capacity:
                heapq.heappush(self.worst_heap, -x)
            elif self.worst_heap and x < -self.worst_heap[0]:
                heapq.heapreplace(self.worst_heap, -x)
        self.path *= 1 + x
        if not math.isfinite(self.path) or self.path <= 0:
            raise MetricsError("equity compounding overflow/underflow")
        self.peak = max(self.peak, self.path)
        self.drawdown = max(self.drawdown, 1 - self.path / self.peak)
        self.count = index

    def release(self) -> None:
        """Release bounded heap or disk-backed return storage after use."""
        if self.returns is not None:
            self.returns.close()
            self.returns = None
        if self.worst_heap is not None:
            self.worst_heap.clear()
            self.worst_heap = None

    def result(self, *, periods_per_year: float,
               require_exact_count: bool = False) -> dict[str, object]:
        count = self.count
        if (count < 2 or not finite_real(periods_per_year)
                or periods_per_year <= 0
                or (require_exact_count and count != self.max_count)):
            raise MetricsError("at least 2 returns and valid frequency/quantile required")
        tail_count = max(1, math.ceil(count * self.tail_quantile))
        if self.returns is not None:
            if len(self.returns) != count:
                raise MetricsError("return spool count disagrees with streaming metrics")
            threshold, tail_mean = _disk_backed_tail(self.returns, tail_count)
            value_at_risk = -threshold
            conditional_value_at_risk = -tail_mean
        else:
            assert self.worst_heap is not None
            if len(self.worst_heap) < tail_count:
                raise MetricsError("return tail does not cover the requested quantile")
            tail = list(self.worst_heap)
            while len(tail) > tail_count:
                heapq.heappop(tail)
            value_at_risk = tail[0]
            conditional_value_at_risk = math.fsum(tail) / len(tail)
        sd = self.m2_scale * math.sqrt(self.m2_ssq / (count - 1))
        downside_deviation = self.downside_scale * math.sqrt(self.downside_ssq / count)
        if not math.isfinite(sd) or not math.isfinite(downside_deviation):
            raise MetricsError("return dispersion overflow")
        sortino = ((self.mean - self.target_per_period) / downside_deviation
                   * math.sqrt(periods_per_year) if downside_deviation > 0 else None)
        sharpe = ((self.mean - self.risk_free_per_period) / sd
                  * math.sqrt(periods_per_year) if sd > 0 else None)
        if any(value is not None and not math.isfinite(value) for value in (sortino, sharpe)):
            raise MetricsError("annualized return metric overflow")
        return {
            "net_return": self.path - 1,
            "max_drawdown": self.drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "mean_period_return": self.mean,
            "std_period_return": sd,
            "undefined_metrics": [metric for metric, val in (("sharpe", sharpe), ("sortino", sortino))
                                  if val is None],
            "value_at_risk": value_at_risk,
            "conditional_value_at_risk": conditional_value_at_risk,
            "observations": count,
        }


def series_metrics(returns: Sequence[float], *, periods_per_year: float,
                   risk_free_per_period: float = 0.0, target_per_period: float = 0.0,
                   tail_quantile: float = 0.05) -> dict[str, object]:
    count = len(returns)
    if (count < 2 or any(not finite_real(v)
                         for v in (periods_per_year, risk_free_per_period,
                                   target_per_period, tail_quantile))
            or periods_per_year <= 0 or not 0 < tail_quantile < 0.5):
        raise MetricsError("at least 2 returns and valid frequency/quantile required")
    accumulator = _StreamingSeriesMetrics(
        max_count=count,
        risk_free_per_period=risk_free_per_period,
        target_per_period=target_per_period,
        tail_quantile=tail_quantile,
    )
    for value in returns:
        accumulator.add(value)
    return accumulator.result(periods_per_year=periods_per_year, require_exact_count=True)


class AccountFactsAccumulator:
    """Validate account rows incrementally without retaining normalized row dictionaries."""

    _REQUIRED = {'ts_ns', 'equity', 'external_cashflow', 'turnover_notional',
                 'fee', 'funding', 'slippage_cost'}

    def __init__(self, *, sample_interval_ns: int,
                 allow_price_improvement: bool = False,
                 max_return_count: int | None = None) -> None:
        if type(allow_price_improvement) is not bool:
            raise MetricsError('allow_price_improvement must be Boolean')
        if (isinstance(sample_interval_ns, bool) or not isinstance(sample_interval_ns, int)
                or sample_interval_ns < 1):
            raise MetricsError('regular account snapshots and explicit sampling interval required')
        self.sample_interval_ns = sample_interval_ns
        self.allow_price_improvement = allow_price_improvement
        self.last_ts = -1
        self.previous: dict[str, float] | None = None
        self.first_equity = 0.0
        self.last_equity = 0.0
        self.external_cashflow = 0.0
        self.turnover = 0.0
        self.totals = {'fee': 0.0, 'funding': 0.0, 'slippage_cost': 0.0}
        if max_return_count is not None and (type(max_return_count) is not int
                                             or max_return_count < 0):
            raise MetricsError('maximum account return count must be a nonnegative integer')
        self._return_metrics = (_StreamingSeriesMetrics(
            max_count=max_return_count, risk_free_per_period=0.0,
            target_per_period=0.0, tail_quantile=0.05,
        ) if max_return_count is not None else None)
        self.returns: array[float] | None = (
            None if self._return_metrics is not None else array('d')
        )
        self.count = 0
        self._failure: Exception | None = None
        self._released = False

    def add(self, fact: Mapping[str, object]) -> None:
        if self._released:
            raise MetricsError('cannot update a released account-facts accumulator')
        if self._failure is not None:
            raise MetricsError("cannot continue after a failed account fact") from self._failure
        try:
            self._add(fact)
        except Exception as exc:
            # Return metrics and the disk-backed sequence are not generally
            # reversible after a failed update. Refuse reuse so a caller cannot
            # publish a report assembled from partially committed state.
            self._failure = exc
            raise

    def _add(self, fact: Mapping[str, object]) -> None:
        if not self._REQUIRED.issubset(fact):
            raise MetricsError('account facts missing required fields')
        ts = fact['ts_ns']
        if isinstance(ts, bool) or not isinstance(ts, int) or ts <= self.last_ts:
            raise MetricsError('account time order invalid')
        if self.last_ts != -1 and ts - self.last_ts != self.sample_interval_ns:
            raise MetricsError('irregular account sampling; annualization would be invalid')
        row = {key: _number(fact[key], key) for key in self._REQUIRED if key != 'ts_ns'}
        # Native commission is signed: negative values represent exchange rebates.
        if (row['equity'] <= 0 or row['turnover_notional'] < 0
                or (row['slippage_cost'] < 0 and not self.allow_price_improvement)):
            raise MetricsError('invalid account values / exhausted equity')
        if self.previous is None:
            self.first_equity = row['equity']
        else:
            adjusted = (row['equity'] - row['external_cashflow']) / self.previous['equity'] - 1
            if not math.isfinite(adjusted):
                raise MetricsError('cashflow-adjusted period return nonfinite')
            turnover_increment = row['turnover_notional'] / self.previous['equity']
            next_turnover = _finite_sum(self.turnover, turnover_increment, "turnover ratio")
            next_totals = {
                field: _finite_sum(self.totals[field], row[field], f"total {field}")
                for field in self.totals
            }
            next_external_cashflow = _finite_sum(
                self.external_cashflow, row['external_cashflow'], "external cashflow",
            )
            if self._return_metrics is None:
                assert self.returns is not None
                self.returns.append(adjusted)
            else:
                self._return_metrics.add(adjusted)
            self.turnover = next_turnover
            self.totals.update(next_totals)
            self.external_cashflow = next_external_cashflow
        self.previous = row
        self.last_equity = row['equity']
        self.last_ts = ts
        self.count += 1

    def result(self, *, periods_per_year: float,
               realized_trade_pnls: Sequence[float] | None = None) -> dict:
        if self._released:
            raise MetricsError('cannot report from a released account-facts accumulator')
        if self._failure is not None:
            raise MetricsError("cannot report after a failed account fact") from self._failure
        if self.count < 3:
            raise MetricsError('regular account snapshots and explicit sampling interval required')
        if self._return_metrics is None:
            assert self.returns is not None
            result = series_metrics(self.returns, periods_per_year=periods_per_year)
        else:
            result = self._return_metrics.result(periods_per_year=periods_per_year)
        result['turnover_ratio'] = self.turnover
        for field, value in self.totals.items():
            result[f'total_{field}'] = value
        try:
            gross_return = math.fsum((
                self.last_equity, -self.first_equity, -self.external_cashflow,
                self.totals['fee'], self.totals['slippage_cost'], -self.totals['funding'],
            )) / self.first_equity
        except OverflowError as exc:
            raise MetricsError('gross return aggregation overflow') from exc
        if not math.isfinite(gross_return):
            raise MetricsError('gross return aggregation overflow')
        result['gross_return_before_reported_costs'] = gross_return
        if realized_trade_pnls is not None:
            pnl = [_number(x, 'realized_pnl') for x in realized_trade_pnls]
            try:
                gross_profit = math.fsum(x for x in pnl if x > 0)
                gross_loss = -math.fsum(x for x in pnl if x < 0)
            except OverflowError as exc:
                raise MetricsError('realized trade PnL aggregation overflow') from exc
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
            if profit_factor is not None and not math.isfinite(profit_factor):
                raise MetricsError('trade profit factor overflow')
            result['trade_profit_factor'] = profit_factor
            result['closed_trade_count'] = len(pnl)
        return result

    def release(self) -> None:
        """Release retained return storage after its result has been captured."""
        if self._released:
            return
        self._released = True
        if self._return_metrics is not None:
            self._return_metrics.release()
            self._return_metrics = None
        self.returns = None
        self.previous = None


def from_account_facts(facts: Iterable[Mapping[str, object]], *,
                       periods_per_year: float,
                       realized_trade_pnls: Sequence[float] | None = None,
                       sample_interval_ns: int,
                       allow_price_improvement: bool = False) -> dict:
    """Build economic metrics from regular native-account snapshots.

    Funding keeps its signed cashflow convention. External cashflow is treated as
    occurring at the sample boundary, so callers must split mid-period flows.
    All costs already entered native equity and must not be subtracted again.
    """
    accumulator = AccountFactsAccumulator(sample_interval_ns=sample_interval_ns,
                                          allow_price_improvement=allow_price_improvement)
    for fact in facts:
        accumulator.add(fact)
    return accumulator.result(periods_per_year=periods_per_year,
                              realized_trade_pnls=realized_trade_pnls)
