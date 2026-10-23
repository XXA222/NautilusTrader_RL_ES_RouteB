"""Economic-only metrics rebuilt from native account/fill facts; never from shaped reward."""
from __future__ import annotations

import math
import statistics
import heapq
from itertools import islice
from pathlib import Path
import struct
import tempfile
from collections.abc import Generator, Iterable, Iterator, Sequence, Mapping
from typing import overload
from .numeric import finite_real


class MetricsError(ValueError):
    pass


class _DiskBackedFloatSequence(Sequence[float]):
    """Replayable float sequence stored in an automatically deleted temp file."""

    _FLOAT_BYTES = 8
    _CHUNK_BYTES = 8192
    _WRITE_BUFFER_BYTES = 64 * 1024

    def __init__(self) -> None:
        self._file = tempfile.TemporaryFile(mode="w+b")
        self._write_buffer = bytearray()
        self._count = 0

    def append(self, value: float) -> None:
        self._write_buffer.extend(struct.pack("!d", value))
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
            yield from (item[0] for item in struct.iter_unpack("!d", block))

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[float]: ...

    def __getitem__(self, index: int | slice) -> float | Sequence[float]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("return sequence indices must be integers or slices")
        position = index + self._count if index < 0 else index
        if not 0 <= position < self._count:
            raise IndexError("return sequence index out of range")
        self._flush_pending()
        self._file.seek(position * self._FLOAT_BYTES)
        value = self._file.read(self._FLOAT_BYTES)
        if len(value) != self._FLOAT_BYTES:
            raise MetricsError("temporary return spool has a truncated float")
        return struct.unpack("!d", value)[0]


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


_TAIL_SORT_CHUNK = 8_192
_TAIL_MERGE_FAN_IN = 32
_TAIL_HEAP_MAX_COUNT = 8_192


def _iter_float_file(path: Path) -> Generator[float, None, None]:
    with path.open("rb") as source:
        while block := source.read(_DiskBackedFloatSequence._CHUNK_BYTES):
            if len(block) % _DiskBackedFloatSequence._FLOAT_BYTES:
                raise MetricsError("temporary return run has a truncated float")
            yield from (item[0] for item in struct.iter_unpack("!d", block))


def _write_float_block(target, values: Sequence[float]) -> None:
    if values:
        target.write(struct.pack(f"!{len(values)}d", *values))


def _write_sorted_run(path: Path, values: list[float], limit: int) -> int:
    values.sort()
    selected = values[:limit]
    with path.open("wb") as target:
        _write_float_block(target, selected)
    return len(selected)


def _merge_float_runs(runs: Sequence[tuple[Path, int]], output: Path,
                      limit: int) -> int:
    ordered = sorted(runs, key=lambda run: run[1])
    iterators = [_iter_float_file(path) for path, _ in ordered]
    written = 0
    block: list[float] = []
    try:
        merged = heapq.merge(*iterators)
        with output.open("wb") as target:
            for value in islice(merged, limit):
                block.append(value)
                written += 1
                if len(block) == _TAIL_SORT_CHUNK:
                    _write_float_block(target, block)
                    block.clear()
            _write_float_block(target, block)
    finally:
        for iterator in iterators:
            iterator.close()
    return written


def _external_tail(returns: Iterable[float], count: int) -> tuple[float, float]:
    """Select exact VaR/CVaR with fixed-size sorted runs and bounded fan-in."""
    with tempfile.TemporaryDirectory(prefix="rles-vartail-") as directory:
        root = Path(directory)
        levels: list[list[tuple[Path, int]]] = []
        next_run = 0

        def new_path() -> Path:
            nonlocal next_run
            path = root / f"run-{next_run:08d}.bin"
            next_run += 1
            return path

        def add_run(path: Path, start: int) -> None:
            level = 0
            run = (path, start)
            while True:
                if level == len(levels):
                    levels.append([])
                levels[level].append(run)
                if len(levels[level]) < _TAIL_MERGE_FAN_IN:
                    return
                group = levels[level]
                merged_path = new_path()
                _merge_float_runs(group, merged_path, count)
                for old_path, _ in group:
                    old_path.unlink()
                run = (merged_path, group[0][1])
                levels[level] = []
                level += 1

        chunk: list[float] = []
        chunk_start = 0
        position = 0
        for value in returns:
            if not chunk:
                chunk_start = position
            chunk.append(value)
            position += 1
            if len(chunk) == _TAIL_SORT_CHUNK:
                add_run(_write_chunk(new_path(), chunk, count), chunk_start)
                chunk = []
        if chunk:
            add_run(_write_chunk(new_path(), chunk, count), chunk_start)

        runs = sorted((run for level in levels for run in level), key=lambda run: run[1])
        while len(runs) > 1:
            merged_runs: list[tuple[Path, int]] = []
            for start in range(0, len(runs), _TAIL_MERGE_FAN_IN):
                group = runs[start:start + _TAIL_MERGE_FAN_IN]
                if len(group) == 1:
                    merged_runs.append(group[0])
                    continue
                output = new_path()
                _merge_float_runs(group, output, count)
                for old_path, _ in group:
                    old_path.unlink()
                merged_runs.append((output, group[0][1]))
            runs = merged_runs

        final_path, _ = runs[0]
        last: float | None = None

        def selected_values() -> Iterator[float]:
            nonlocal last
            for value in _iter_float_file(final_path):
                last = value
                yield value

        tail_mean = statistics.fmean(selected_values())
        if last is None or position < count:
            raise MetricsError("return tail selection produced an incomplete tail")
        return last, tail_mean


def _write_chunk(path: Path, values: list[float], limit: int) -> Path:
    _write_sorted_run(path, values, limit)
    return path


def series_metrics(returns: Sequence[float], *, periods_per_year: float,
                   risk_free_per_period: float = 0.0, target_per_period: float = 0.0,
                   tail_quantile: float = 0.05) -> dict[str, object]:
    if (len(returns) < 2 or any(not finite_real(v)
                                for v in (periods_per_year, risk_free_per_period,
                                          target_per_period, tail_quantile))
            or periods_per_year <= 0 or not 0 < tail_quantile < 0.5):
        raise MetricsError('at least 2 returns and valid frequency/quantile required')
    if isinstance(returns, _DiskBackedFloatSequence):
        r: Sequence[float] = returns
        if any(not finite_real(x) for x in r):
            raise MetricsError('return must be finite numeric')
        if any(x <= -1 for x in r):
            raise MetricsError('returns <= -100% cannot be compounded safely')
    else:
        r = [_number(x, 'return') for x in returns]
        if any(x <= -1 for x in r):
            raise MetricsError('returns <= -100% cannot be compounded safely')
    path = 1.0
    peak = 1.0
    dd = 0.0
    for x in r:
        path *= 1 + x
        if not math.isfinite(path) or path <= 0:
            raise MetricsError('equity compounding overflow/underflow')
        peak = max(peak, path)
        dd = max(dd, 1 - path / peak)
    mean = statistics.fmean(r)
    downside_deviation = math.sqrt(statistics.fmean(
        min(0, x - target_per_period) ** 2 for x in r
    ))
    sortino = ((mean - target_per_period) / downside_deviation
               * math.sqrt(periods_per_year) if downside_deviation > 0 else None)
    sd = statistics.stdev(r, xbar=mean)
    sharpe = ((mean - risk_free_per_period) / sd
              * math.sqrt(periods_per_year) if sd > 0 else None)
    count = max(1, math.ceil(len(r) * tail_quantile))
    if isinstance(r, _DiskBackedFloatSequence) or count > _TAIL_HEAP_MAX_COUNT:
        threshold, tail_mean = _external_tail(r, count)
    else:
        worst = heapq.nsmallest(count, r)
        threshold = worst[-1]
        tail_mean = statistics.fmean(worst)
    result: dict[str, object] = {
        'net_return': path - 1,
        'max_drawdown': dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'mean_period_return': statistics.fmean(r),
        'std_period_return': sd,
        'undefined_metrics': [metric for metric, val in (('sharpe', sharpe), ('sortino', sortino))
                              if val is None],
        'value_at_risk': -threshold,
        'conditional_value_at_risk': -tail_mean,
        'observations': len(r),
    }
    numeric_fields = (
        'net_return', 'max_drawdown', 'sharpe', 'sortino', 'mean_period_return',
        'std_period_return', 'value_at_risk', 'conditional_value_at_risk',
    )
    if any(result[field] is not None and not finite_real(result[field])
           for field in numeric_fields):
        raise MetricsError('series metrics produced a nonfinite result')
    return result


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
        if (max_return_count is not None
                and (type(max_return_count) is not int or max_return_count < 1)):
            raise MetricsError('maximum return count must be a positive integer')
        self.sample_interval_ns = sample_interval_ns
        self.allow_price_improvement = allow_price_improvement
        self.max_return_count = max_return_count
        self.last_ts = -1
        self.previous: dict[str, float] | None = None
        self.first_equity = 0.0
        self.last_equity = 0.0
        self.external_cashflow = 0.0
        self.turnover = 0.0
        self.totals = {'fee': 0.0, 'funding': 0.0, 'slippage_cost': 0.0}
        # Replay from a temporary spool to preserve exact multi-pass statistics
        # without keeping every return in Python heap memory.
        self.returns = _DiskBackedFloatSequence()
        self.count = 0

    def add(self, fact: Mapping[str, object]) -> None:
        if not self._REQUIRED.issubset(fact):
            raise MetricsError('account facts missing required fields')
        ts = fact['ts_ns']
        if isinstance(ts, bool) or not isinstance(ts, int) or ts <= self.last_ts:
            raise MetricsError('account time order invalid')
        if self.last_ts != -1 and ts - self.last_ts != self.sample_interval_ns:
            raise MetricsError('irregular account sampling; annualization would be invalid')
        row = {key: _number(fact[key], key) for key in self._REQUIRED if key != 'ts_ns'}
        if (row['equity'] <= 0 or row['turnover_notional'] < 0 or row['fee'] < 0
                or (row['slippage_cost'] < 0 and not self.allow_price_improvement)):
            raise MetricsError('invalid account values / exhausted equity')
        if self.previous is None:
            self.first_equity = row['equity']
        else:
            adjusted = (row['equity'] - row['external_cashflow']) / self.previous['equity'] - 1
            if not math.isfinite(adjusted):
                raise MetricsError('cashflow-adjusted period return nonfinite')
            if (self.max_return_count is not None
                    and len(self.returns) >= self.max_return_count):
                raise MetricsError('account facts exceed configured maximum return count')
            turnover_increment = _number(
                row['turnover_notional'] / self.previous['equity'], 'turnover increment',
            )
            next_turnover = _number(self.turnover + turnover_increment, 'turnover aggregate')
            next_totals = {
                field: _number(self.totals[field] + row[field], f'{field} aggregate')
                for field in self.totals
            }
            next_external_cashflow = _number(
                self.external_cashflow + row['external_cashflow'],
                'external cashflow aggregate',
            )
            # Validate every derived value before appending to the spool or
            # advancing any of the accumulator's public state.
            self.returns.append(adjusted)
            self.turnover = next_turnover
            self.totals = next_totals
            self.external_cashflow = next_external_cashflow
        self.previous = row
        self.last_equity = row['equity']
        self.last_ts = ts
        self.count += 1

    def result(self, *, periods_per_year: float,
               realized_trade_pnls: Sequence[float] | None = None) -> dict:
        if self.count < 3:
            raise MetricsError('regular account snapshots and explicit sampling interval required')
        result: dict[str, object] = series_metrics(self.returns, periods_per_year=periods_per_year)
        result['turnover_ratio'] = self.turnover
        for field, value in self.totals.items():
            result[f'total_{field}'] = value
        gross_return = (
            self.last_equity - self.first_equity - self.external_cashflow
            + self.totals['fee'] + self.totals['slippage_cost'] - self.totals['funding']
        ) / self.first_equity
        result['gross_return_before_reported_costs'] = _number(
            gross_return, 'gross return before reported costs',
        )
        if realized_trade_pnls is not None:
            pnl = [_number(x, 'realized_pnl') for x in realized_trade_pnls]
            gross_profit = _number(sum(x for x in pnl if x > 0), 'gross profit aggregate')
            gross_loss = _number(-sum(x for x in pnl if x < 0), 'gross loss aggregate')
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
            result['trade_profit_factor'] = (
                _number(profit_factor, 'trade profit factor')
                if profit_factor is not None else None
            )
            result['closed_trade_count'] = len(pnl)
        return result


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
