"""Economic-only metrics rebuilt from native account/fill facts; never from shaped reward."""
from __future__ import annotations

import math
from array import array
from heapq import heappop, heappush, heapreplace
from collections.abc import Iterable, Sequence, Mapping
from .numeric import finite_real


class MetricsError(ValueError):
    pass


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


class _StreamingSeriesMetrics:
    """Incremental exact economic metrics with a bounded lower-tail heap."""

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
        self.worst_heap: list[float] = []
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
        if len(self.worst_heap) < self.tail_capacity:
            heappush(self.worst_heap, -x)
        elif self.worst_heap and x < -self.worst_heap[0]:
            heapreplace(self.worst_heap, -x)
        self.path *= 1 + x
        if not math.isfinite(self.path) or self.path <= 0:
            raise MetricsError("equity compounding overflow/underflow")
        self.peak = max(self.peak, self.path)
        self.drawdown = max(self.drawdown, 1 - self.path / self.peak)
        self.count = index

    def result(self, *, periods_per_year: float,
               require_exact_count: bool = False) -> dict[str, object]:
        count = self.count
        if (count < 2 or not finite_real(periods_per_year)
                or periods_per_year <= 0
                or (require_exact_count and count != self.max_count)):
            raise MetricsError("at least 2 returns and valid frequency/quantile required")
        tail_count = max(1, math.ceil(count * self.tail_quantile))
        if len(self.worst_heap) < tail_count:
            raise MetricsError("return tail does not cover the requested quantile")
        # max_count may exceed completed rows when duplicate barriers collapse
        # into one sample. Copy before trimming so result() can be called during
        # accumulation without discarding candidates needed by later samples.
        tail = list(self.worst_heap)
        while len(tail) > tail_count:
            heappop(tail)
        sd = self.m2_scale * math.sqrt(self.m2_ssq / (count - 1))
        downside_deviation = self.downside_scale * math.sqrt(self.downside_ssq / count)
        if not math.isfinite(sd) or not math.isfinite(downside_deviation):
            raise MetricsError("return dispersion overflow")
        sortino = ((self.mean - self.target_per_period) / downside_deviation
                   * math.sqrt(periods_per_year) if downside_deviation > 0 else None)
        sharpe = ((self.mean - self.risk_free_per_period) / sd
                  * math.sqrt(periods_per_year) if sd > 0 else None)
        return {
            "net_return": self.path - 1,
            "max_drawdown": self.drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "mean_period_return": self.mean,
            "std_period_return": sd,
            "undefined_metrics": [metric for metric, val in (("sharpe", sharpe), ("sortino", sortino))
                                  if val is None],
            "value_at_risk": tail[0],
            "conditional_value_at_risk": math.fsum(tail) / len(tail),
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
            if self._return_metrics is None:
                assert self.returns is not None
                self.returns.append(adjusted)
            else:
                self._return_metrics.add(adjusted)
            self.turnover += row['turnover_notional'] / self.previous['equity']
            for field in self.totals:
                self.totals[field] += row[field]
            self.external_cashflow += row['external_cashflow']
        self.previous = row
        self.last_equity = row['equity']
        self.last_ts = ts
        self.count += 1

    def result(self, *, periods_per_year: float,
               realized_trade_pnls: Sequence[float] | None = None) -> dict:
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
        result['gross_return_before_reported_costs'] = (
            self.last_equity - self.first_equity - self.external_cashflow
            + self.totals['fee'] + self.totals['slippage_cost'] - self.totals['funding']
        ) / self.first_equity
        if realized_trade_pnls is not None:
            pnl = [_number(x, 'realized_pnl') for x in realized_trade_pnls]
            gross_profit = sum(x for x in pnl if x > 0)
            gross_loss = -sum(x for x in pnl if x < 0)
            result['trade_profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else None
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
