"""Reproduce the deterministic memory/time comparison for return metrics."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
from typing import Any, overload

from rles_research.metrics import series_metrics


class SyntheticReturns(Sequence[float]):
    """A lazy, deterministic zero-drift sequence with no preallocated data."""

    def __init__(self, count: int) -> None:
        self._size = count

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[float]: ...

    def __getitem__(self, index: int | slice) -> float | Sequence[float]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            return tuple(self[position] for position in range(start, stop, step))
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(index)
        return ((index % 101) - 50) * 1e-7


def _measure(call: Any, repeats: int) -> tuple[Any, int, float]:
    tracemalloc.start()
    try:
        result = call()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        timings.append(time.perf_counter() - start)
    return result, peak, statistics.median(timings)


def _legacy_metrics(values: Sequence[float]) -> dict[str, float]:
    """Represent the previous list/downside-list/full-sort implementation."""
    returns = [float(value) for value in values]
    path = peak = 1.0
    drawdown = 0.0
    for value in returns:
        path *= 1 + value
        peak = max(peak, path)
        drawdown = max(drawdown, 1 - path / peak)
    downside = [min(0, value) ** 2 for value in returns]
    tail_count = max(1, math.ceil(len(returns) * 0.05))
    worst = sorted(returns)[:tail_count]
    mean = statistics.fmean(returns)
    deviation = statistics.stdev(returns)
    downside_deviation = math.sqrt(statistics.fmean(downside))
    return {
        "net_return": path - 1,
        "max_drawdown": drawdown,
        "mean_period_return": mean,
        "std_period_return": deviation,
        "sharpe": mean / deviation * math.sqrt(525600),
        "sortino": mean / downside_deviation * math.sqrt(525600),
        "value_at_risk": -worst[-1],
        "conditional_value_at_risk": -statistics.fmean(worst),
    }


def run(sizes: Sequence[int], repeats: int) -> dict[str, object]:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeat count must be a positive integer")
    cases = []
    for count in sizes:
        if type(count) is not int or count < 3:
            raise ValueError("each sample size must be an integer >= 3")
        values = SyntheticReturns(count)
        actual, actual_peak, actual_seconds = _measure(
            lambda: series_metrics(values, periods_per_year=525600), repeats,
        )
        expected, legacy_peak, legacy_seconds = _measure(lambda: _legacy_metrics(values), repeats)
        for name, reference in expected.items():
            observed = actual[name]
            if not isinstance(observed, (int, float)) or not math.isclose(
                    observed, reference, rel_tol=2e-13, abs_tol=1e-14):
                raise AssertionError(f"{name} differs at n={count}: {observed} != {reference}")
        cases.append({
            "observations": count,
            "one_pass_peak_python_bytes": actual_peak,
            "legacy_peak_python_bytes": legacy_peak,
            "peak_ratio": actual_peak / legacy_peak,
            "one_pass_seconds": actual_seconds,
            "legacy_seconds": legacy_seconds,
            "time_ratio": actual_seconds / legacy_seconds,
            "metric_values_match": True,
            "timing_repeats": repeats,
        })
    return {
        "schema": "rles.series-metrics-memory-profile.v1",
        "python": platform.python_version(),
        "sequence": "lazy deterministic ((index % 101) - 50) * 1e-7",
        "comparison": "current one-pass metrics vs previous returns/downside/full-sort implementation",
        "measurement": "tracemalloc peak is captured separately from elapsed wall time",
        "limitations": [
            "synthetic metric-only profile; not Nautilus end-to-end RSS",
            "does not include the upstream AccountFactsAccumulator return vector or factory snapshots",
        ],
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1440, 43200, 1_000_000])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.sizes, args.repeat)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
