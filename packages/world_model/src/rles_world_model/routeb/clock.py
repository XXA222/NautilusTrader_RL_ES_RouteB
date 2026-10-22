"""Strictly causal UTC multi-resolution close/availability clock.

A high-timeframe bar becomes visible only when *all* of its constituent minute bars
are available. Does not synthesize partially closed high-timeframe observations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .market import Candle, MINUTE_NS


@dataclass(frozen=True, slots=True)
class ClosedBucket:
    timeframe_minutes: int
    close_ns: int
    available_at_ns: int
    instrument: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MultiScaleClock:
    def __init__(self, bars: Iterable[Candle]):
        candles = tuple(bars)
        if not candles:
            raise ValueError("empty clock")
        self.bars = candles
        for a,b in zip(candles, candles[1:]):
            if (b.ts_close_ns - a.ts_close_ns != MINUTE_NS or a.instrument != b.instrument
                or b.available_at_ns < a.available_at_ns):
                raise ValueError("clock requires one contiguous instrument")

    @staticmethod
    def bucket_close(ts_close_ns: int, timeframe_minutes: int) -> int:
        if timeframe_minutes not in (1, 60, 1440):
            raise ValueError("timeframe must be 1, 60, or 1440")
        span = timeframe_minutes * MINUTE_NS
        return ((ts_close_ns - 1) // span + 1) * span

    def buckets(self, timeframe_minutes: int, *, as_of_ns: int) -> tuple[ClosedBucket, ...]:
        if timeframe_minutes not in (1, 60, 1440):
            raise ValueError("invalid timeframe")
        if not isinstance(as_of_ns, int) or as_of_ns <= 0:
            raise ValueError("invalid time barrier")
        n = timeframe_minutes
        buckets: dict[int, list[Candle]] = {}
        for bar in self.bars:
            end = self.bucket_close(bar.ts_close_ns, n)
            if end <= as_of_ns:
                buckets.setdefault(end, []).append(bar)
        out=[]
        for end, items in sorted(buckets.items()):
            if (len(items) != n or items[0].ts_close_ns != end - (n - 1) * MINUTE_NS
                or items[-1].ts_close_ns != end or
                max(item.available_at_ns for item in items) > as_of_ns):
                continue
            out.append(ClosedBucket(n, end, max(x.available_at_ns for x in items),
                                    items[0].instrument, items[0].open,
                                    max(x.high for x in items), min(x.low for x in items),
                                    items[-1].close, sum(x.volume for x in items)))
        return tuple(out)

    def event_timeline(self, *, as_of_ns: int) -> dict[int, tuple[ClosedBucket, ...]]:
        """Availability-time events; times may be delayed relative to bar closes."""
        by_time: dict[int,list[ClosedBucket]]={}
        for tf in (1,60,1440):
            for b in self.buckets(tf,as_of_ns=as_of_ns):
                by_time.setdefault(b.available_at_ns,[]).append(b)
        return {ts: tuple(sorted(b, key=lambda x: -x.timeframe_minutes))
                for ts,b in sorted(by_time.items())}

    def latest(self, timeframe_minutes: int, decision_ns: int) -> ClosedBucket | None:
        bs = self.buckets(timeframe_minutes, as_of_ns=decision_ns)
        return bs[-1] if bs else None
