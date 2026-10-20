"""Validated immutable 1-minute OHLCV records, independent from existing RLES DTOs.

The receiving integration must adapt existing MarketFrame instances through `as_candle`;
it must not modify, duplicate or shadow the existing rles_world_model.contracts source.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol

MINUTE_NS = 60_000_000_000


class FrameLike(Protocol):
    ts_close_ns: int
    available_at_ns: int
    instrument: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Candle:
    ts_close_ns: int
    available_at_ns: int
    instrument: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if type(self.ts_close_ns) is not int or self.ts_close_ns <= 0 or self.ts_close_ns % MINUTE_NS:
            raise ValueError("1m candle close_ns must be positive and UTC minute aligned")
        if type(self.available_at_ns) is not int or self.available_at_ns < self.ts_close_ns:
            raise ValueError("bar cannot be visible before its close")
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise ValueError("instrument required")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
            raise ValueError("OHLCV must be finite")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError("positive price and nonnegative volume required")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("invalid OHLC geometry")


def as_candle(frame: FrameLike) -> Candle:
    return Candle(frame.ts_close_ns, frame.available_at_ns, frame.instrument,
                  frame.open, frame.high, frame.low, frame.close, frame.volume)


@dataclass(frozen=True, slots=True)
class MarketSequence:
    candles: tuple[Candle, ...]
    dataset_id: str
    scope: str

    def __post_init__(self) -> None:
        if self.scope not in {"train", "synthetic_test_only", "validation", "test", "holdout", "inference_visible"}:
            raise ValueError("unknown market research scope")
        if not self.dataset_id or not self.candles:
            raise ValueError("dataset identity and candles required")
        first = self.candles[0]
        for a, b in zip(self.candles, self.candles[1:]):
            if b.ts_close_ns - a.ts_close_ns != MINUTE_NS:
                raise ValueError("missing/duplicated/out-of-order minute candle")
            if b.available_at_ns < a.available_at_ns:
                raise ValueError("data availability regressed")
            if b.instrument != first.instrument:
                raise ValueError("mixed market instruments")

    @classmethod
    def from_frames(cls, frames: Iterable[FrameLike], *, dataset_id: str,
                    scope: str) -> MarketSequence:
        return cls(tuple(as_candle(f) for f in frames), dataset_id, scope)

    def as_of(self, decision_ts_ns: int) -> tuple[Candle, ...]:
        if not isinstance(decision_ts_ns, int) or decision_ts_ns <= 0:
            raise ValueError("invalid decision clock")
        return tuple(f for f in self.candles
                     if f.ts_close_ns <= decision_ts_ns and f.available_at_ns <= decision_ts_ns)
